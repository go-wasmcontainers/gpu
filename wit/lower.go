package wit

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Allocator gives a block of guest memory (the guest's cabi_realloc).
type Allocator interface {
	Alloc(align, size uint32) (uint32, error)
}

// Lowering a host result may need guest allocations (strings, lists),
// which a host obtains from the guest's cabi_realloc one at a time,
// asynchronously in the browser. So lowering runs twice: Plan walks the
// value with a memory that discards writes and records the allocations it
// would make; the host obtains them; Apply walks it again, identically,
// handing out the obtained pointers in the same order and writing.

// AllocReq is one allocation a lowering needs.
type AllocReq struct{ Align, Size uint32 }

type discard struct{}

func (discard) Read(off, n uint32) ([]byte, bool) { return make([]byte, n), true }
func (discard) Write(uint32, []byte) bool         { return true }

type recorder struct {
	reqs []AllocReq
	next uint32
}

func (r *recorder) Alloc(align, size uint32) (uint32, error) {
	r.reqs = append(r.reqs, AllocReq{align, size})
	// Distinct, aligned, never-written addresses: the plan discards writes.
	p := alignTo(r.next+8, max(align, 1))
	r.next = p + size
	return p, nil
}

type replay struct {
	ptrs []uint32
	reqs []AllocReq
	i    int
}

func (r *replay) Alloc(align, size uint32) (uint32, error) {
	if r.i >= len(r.ptrs) || r.reqs[r.i] != (AllocReq{align, size}) {
		return 0, fmt.Errorf("lowering replay diverged at allocation %d", r.i)
	}
	p := r.ptrs[r.i]
	r.i++
	return p, nil
}

// Lowering is a result being lowered: its allocations, then Apply.
type Lowering struct {
	f    *Func
	v    any
	ret  uint32 // the return area, when the result goes through memory
	Reqs []AllocReq

	elem  *Type // PlanElems: the element type, stored from ret on
	elems []any
}

// PlanElems prepares storing vs, each a t, one after another from p (a
// stream read's buffer: stream.read lowers its elements there).
func PlanElems(p uint32, t *Type, vs []any) (*Lowering, error) {
	if p%Align(t) != 0 {
		return nil, trap("misaligned buffer at %#x", p)
	}
	n := uint64(len(vs)) * uint64(Size(t))
	if n > MaxListBytes || uint64(p)+n > 1<<32 {
		// The second bound keeps p + i*size from wrapping around.
		return nil, trap("%d elements at %#x", len(vs), p)
	}
	l := &Lowering{ret: p, elem: t, elems: vs}
	rec := &recorder{}
	if _, err := l.lower(discard{}, rec); err != nil {
		return nil, err
	}
	l.Reqs = rec.reqs
	return l, nil
}

// PlanResult prepares lowering f's result v. args are the call's core
// arguments (the last is the return area when Sig(f).RetPtr).
func PlanResult(f *Func, v any, args []uint64) (*Lowering, error) {
	l := &Lowering{f: f, v: v}
	s := Sig(f)
	if s.RetPtr {
		if len(args) == 0 {
			return nil, trap("no return area")
		}
		l.ret = uint32(args[len(args)-1])
		if l.ret%Align(f.Result) != 0 {
			return nil, trap("misaligned return area at %#x", l.ret)
		}
	}
	rec := &recorder{}
	if _, err := l.lower(discard{}, rec); err != nil {
		return nil, err
	}
	l.Reqs = rec.reqs
	return l, nil
}

// Apply writes the result with the obtained pointers (one per Reqs, in
// order) and returns the flat result (0 when it went through memory, or
// when f has no result).
func (l *Lowering) Apply(m Memory, ptrs []uint32) (uint64, error) {
	for i, p := range ptrs {
		r := l.Reqs[i]
		if r.Align > 1 && p%r.Align != 0 {
			return 0, trap("allocation %d misaligned at %#x", i, p)
		}
		if _, ok := m.Read(p, r.Size); !ok {
			return 0, trap("allocation %d outside memory", i)
		}
	}
	return l.lower(m, &replay{ptrs: ptrs, reqs: l.Reqs})
}

func (l *Lowering) lower(m Memory, a Allocator) (uint64, error) {
	if l.elem != nil {
		sz := Size(l.elem)
		for i, v := range l.elems {
			if err := Store(m, a, l.ret+uint32(i)*sz, l.elem, v); err != nil {
				return 0, err
			}
		}
		return 0, nil
	}
	if l.f.Result == nil {
		return 0, nil
	}
	if Sig(l.f).RetPtr {
		return 0, Store(m, a, l.ret, l.f.Result, l.v)
	}
	flat, err := LowerFlat(m, a, l.f.Result, l.v)
	if err != nil {
		return 0, err
	}
	if len(flat) != 1 {
		return 0, fmt.Errorf("lowered to %d core values", len(flat))
	}
	return flat[0], nil
}

func write(m Memory, p uint32, b []byte) error {
	if !m.Write(p, b) {
		return trap("%d bytes at %#x out of bounds", len(b), p)
	}
	return nil
}

func storeInt(m Memory, p uint32, n uint32, v uint64) error {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	return write(m, p, b[:n])
}

// intOf reads any integer value as its bits.
func intOf(v any) (uint64, error) {
	switch x := v.(type) {
	case uint64:
		return x, nil
	case int64:
		return uint64(x), nil
	case uint32:
		return uint64(x), nil
	case int:
		return uint64(x), nil
	case bool:
		if x {
			return 1, nil
		}
		return 0, nil
	case rune:
		return uint64(uint32(x)), nil
	}
	return 0, fmt.Errorf("not an integer: %T", v)
}

// Store writes v of type t at p, allocating strings and lists with a.
func Store(m Memory, a Allocator, p uint32, t *Type, v any) error {
	switch t.Kind {
	case Bool, U8, U16, U32, U64, S8, S16, S32, S64, Char, Flags, Own, Borrow, Future, Stream:
		// A future or stream is lowered as the index of its readable end in
		// the handle table (lower_stream), which the host created.
		x, err := intOf(v)
		if err != nil {
			return err
		}
		return storeInt(m, p, Size(t), x)
	case F32:
		f, ok := v.(float32)
		if !ok {
			return fmt.Errorf("not an f32: %T", v)
		}
		return storeInt(m, p, 4, uint64(math.Float32bits(f)))
	case F64:
		f, ok := v.(float64)
		if !ok {
			return fmt.Errorf("not an f64: %T", v)
		}
		return storeInt(m, p, 8, math.Float64bits(f))
	case String, List:
		ptr, n, err := storeRange(m, a, t, v)
		if err != nil {
			return err
		}
		if err := storeInt(m, p, 4, uint64(ptr)); err != nil {
			return err
		}
		return storeInt(m, p+4, 4, uint64(n))
	case Record, Tuple:
		fs, ok := v.([]any)
		if !ok || len(fs) != len(t.Fields) {
			return fmt.Errorf("record value %T", v)
		}
		q := p
		for i, f := range t.Fields {
			q = alignTo(q, Align(f.Type))
			if err := Store(m, a, q, f.Type, fs[i]); err != nil {
				return err
			}
			q += Size(f.Type)
		}
		return nil
	}
	d, pv, err := caseOf(t, v)
	if err != nil {
		return err
	}
	cs := cases(t)
	ds := discSize(len(cs))
	if err := storeInt(m, p, ds, uint64(d)); err != nil {
		return err
	}
	if c := cs[d]; c.Type != nil {
		return Store(m, a, alignTo(p+ds, maxCaseAlign(cs)), c.Type, pv)
	}
	return nil
}

// storeRange allocates and writes a string's bytes or a list's elements.
func storeRange(m Memory, a Allocator, t *Type, v any) (uint32, uint32, error) {
	if t.Kind == String {
		s, ok := v.(string)
		if !ok {
			b, isBytes := v.([]byte)
			if !isBytes {
				return 0, 0, fmt.Errorf("not a string: %T", v)
			}
			s = string(b)
		}
		if len(s) > MaxListBytes {
			return 0, 0, fmt.Errorf("string of %d bytes", len(s))
		}
		if len(s) == 0 {
			return 1, 0, nil
		}
		p, err := a.Alloc(1, uint32(len(s)))
		if err != nil {
			return 0, 0, err
		}
		return p, uint32(len(s)), write(m, p, []byte(s))
	}
	// list<u8> may be given as []byte (bulk data).
	if b, ok := v.([]byte); ok && t.Elem.Kind == U8 {
		if len(b) > MaxListBytes {
			return 0, 0, fmt.Errorf("list of %d bytes", len(b))
		}
		if len(b) == 0 {
			return 1, 0, nil
		}
		p, err := a.Alloc(1, uint32(len(b)))
		if err != nil {
			return 0, 0, err
		}
		return p, uint32(len(b)), write(m, p, b)
	}
	es, ok := v.([]any)
	if !ok {
		return 0, 0, fmt.Errorf("not a list: %T", v)
	}
	sz, al := Size(t.Elem), Align(t.Elem)
	if uint64(len(es))*uint64(sz) > MaxListBytes {
		return 0, 0, fmt.Errorf("list of %d elements", len(es))
	}
	if len(es) == 0 {
		return al, 0, nil
	}
	p, err := a.Alloc(al, uint32(len(es))*sz)
	if err != nil {
		return 0, 0, err
	}
	for i, e := range es {
		if err := Store(m, a, p+uint32(i)*sz, t.Elem, e); err != nil {
			return 0, 0, err
		}
	}
	return p, uint32(len(es)), nil
}

// LowerFlat is lower_flat: v of type t as core values.
func LowerFlat(m Memory, a Allocator, t *Type, v any) ([]uint64, error) {
	switch t.Kind {
	case Bool, U8, U16, U32, Char, Flags, Own, Borrow, Future, Stream:
		x, err := intOf(v)
		return []uint64{uint64(uint32(x))}, err
	case S8, S16, S32:
		x, err := intOf(v)
		return []uint64{uint64(uint32(x))}, err
	case U64, S64:
		x, err := intOf(v)
		return []uint64{x}, err
	case F32:
		f, ok := v.(float32)
		if !ok {
			return nil, fmt.Errorf("not an f32: %T", v)
		}
		return []uint64{uint64(math.Float32bits(f))}, nil
	case F64:
		f, ok := v.(float64)
		if !ok {
			return nil, fmt.Errorf("not an f64: %T", v)
		}
		return []uint64{math.Float64bits(f)}, nil
	case String, List:
		p, n, err := storeRange(m, a, t, v)
		return []uint64{uint64(p), uint64(n)}, err
	case Record, Tuple:
		fs, ok := v.([]any)
		if !ok || len(fs) != len(t.Fields) {
			return nil, fmt.Errorf("record value %T", v)
		}
		var out []uint64
		for i, f := range t.Fields {
			x, err := LowerFlat(m, a, f.Type, fs[i])
			if err != nil {
				return nil, err
			}
			out = append(out, x...)
		}
		return out, nil
	}
	d, pv, err := caseOf(t, v)
	if err != nil {
		return nil, err
	}
	flat := Flatten(t)
	out := make([]uint64, len(flat))
	out[0] = uint64(d)
	if c := cases(t)[d]; c.Type != nil {
		payload, err := LowerFlat(m, a, c.Type, pv)
		if err != nil {
			return nil, err
		}
		// The joined slots hold the payload's bits (an f32 as its i32
		// bits, an i32 zero-extended into an i64).
		copy(out[1:], payload)
	}
	return out, nil
}
