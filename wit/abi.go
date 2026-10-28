package wit

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"unicode/utf8"
)

// The Canonical ABI for a 32-bit memory (CanonicalABI.md: alignment,
// elem_size, flatten_type, load, store, lift_flat, lower_flat), with the
// spec's traps as errors.

// Core value types, as the engine's host modules spell them.
const (
	I32  = 'i'
	I64  = 'I'
	F32c = 'f'
	F64c = 'F'
)

// Limits of the ABI (MAX_FLAT_PARAMS, MAX_FLAT_RESULTS).
const (
	MaxFlatParams  = 16
	MaxFlatResults = 1
	// MaxListBytes bounds one list or string a guest passes or receives.
	MaxListBytes = 1 << 28
)

// Values: bool; uint64 (u8..u64); int64 (s8..s64); float32; float64; rune
// (char); string; []any (list, record and tuple fields in order);
// VariantValue (variant); uint32 (enum case index, flags bits, handles);
// nil or Some (option); ResultValue (result).

// VariantValue is a variant case and its payload (nil without one).
type VariantValue struct {
	Case  int
	Value any
}

// Some is a present option value.
type Some struct{ Value any }

// ResultValue is a result: Value is the ok or the error payload.
type ResultValue struct {
	Err   bool
	Value any
}

// Memory is a guest's linear memory.
type Memory interface {
	Read(off, n uint32) ([]byte, bool)
	Write(off uint32, b []byte) bool
}

// ErrTrap wraps every error that the Canonical ABI defines as a trap.
var ErrTrap = errors.New("canonical ABI trap")

func trap(format string, a ...any) error {
	return fmt.Errorf("%w: %s", ErrTrap, fmt.Sprintf(format, a...))
}

func discSize(n int) uint32 {
	switch {
	case n <= 1<<8:
		return 1
	case n <= 1<<16:
		return 2
	}
	return 4
}

func flagsSize(n int) uint32 {
	switch {
	case n <= 8:
		return 1
	case n <= 16:
		return 2
	}
	return 4
}

// cases are a type's variant cases after despecialization.
func cases(t *Type) []Case {
	switch t.Kind {
	case Variant:
		return t.Cases
	case Enum:
		c := make([]Case, len(t.Labels))
		for i, l := range t.Labels {
			c[i] = Case{Name: l}
		}
		return c
	case Option:
		return []Case{{Name: "none"}, {Name: "some", Type: t.Elem}}
	case Result:
		return []Case{{Name: "ok", Type: t.Elem}, {Name: "error", Type: t.Err}}
	}
	return nil
}

func isVariant(k Kind) bool { return k == Variant || k == Enum || k == Option || k == Result }

func alignTo(p, a uint32) uint32 { return (p + a - 1) / a * a }

// Align is the type's alignment.
func Align(t *Type) uint32 {
	switch t.Kind {
	case Bool, U8, S8:
		return 1
	case U16, S16:
		return 2
	case U32, S32, F32, Char, String, List, Own, Borrow, Future, Stream:
		return 4
	case U64, S64, F64:
		return 8
	case Record, Tuple:
		a := uint32(1)
		for _, f := range t.Fields {
			a = max(a, Align(f.Type))
		}
		return a
	case Flags:
		return flagsSize(len(t.Labels))
	}
	cs := cases(t)
	return max(discSize(len(cs)), maxCaseAlign(cs))
}

func maxCaseAlign(cs []Case) uint32 {
	a := uint32(1)
	for _, c := range cs {
		if c.Type != nil {
			a = max(a, Align(c.Type))
		}
	}
	return a
}

// Size is the type's elem_size.
func Size(t *Type) uint32 {
	switch t.Kind {
	case Bool, U8, S8:
		return 1
	case U16, S16:
		return 2
	case U32, S32, F32, Char, Own, Borrow, Future, Stream:
		return 4
	case U64, S64, F64, String, List:
		return 8
	case Record, Tuple:
		s := uint32(0)
		for _, f := range t.Fields {
			s = alignTo(s, Align(f.Type))
			s += Size(f.Type)
		}
		return alignTo(s, Align(t))
	case Flags:
		return flagsSize(len(t.Labels))
	}
	cs := cases(t)
	s := alignTo(discSize(len(cs)), maxCaseAlign(cs))
	var payload uint32
	for _, c := range cs {
		if c.Type != nil {
			payload = max(payload, Size(c.Type))
		}
	}
	return alignTo(s+payload, Align(t))
}

// Flatten is flatten_type: the core value types of t.
func Flatten(t *Type) []byte {
	switch t.Kind {
	case Bool, U8, U16, U32, S8, S16, S32, Char, Flags, Own, Borrow, Future, Stream:
		return []byte{I32}
	case U64, S64:
		return []byte{I64}
	case F32:
		return []byte{F32c}
	case F64:
		return []byte{F64c}
	case String, List:
		return []byte{I32, I32}
	case Record, Tuple:
		var out []byte
		for _, f := range t.Fields {
			out = append(out, Flatten(f.Type)...)
		}
		return out
	}
	var flat []byte
	for _, c := range cases(t) {
		if c.Type == nil {
			continue
		}
		for i, ft := range Flatten(c.Type) {
			if i < len(flat) {
				flat[i] = join(flat[i], ft)
			} else {
				flat = append(flat, ft)
			}
		}
	}
	return append([]byte{I32}, flat...)
}

func join(a, b byte) byte {
	if a == b {
		return a
	}
	if (a == I32 && b == F32c) || (a == F32c && b == I32) {
		return I32
	}
	return I64
}

// CoreSig is a function's core signature when lowered synchronously
// (flatten_functype, 'lower'): params beyond MaxFlatParams go through one
// pointer; results beyond MaxFlatResults through a return-area pointer
// appended to the params.
type CoreSig struct {
	Params, Results []byte
	ParamsInMemory  bool
	RetPtr          bool
}

// Sig computes f's core signature.
func Sig(f *Func) CoreSig {
	var s CoreSig
	for _, p := range f.Params {
		s.Params = append(s.Params, Flatten(p.Type)...)
	}
	if len(s.Params) > MaxFlatParams {
		s.Params = []byte{I32}
		s.ParamsInMemory = true
	}
	if f.Result != nil {
		s.Results = Flatten(f.Result)
	}
	if len(s.Results) > MaxFlatResults {
		s.Params = append(s.Params, I32)
		s.Results = nil
		s.RetPtr = true
	}
	return s
}

// paramTuple is the tuple of f's parameter types.
func paramTuple(f *Func) *Type {
	t := &Type{Kind: Tuple}
	for i, p := range f.Params {
		t.Fields = append(t.Fields, Field{Name: fmt.Sprint(i), Type: p.Type})
	}
	return t
}

// --- loads -------------------------------------------------------------

func read(m Memory, p, n uint32) ([]byte, error) {
	b, ok := m.Read(p, n)
	if !ok {
		return nil, trap("%d bytes at %#x out of bounds", n, p)
	}
	return b, nil
}

func loadInt(m Memory, p, n uint32) (uint64, error) {
	b, err := read(m, p, n)
	if err != nil {
		return 0, err
	}
	switch n {
	case 1:
		return uint64(b[0]), nil
	case 2:
		return uint64(binary.LittleEndian.Uint16(b)), nil
	case 4:
		return uint64(binary.LittleEndian.Uint32(b)), nil
	}
	return binary.LittleEndian.Uint64(b), nil
}

// Load reads a value of type t at p (aligned).
func Load(m Memory, p uint32, t *Type) (any, error) {
	if p%Align(t) != 0 {
		return nil, trap("misaligned %v at %#x", t.Kind, p)
	}
	switch t.Kind {
	case Bool:
		v, err := loadInt(m, p, 1)
		return v != 0, err
	case U8, U16, U32, U64:
		return loadInt(m, p, Size(t))
	case S8, S16, S32, S64:
		v, err := loadInt(m, p, Size(t))
		return signExtend(v, Size(t)*8), err
	case F32:
		v, err := loadInt(m, p, 4)
		return math.Float32frombits(uint32(v)), err
	case F64:
		v, err := loadInt(m, p, 8)
		return math.Float64frombits(v), err
	case Char:
		v, err := loadInt(m, p, 4)
		if err != nil {
			return nil, err
		}
		return char(uint32(v))
	case String, List:
		ptr, err := loadInt(m, p, 4)
		if err != nil {
			return nil, err
		}
		n, err := loadInt(m, p+4, 4)
		if err != nil {
			return nil, err
		}
		if t.Kind == String {
			return loadString(m, uint32(ptr), uint32(n))
		}
		return loadList(m, uint32(ptr), uint32(n), t.Elem)
	case Record, Tuple:
		out := make([]any, len(t.Fields))
		q := p
		for i, f := range t.Fields {
			q = alignTo(q, Align(f.Type))
			v, err := Load(m, q, f.Type)
			if err != nil {
				return nil, err
			}
			out[i] = v
			q += Size(f.Type)
		}
		return out, nil
	case Flags:
		return loadInt(m, p, flagsSize(len(t.Labels)))
	case Own, Borrow:
		v, err := loadInt(m, p, 4)
		return uint32(v), err
	case Future, Stream:
		return nil, trap("future and stream values are not supported")
	}
	cs := cases(t)
	ds := discSize(len(cs))
	d, err := loadInt(m, p, ds)
	if err != nil {
		return nil, err
	}
	if d >= uint64(len(cs)) {
		return nil, trap("case %d of %d", d, len(cs))
	}
	var v any
	if c := cs[d]; c.Type != nil {
		if v, err = Load(m, alignTo(p+ds, maxCaseAlign(cs)), c.Type); err != nil {
			return nil, err
		}
	}
	return variantValue(t, int(d), v), nil
}

func signExtend(v uint64, bits uint32) int64 {
	shift := 64 - bits
	return int64(v<<shift) >> shift
}

func char(v uint32) (rune, error) {
	if v >= 0x110000 || (v >= 0xD800 && v <= 0xDFFF) {
		return 0, trap("invalid char %#x", v)
	}
	return rune(v), nil
}

func loadString(m Memory, ptr, n uint32) (string, error) {
	if n > MaxListBytes {
		return "", trap("string of %d bytes", n)
	}
	b, err := read(m, ptr, n)
	if err != nil {
		return "", err
	}
	if !utf8.Valid(b) {
		return "", trap("string is not UTF-8")
	}
	return string(b), nil
}

// loadList loads a list: []byte for list<u8> (bulk data, copied), []any
// otherwise.
func loadList(m Memory, ptr, n uint32, elem *Type) (any, error) {
	if elem.Kind == U8 {
		if n > MaxListBytes {
			return nil, trap("list of %d bytes", n)
		}
		b, err := read(m, ptr, n)
		if err != nil {
			return nil, err
		}
		return append([]byte(nil), b...), nil
	}
	es := Size(elem)
	if uint64(n)*uint64(es) > MaxListBytes {
		return nil, trap("list of %d elements of %d bytes", n, es)
	}
	if ptr%Align(elem) != 0 {
		return nil, trap("misaligned list at %#x", ptr)
	}
	if _, err := read(m, ptr, n*es); err != nil {
		return nil, err
	}
	out := make([]any, n)
	for i := range out {
		v, err := Load(m, ptr+uint32(i)*es, elem)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

// variantValue builds the Go value of case d of a variant-like type.
func variantValue(t *Type, d int, v any) any {
	switch t.Kind {
	case Enum:
		return uint32(d)
	case Option:
		if d == 0 {
			return nil
		}
		return Some{v}
	case Result:
		return ResultValue{Err: d == 1, Value: v}
	}
	return VariantValue{Case: d, Value: v}
}

// caseOf splits a variant-like value into its case and payload.
func caseOf(t *Type, v any) (int, any, error) {
	switch t.Kind {
	case Enum:
		d, ok := v.(uint32)
		if !ok || int(d) >= len(t.Labels) {
			return 0, nil, fmt.Errorf("enum value %v", v)
		}
		return int(d), nil, nil
	case Option:
		if v == nil {
			return 0, nil, nil
		}
		s, ok := v.(Some)
		if !ok {
			return 0, nil, fmt.Errorf("option value %T", v)
		}
		return 1, s.Value, nil
	case Result:
		r, ok := v.(ResultValue)
		if !ok {
			return 0, nil, fmt.Errorf("result value %T", v)
		}
		if r.Err {
			return 1, r.Value, nil
		}
		return 0, r.Value, nil
	}
	vv, ok := v.(VariantValue)
	if !ok || vv.Case < 0 || vv.Case >= len(t.Cases) {
		return 0, nil, fmt.Errorf("variant value %v", v)
	}
	return vv.Case, vv.Value, nil
}

// --- flat lifting ------------------------------------------------------

// flatIter yields core values, coercing as lift_flat_variant does.
type flatIter struct {
	vals []uint64
	i    int
}

func (it *flatIter) next() (uint64, error) {
	if it.i >= len(it.vals) {
		return 0, trap("too few core values")
	}
	v := it.vals[it.i]
	it.i++
	return v, nil
}

// ParamTuple is the tuple of f's parameter types (what goes to memory
// when the parameters do not fit flat).
func ParamTuple(f *Func) *Type { return paramTuple(f) }

// LiftFlat lifts one value of type t from its core values (a function's
// flat result, on the caller's side).
func LiftFlat(t *Type, vals []uint64, m Memory) (any, error) {
	return liftFlat(&flatIter{vals: vals}, t, m)
}

// LiftParams lifts f's parameters from its core arguments (loading the
// parameter tuple from memory when they did not fit flat).
func LiftParams(f *Func, args []uint64, m Memory) ([]any, error) {
	s := Sig(f)
	if s.ParamsInMemory {
		if len(args) < 1 {
			return nil, trap("no parameter pointer")
		}
		tt := paramTuple(f)
		ptr := uint32(args[0])
		if ptr%Align(tt) != 0 {
			return nil, trap("misaligned parameters at %#x", ptr)
		}
		v, err := Load(m, ptr, tt)
		if err != nil {
			return nil, err
		}
		return v.([]any), nil
	}
	it := &flatIter{vals: args}
	out := make([]any, len(f.Params))
	for i, p := range f.Params {
		v, err := liftFlat(it, p.Type, m)
		if err != nil {
			return nil, fmt.Errorf("parameter %s: %w", p.Name, err)
		}
		out[i] = v
	}
	return out, nil
}

func liftFlat(it *flatIter, t *Type, m Memory) (any, error) {
	switch t.Kind {
	case Bool:
		v, err := it.next()
		return uint32(v) != 0, err
	case U8, U16, U32:
		v, err := it.next()
		return uint64(uint32(v)) & (1<<(Size(t)*8) - 1), err
	case U64:
		return it.next()
	case S8, S16, S32:
		v, err := it.next()
		return signExtend(uint64(uint32(v)), Size(t)*8), err
	case S64:
		v, err := it.next()
		return int64(v), err
	case F32:
		v, err := it.next()
		return math.Float32frombits(uint32(v)), err
	case F64:
		v, err := it.next()
		return math.Float64frombits(v), err
	case Char:
		v, err := it.next()
		if err != nil {
			return nil, err
		}
		return char(uint32(v))
	case String, List:
		ptr, err := it.next()
		if err != nil {
			return nil, err
		}
		n, err := it.next()
		if err != nil {
			return nil, err
		}
		if t.Kind == String {
			return loadString(m, uint32(ptr), uint32(n))
		}
		return loadList(m, uint32(ptr), uint32(n), t.Elem)
	case Record, Tuple:
		out := make([]any, len(t.Fields))
		for i, f := range t.Fields {
			v, err := liftFlat(it, f.Type, m)
			if err != nil {
				return nil, err
			}
			out[i] = v
		}
		return out, nil
	case Flags:
		v, err := it.next()
		return uint64(uint32(v)), err
	case Own, Borrow:
		v, err := it.next()
		return uint32(v), err
	case Future, Stream:
		return nil, trap("future and stream values are not supported")
	}
	flat := Flatten(t)
	d, err := it.next()
	if err != nil {
		return nil, err
	}
	cs := cases(t)
	if uint32(d) >= uint32(len(cs)) {
		return nil, trap("case %d of %d", d, len(cs))
	}
	payload := flat[1:]
	sub := &flatIter{}
	for range payload {
		v, err := it.next()
		if err != nil {
			return nil, err
		}
		sub.vals = append(sub.vals, v)
	}
	var v any
	if c := cs[d]; c.Type != nil {
		// Coerce from the joined types to the case's own (wrap, bit-cast):
		// the core values hold bits, so only i64 -> i32 wrapping matters,
		// done by the 32-bit reads above.
		if v, err = liftFlat(sub, c.Type, m); err != nil {
			return nil, err
		}
	}
	return variantValue(t, int(d), v), nil
}
