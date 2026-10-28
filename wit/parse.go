// Package wit reads WIT (the WebAssembly Interface Types language,
// component-model design/mvp/WIT.md) for the subset the wasi-gfx packages
// use, and implements the Canonical ABI (design/mvp/CanonicalABI.md) over a
// core module's linear memory: alignment, sizes, flattening, loads, stores,
// flat lifting and lowering. A core-module host serves WIT-described
// imports with it, without a component-model runtime.
package wit

import (
	"fmt"
	"strings"
	"unicode"
)

// Kind is a type's constructor.
type Kind int

// Type kinds (CanonicalABI.md value types).
const (
	Bool Kind = iota
	U8
	U16
	U32
	U64
	S8
	S16
	S32
	S64
	F32
	F64
	Char
	String
	List
	Record
	Tuple
	Variant
	Enum
	Flags
	Option
	Result
	Own
	Borrow
	Future
	Stream
	Named // an unresolved reference, replaced by Resolve
)

// Type is a WIT value type.
type Type struct {
	Kind   Kind
	Name   string  // type name for named types, resource name for Own/Borrow
	Elem   *Type   // List, Option, Future, Stream; Result's ok
	Err    *Type   // Result's error
	Fields []Field // Record, Tuple (labels "0", "1", ...)
	Cases  []Case  // Variant
	Labels []string
}

// Field is a record field.
type Field struct {
	Name string
	Type *Type
}

// Case is a variant case; Type is nil for a case without payload.
type Case struct {
	Name string
	Type *Type
}

// Param is a function parameter.
type Param struct {
	Name string
	Type *Type
}

// Func is a WIT function; for a method, Params[0] is "self".
type Func struct {
	Name   string // as in WIT: "create-buffer", or "" for a constructor
	Kind   FuncKind
	Async  bool
	Params []Param
	Result *Type // nil: no result
}

// FuncKind says where a function is declared.
type FuncKind int

// Function kinds and their core import names (wit-parser's mangling).
const (
	Freestanding FuncKind = iota // "name"
	Method                       // "[method]resource.name"
	Static                       // "[static]resource.name"
	Constructor                  // "[constructor]resource"
)

// Resource is a resource type and its functions.
type Resource struct {
	Name  string
	Funcs []*Func
}

// Interface is a WIT interface: its named types, resources and functions.
type Interface struct {
	Package   string // "wasi:webgpu@0.3.0-rc.2"
	Name      string
	Types     map[string]*Type
	Resources map[string]*Resource
	Funcs     []*Func
	order     []string // resource declaration order
}

// ImportModule is the core import module name, "ns:pkg/iface@ver".
func (i *Interface) ImportModule() string {
	ns, ver, _ := strings.Cut(i.Package, "@")
	if ver != "" {
		return ns + "/" + i.Name + "@" + ver
	}
	return ns + "/" + i.Name
}

// ImportName is a function's core import name.
func (f *Func) ImportName(resource string) string {
	switch f.Kind {
	case Method:
		return "[method]" + resource + "." + f.Name
	case Static:
		return "[static]" + resource + "." + f.Name
	case Constructor:
		return "[constructor]" + resource
	}
	return f.Name
}

// ResourceNames lists the resources in declaration order.
func (i *Interface) ResourceNames() []string { return i.order }

// parser is a recursive-descent parser over tokens.
type parser struct {
	toks []string
	pos  int
	err  error
}

func tokenize(src string) ([]string, error) {
	var toks []string
	r := []rune(src)
	for i := 0; i < len(r); {
		c := r[i]
		switch {
		case unicode.IsSpace(c):
			i++
		case c == '/' && i+1 < len(r) && r[i+1] == '/':
			for i < len(r) && r[i] != '\n' {
				i++
			}
		case c == '/' && i+1 < len(r) && r[i+1] == '*':
			end := strings.Index(string(r[i+2:]), "*/")
			if end < 0 {
				return nil, fmt.Errorf("unterminated comment")
			}
			i += 2 + len([]rune(string(r[i+2:])[:end])) + 2
		case c == '-' && i+1 < len(r) && r[i+1] == '>':
			toks = append(toks, "->")
			i += 2
		case strings.ContainsRune("{}()<>,:;=.*/", c):
			toks = append(toks, string(c))
			i++
		case c == '%' || c == '@' || unicode.IsLetter(c) || unicode.IsDigit(c) || c == '_':
			j := i
			for j < len(r) && (r[j] == '%' || r[j] == '@' || r[j] == '-' || r[j] == '_' || r[j] == '.' && i < j && unicode.IsDigit(r[j-1]) ||
				unicode.IsLetter(r[j]) || unicode.IsDigit(r[j])) {
				j++
			}
			toks = append(toks, string(r[i:j]))
			i = j
		default:
			return nil, fmt.Errorf("unexpected character %q", c)
		}
	}
	return toks, nil
}

func (p *parser) peek() string {
	if p.pos < len(p.toks) {
		return p.toks[p.pos]
	}
	return ""
}

func (p *parser) next() string {
	t := p.peek()
	if p.pos < len(p.toks) {
		p.pos++
	}
	return t
}

func (p *parser) expect(t string) {
	if p.err != nil {
		return
	}
	if got := p.next(); got != t {
		p.err = fmt.Errorf("expected %q, got %q (token %d)", t, got, p.pos)
	}
}

// ident reads an identifier, dropping a keyword escape "%".
func (p *parser) ident() string {
	t := p.next()
	if p.err == nil && (t == "" || strings.ContainsAny(t[:1], "{}()<>,:;=.*/")) {
		p.err = fmt.Errorf("expected an identifier, got %q (token %d)", t, p.pos)
	}
	return strings.TrimPrefix(t, "%")
}

// Parse reads WIT source holding one package and its interfaces.
func Parse(src string) (map[string]*Interface, error) {
	toks, err := tokenize(src)
	if err != nil {
		return nil, err
	}
	p := &parser{toks: toks}
	out := map[string]*Interface{}
	pkg := ""
	for p.err == nil && p.peek() != "" {
		switch p.next() {
		case "package":
			pkg = p.qualified()
			p.expect(";")
		case "interface":
			i := &Interface{Package: pkg, Name: p.ident(), Types: map[string]*Type{}, Resources: map[string]*Resource{}}
			p.expect("{")
			for p.err == nil && p.peek() != "}" {
				p.item(i)
			}
			p.expect("}")
			out[i.Name] = i
		case "world":
			p.ident()
			p.skipBlock()
		default:
			p.err = fmt.Errorf("unexpected %q at top level (token %d)", p.toks[p.pos-1], p.pos)
		}
	}
	if p.err != nil {
		return nil, fmt.Errorf("wit: %w", p.err)
	}
	for _, i := range out {
		if err := i.resolve(); err != nil {
			return nil, fmt.Errorf("wit: interface %s: %w", i.Name, err)
		}
	}
	return out, nil
}

// qualified reads "ns:name@ver" or "ns:name/iface@ver" as one string.
func (p *parser) qualified() string {
	var b strings.Builder
	for p.err == nil {
		t := p.peek()
		if t == ";" || t == "{" || t == "" {
			break
		}
		b.WriteString(p.next())
	}
	return b.String()
}

func (p *parser) skipBlock() {
	p.expect("{")
	depth := 1
	for p.err == nil && depth > 0 {
		switch p.next() {
		case "{":
			depth++
		case "}":
			depth--
		case "":
			p.err = fmt.Errorf("unterminated block")
		}
	}
}

func (p *parser) item(i *Interface) {
	switch t := p.peek(); t {
	case "record":
		p.next()
		name := p.ident()
		ty := &Type{Kind: Record}
		p.expect("{")
		for p.err == nil && p.peek() != "}" {
			f := Field{Name: p.ident()}
			p.expect(":")
			f.Type = p.typ()
			ty.Fields = append(ty.Fields, f)
			if p.peek() == "," {
				p.next()
			}
		}
		p.expect("}")
		i.Types[name] = ty
	case "variant":
		p.next()
		name := p.ident()
		ty := &Type{Kind: Variant}
		p.expect("{")
		for p.err == nil && p.peek() != "}" {
			c := Case{Name: p.ident()}
			if p.peek() == "(" {
				p.next()
				c.Type = p.typ()
				p.expect(")")
			}
			ty.Cases = append(ty.Cases, c)
			if p.peek() == "," {
				p.next()
			}
		}
		p.expect("}")
		i.Types[name] = ty
	case "enum", "flags":
		p.next()
		name := p.ident()
		k := Enum
		if t == "flags" {
			k = Flags
		}
		ty := &Type{Kind: k}
		p.expect("{")
		for p.err == nil && p.peek() != "}" {
			ty.Labels = append(ty.Labels, p.ident())
			if p.peek() == "," {
				p.next()
			}
		}
		p.expect("}")
		i.Types[name] = ty
	case "type":
		p.next()
		name := p.ident()
		p.expect("=")
		i.Types[name] = p.typ()
		p.expect(";")
	case "resource":
		p.next()
		r := &Resource{Name: p.ident()}
		i.Resources[r.Name] = r
		i.order = append(i.order, r.Name)
		if p.peek() == ";" {
			p.next()
			return
		}
		p.expect("{")
		for p.err == nil && p.peek() != "}" {
			if p.peek() == "constructor" {
				p.next()
				f := &Func{Kind: Constructor, Params: p.params(), Result: &Type{Kind: Own, Name: r.Name}}
				p.expect(";")
				r.Funcs = append(r.Funcs, f)
				continue
			}
			name := p.ident()
			p.expect(":")
			f := &Func{Name: name, Kind: Method}
			if p.peek() == "static" {
				p.next()
				f.Kind = Static
			}
			p.function(f)
			if f.Kind == Method {
				f.Params = append([]Param{{Name: "self", Type: &Type{Kind: Borrow, Name: r.Name}}}, f.Params...)
			}
			r.Funcs = append(r.Funcs, f)
		}
		p.expect("}")
	case "use":
		// Types from another interface: not needed by the packages read
		// here, whose uses name types this package resolves itself.
		for p.err == nil && p.peek() != ";" {
			p.next()
		}
		p.expect(";")
	default:
		f := &Func{Name: p.ident(), Kind: Freestanding}
		p.expect(":")
		p.function(f)
		i.Funcs = append(i.Funcs, f)
	}
}

// function reads "[async] func(params) [-> type];".
func (p *parser) function(f *Func) {
	if p.peek() == "async" {
		p.next()
		f.Async = true
	}
	p.expect("func")
	f.Params = p.params()
	if p.peek() == "->" {
		p.next()
		f.Result = p.typ()
	}
	p.expect(";")
}

func (p *parser) params() []Param {
	p.expect("(")
	var ps []Param
	for p.err == nil && p.peek() != ")" {
		pr := Param{Name: p.ident()}
		p.expect(":")
		pr.Type = p.typ()
		ps = append(ps, pr)
		if p.peek() == "," {
			p.next()
		}
	}
	p.expect(")")
	return ps
}

var prims = map[string]Kind{"bool": Bool, "u8": U8, "u16": U16, "u32": U32, "u64": U64,
	"s8": S8, "s16": S16, "s32": S32, "s64": S64, "f32": F32, "f64": F64, "float32": F32, "float64": F64,
	"char": Char, "string": String}

func (p *parser) typ() *Type {
	if p.err != nil {
		return &Type{Kind: Bool}
	}
	name := p.ident()
	if k, ok := prims[name]; ok {
		return &Type{Kind: k}
	}
	switch name {
	case "list", "option", "future", "stream":
		k := map[string]Kind{"list": List, "option": Option, "future": Future, "stream": Stream}[name]
		t := &Type{Kind: k}
		if (k == Future || k == Stream) && p.peek() != "<" {
			return t
		}
		p.expect("<")
		t.Elem = p.typ()
		p.expect(">")
		return t
	case "result":
		t := &Type{Kind: Result}
		if p.peek() != "<" {
			return t
		}
		p.next()
		if p.peek() == "_" {
			p.next()
		} else {
			t.Elem = p.typ()
		}
		if p.peek() == "," {
			p.next()
			t.Err = p.typ()
		}
		p.expect(">")
		return t
	case "tuple":
		t := &Type{Kind: Tuple}
		p.expect("<")
		for i := 0; p.err == nil && p.peek() != ">"; i++ {
			t.Fields = append(t.Fields, Field{Name: fmt.Sprint(i), Type: p.typ()})
			if p.peek() == "," {
				p.next()
			}
		}
		p.expect(">")
		return t
	case "borrow", "own":
		t := &Type{Kind: Borrow}
		if name == "own" {
			t.Kind = Own
		}
		p.expect("<")
		t.Name = p.ident()
		p.expect(">")
		return t
	}
	return &Type{Kind: Named, Name: name}
}

// resolve replaces named references by their definitions (resources by
// own handles), detecting unknown names and cycles.
func (i *Interface) resolve() error {
	var walk func(t *Type, seen map[string]bool) error
	walk = func(t *Type, seen map[string]bool) error {
		if t == nil {
			return nil
		}
		if t.Kind == Named {
			if _, ok := i.Resources[t.Name]; ok {
				*t = Type{Kind: Own, Name: t.Name}
				return nil
			}
			def, ok := i.Types[t.Name]
			if !ok {
				return fmt.Errorf("unknown type %q", t.Name)
			}
			if seen[t.Name] {
				return fmt.Errorf("type %q refers to itself", t.Name)
			}
			seen[t.Name] = true
			if err := walk(def, seen); err != nil {
				return err
			}
			delete(seen, t.Name)
			name := t.Name
			*t = *def
			if t.Name == "" {
				t.Name = name
			}
			return nil
		}
		if (t.Kind == Own || t.Kind == Borrow) && i.Resources[t.Name] == nil {
			return fmt.Errorf("unknown resource %q", t.Name)
		}
		for _, c := range []*Type{t.Elem, t.Err} {
			if err := walk(c, seen); err != nil {
				return err
			}
		}
		for _, f := range t.Fields {
			if err := walk(f.Type, seen); err != nil {
				return err
			}
		}
		for _, c := range t.Cases {
			if err := walk(c.Type, seen); err != nil {
				return err
			}
		}
		return nil
	}
	for name, t := range i.Types {
		if err := walk(t, map[string]bool{name: true}); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	funcs := append([]*Func(nil), i.Funcs...)
	for _, r := range i.Resources {
		funcs = append(funcs, r.Funcs...)
	}
	for _, f := range funcs {
		for _, p := range f.Params {
			if err := walk(p.Type, map[string]bool{}); err != nil {
				return fmt.Errorf("%s: %w", f.Name, err)
			}
		}
		if err := walk(f.Result, map[string]bool{}); err != nil {
			return fmt.Errorf("%s: %w", f.Name, err)
		}
	}
	return nil
}
