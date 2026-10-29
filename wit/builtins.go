package wit

// The Canonical ABI built-ins a core module imports to use the streams a
// function returns, named as wit-component names them (wasm-tools,
// crates/wit-component/src/validation.rs; signatures from component-model
// design/mvp/CanonicalABI.md, "canon stream.read", "canon waitable-set.*").
// Host and guest take the names from here.

// CoreImport is a core import's name and core signature (I32 etc.).
type CoreImport struct{ Name, Params, Result string }

// Stream built-in kinds, in StreamBuiltins order.
const (
	StreamRead = iota
	StreamReadAsync
	StreamCancelRead
	StreamCancelReadAsync
	StreamDropReadable
)

// StreamBuiltins are the built-ins for the stream that function fn (its
// core import name) returns, imported from fn's module; the index 0 is
// that stream's position among the function's streams (its only one).
func StreamBuiltins(fn string) []CoreImport {
	return []CoreImport{
		StreamRead:            {"[stream-read-0]" + fn, "iii", "i"},
		StreamReadAsync:       {"[async-lower][stream-read-0]" + fn, "iii", "i"},
		StreamCancelRead:      {"[stream-cancel-read-0]" + fn, "i", "i"},
		StreamCancelReadAsync: {"[async-lower][stream-cancel-read-0]" + fn, "i", "i"},
		StreamDropReadable:    {"[stream-drop-readable-0]" + fn, "i", ""},
	}
}

// RootModule is the module of the waitable-set built-ins.
const RootModule = "$root"

// Waitable-set built-in kinds, in WaitableBuiltins order.
const (
	WaitableSetNew = iota
	WaitableSetDrop
	WaitableJoin
	WaitableSetPoll
	WaitableSetWait
)

// WaitableBuiltins are the $root module's waitable-set built-ins.
var WaitableBuiltins = []CoreImport{
	WaitableSetNew:  {"[waitable-set-new]", "", "i"},
	WaitableSetDrop: {"[waitable-set-drop]", "i", ""},
	WaitableJoin:    {"[waitable-join]", "ii", ""},
	WaitableSetPoll: {"[waitable-set-poll]", "ii", "i"},
	WaitableSetWait: {"[waitable-set-wait]", "ii", "i"},
}

// Canonical ABI values of those built-ins.
const (
	EventNone       = 0           // waitable-set.poll: nothing ready
	EventStreamRead = 2           // a stream read finished
	CopyCompleted   = 0           // payload low 4 bits
	CopyDropped     = 1           // the writer is gone
	CopyCancelled   = 2           // the read was cancelled
	Blocked         = 0xffff_ffff // an async read did not finish yet
)

// ReturnsStream reports whether f's result is a stream (whose built-ins
// StreamBuiltins names).
func ReturnsStream(f *Func) bool { return f.Result != nil && f.Result.Kind == Stream }
