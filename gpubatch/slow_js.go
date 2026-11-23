//go:build js

package gpubatch

import "syscall/js"

// slowKeep bounds self.__batchSlow, the browser's list of slow submits.
const slowKeep = 400

func init() {
	Created = func(n int) {
		g := js.Global()
		total := n
		if v := g.Get("__batchCreated"); !v.IsUndefined() {
			total += v.Int()
		}
		g.Set("__batchCreated", total)
	}
	SlowSubmit = func(i SubmitInfo) {
		l := js.Global().Get("__batchSlow")
		if l.IsUndefined() {
			l = js.Global().Get("Array").New()
			js.Global().Set("__batchSlow", l)
		}
		if l.Length() >= slowKeep {
			return
		}
		made := js.Global().Get("Array").New()
		for _, c := range i.Creations {
			made.Call("push", map[string]any{"kind": c.Kind, "bytes": c.Bytes, "ran": float64(c.Ran.Microseconds()) / 1000})
		}
		l.Call("push", map[string]any{
			"made":     made,
			"at":       js.Global().Get("performance").Call("now").Float() + js.Global().Get("performance").Get("timeOrigin").Float(),
			"ran":      float64(i.Ran.Microseconds()) / 1000,
			"waited":   float64(i.Waited.Microseconds()) / 1000,
			"bytes":    i.Bytes,
			"commands": i.Commands,
			"creates":  i.Creates,
		})
	}
}
