//go:build js

package webgpu

import "syscall/js"

// presentSlowKeep bounds self.__presentSlow, the browser's list of slow presents.
const presentSlowKeep = 600

func init() {
	SlowPresent = func(i PresentInfo) {
		l := js.Global().Get("__presentSlow")
		if l.IsUndefined() {
			l = js.Global().Get("Array").New()
			js.Global().Set("__presentSlow", l)
		}
		if l.Length() >= presentSlowKeep {
			return
		}
		perf := js.Global().Get("performance")
		ms := func(d interface{ Microseconds() int64 }) float64 { return float64(d.Microseconds()) / 1000 }
		l.Call("push", map[string]any{
			"at":     perf.Call("now").Float() + perf.Get("timeOrigin").Float(),
			"paced":  ms(i.Paced),
			"acked":  ms(i.Acked),
			"handed": ms(i.Handed),
		})
	}
}
