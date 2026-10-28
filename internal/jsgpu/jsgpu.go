//go:build js

// Package jsgpu is a thin Go layer over the browser's WebGPU API
// (https://www.w3.org/TR/webgpu/) through syscall/js, for code running in
// a Web Worker (the playground's engine worker): a device, storage
// buffers, compute pipelines from WGSL, dispatches and readback. Every
// promise is awaited from a goroutine, never from a JS callback.
package jsgpu

import (
	"errors"
	"fmt"
	"sync"
	"syscall/js"
)

// Await waits for a promise. It must not run inside a js.FuncOf callback.
func Await(p js.Value) (js.Value, error) {
	type res struct {
		v   js.Value
		err error
	}
	ch := make(chan res, 1)
	var ok, fail js.Func
	ok = js.FuncOf(func(_ js.Value, a []js.Value) any {
		v := js.Undefined()
		if len(a) > 0 {
			v = a[0]
		}
		ch <- res{v: v}
		return nil
	})
	fail = js.FuncOf(func(_ js.Value, a []js.Value) any {
		msg := "rejected"
		if len(a) > 0 {
			msg = a[0].Call("toString").String()
		}
		ch <- res{err: errors.New(msg)}
		return nil
	})
	p.Call("then", ok, fail)
	r := <-ch
	ok.Release()
	fail.Release()
	return r.v, r.err
}

// AdapterInfo describes the adapter a device came from.
type AdapterInfo struct {
	Vendor, Architecture string
	// Fallback is a software adapter (SwiftShader and the like): work done
	// on it is not GPU work and must not be reported as such.
	Fallback bool
}

// Device is a WebGPU device.
type Device struct {
	JS    js.Value
	Info  AdapterInfo
	Queue js.Value

	mu        sync.Mutex
	pipelines map[string]js.Value
	lost      bool
}

// ErrNoAdapter is returned when the browser offers no WebGPU adapter.
var ErrNoAdapter = errors.New("webgpu: no adapter")

// NewDevice requests an adapter and a device. A fallback adapter is
// refused unless allowFallback.
func NewDevice(allowFallback bool) (*Device, error) {
	gpu := js.Global().Get("navigator").Get("gpu")
	if gpu.IsUndefined() || gpu.IsNull() {
		return nil, ErrNoAdapter
	}
	a, err := Await(gpu.Call("requestAdapter"))
	if err != nil {
		return nil, err
	}
	if a.IsNull() || a.IsUndefined() {
		return nil, ErrNoAdapter
	}
	info := a.Get("info")
	ai := AdapterInfo{}
	if !info.IsUndefined() {
		ai.Vendor = info.Get("vendor").String()
		ai.Architecture = info.Get("architecture").String()
		if f := info.Get("isFallbackAdapter"); f.Type() == js.TypeBoolean {
			ai.Fallback = f.Bool()
		}
	}
	if f := a.Get("isFallbackAdapter"); f.Type() == js.TypeBoolean && f.Bool() {
		ai.Fallback = true
	}
	if ai.Fallback && !allowFallback {
		return nil, fmt.Errorf("webgpu: only a fallback adapter (%s %s)", ai.Vendor, ai.Architecture)
	}
	d, err := Await(a.Call("requestDevice"))
	if err != nil {
		return nil, err
	}
	dev := &Device{JS: d, Info: ai, Queue: d.Get("queue"), pipelines: map[string]js.Value{}}
	go func() {
		Await(d.Get("lost"))
		dev.mu.Lock()
		dev.lost = true
		dev.mu.Unlock()
	}()
	return dev, nil
}

// Lost reports whether the device was lost.
func (d *Device) Lost() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.lost
}

// Destroy releases the device and everything it holds.
func (d *Device) Destroy() { d.JS.Call("destroy") }

// Buffer usage flags (GPUBufferUsage).
const (
	UsageMapRead = 0x0001
	UsageCopySrc = 0x0004
	UsageCopyDst = 0x0008
	UsageUniform = 0x0040
	UsageStorage = 0x0080
)

// Buffer creates a buffer of size bytes (rounded up to 4).
func (d *Device) Buffer(size int, usage int) js.Value {
	size = max(4, (size+3)&^3)
	desc := js.Global().Get("Object").New()
	desc.Set("size", size)
	desc.Set("usage", usage)
	return d.JS.Call("createBuffer", desc)
}

// Upload creates a storage buffer holding b.
func (d *Device) Upload(b []byte) js.Value {
	buf := d.Buffer(len(b), UsageStorage|UsageCopyDst|UsageCopySrc)
	if len(b) > 0 {
		padded := b
		if len(b)%4 != 0 {
			padded = append(append([]byte(nil), b...), make([]byte, 4-len(b)%4)...)
		}
		u := js.Global().Get("Uint8Array").New(len(padded))
		js.CopyBytesToJS(u, padded)
		d.Queue.Call("writeBuffer", buf, 0, u)
	}
	return buf
}

// Read copies size bytes out of a storage buffer (copy to a mappable
// buffer, mapAsync, getMappedRange).
func (d *Device) Read(src js.Value, size int) ([]byte, error) {
	n := max(4, (size+3)&^3)
	staging := d.Buffer(n, UsageMapRead|UsageCopyDst)
	defer staging.Call("destroy")
	enc := d.JS.Call("createCommandEncoder")
	enc.Call("copyBufferToBuffer", src, 0, staging, 0, n)
	d.Queue.Call("submit", []any{enc.Call("finish")})
	if _, err := Await(staging.Call("mapAsync", 1)); err != nil { // GPUMapMode.READ
		return nil, err
	}
	u := js.Global().Get("Uint8Array").New(staging.Call("getMappedRange", 0, n))
	b := make([]byte, n)
	js.CopyBytesToGo(b, u)
	staging.Call("unmap")
	return b[:size], nil
}

// Pipeline returns the compute pipeline of a WGSL module (entry point
// "main", layout "auto"), compiled once per source.
func (d *Device) Pipeline(wgsl string) (js.Value, error) {
	d.mu.Lock()
	p, ok := d.pipelines[wgsl]
	d.mu.Unlock()
	if ok {
		return p, nil
	}
	md := js.Global().Get("Object").New()
	md.Set("code", wgsl)
	mod := d.JS.Call("createShaderModule", md)
	desc := js.Global().Get("Object").New()
	desc.Set("layout", "auto")
	comp := js.Global().Get("Object").New()
	comp.Set("module", mod)
	comp.Set("entryPoint", "main")
	desc.Set("compute", comp)
	pv, err := Await(d.JS.Call("createComputePipelineAsync", desc))
	if err != nil {
		return js.Value{}, fmt.Errorf("webgpu: pipeline: %w", err)
	}
	d.mu.Lock()
	d.pipelines[wgsl] = pv
	d.mu.Unlock()
	return pv, nil
}

// Dispatch runs pipeline over n invocations (64 per workgroup, laid over
// x and y when n exceeds one dimension's limit) with buffers bound at
// group 0, bindings 0.. in order.
func (d *Device) Dispatch(pipeline js.Value, n int, buffers ...js.Value) {
	entries := make([]any, len(buffers))
	for i, b := range buffers {
		e := js.Global().Get("Object").New()
		e.Set("binding", i)
		r := js.Global().Get("Object").New()
		r.Set("buffer", b)
		e.Set("resource", r)
		entries[i] = e
	}
	bg := js.Global().Get("Object").New()
	bg.Set("layout", pipeline.Call("getBindGroupLayout", 0))
	bg.Set("entries", entries)
	group := d.JS.Call("createBindGroup", bg)
	enc := d.JS.Call("createCommandEncoder")
	pass := enc.Call("beginComputePass")
	pass.Call("setPipeline", pipeline)
	pass.Call("setBindGroup", 0, group)
	groups := (n + 63) / 64
	x, y := groups, 1
	if x > 65535 {
		y = (x + 65534) / 65535
		x = 65535
	}
	pass.Call("dispatchWorkgroups", x, y)
	pass.Call("end")
	d.Queue.Call("submit", []any{enc.Call("finish")})
}

// Done waits until the queue's submitted work has completed.
func (d *Device) Done() error {
	_, err := Await(d.Queue.Call("onSubmittedWorkDone"))
	return err
}
