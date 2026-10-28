//go:build wasip1

package wasinn

import "unsafe"

//go:wasmimport wasi_ephemeral_nn load
func nnLoad(builders unsafe.Pointer, n, enc, target uint32, graph unsafe.Pointer) uint32

//go:wasmimport wasi_ephemeral_nn load_by_name
func nnLoadByName(name unsafe.Pointer, n uint32, graph unsafe.Pointer) uint32

//go:wasmimport wasi_ephemeral_nn init_execution_context
func nnInit(graph uint32, ctx unsafe.Pointer) uint32

//go:wasmimport wasi_ephemeral_nn set_input
func nnSetInput(ctx, index uint32, tensor unsafe.Pointer) uint32

//go:wasmimport wasi_ephemeral_nn get_output
func nnGetOutput(ctx, index uint32, buf unsafe.Pointer, max uint32, written unsafe.Pointer) uint32

//go:wasmimport wasi_ephemeral_nn compute
func nnCompute(ctx uint32) uint32

func callLoad(b unsafe.Pointer, n, enc, target uint32, g unsafe.Pointer) error {
	return check(nnLoad(b, n, enc, target, g))
}
func callLoadByName(p unsafe.Pointer, n uint32, g unsafe.Pointer) error {
	return check(nnLoadByName(p, n, g))
}
func callInit(g uint32, c unsafe.Pointer) error { return check(nnInit(g, c)) }
func callSetInput(c, i uint32, t unsafe.Pointer) error {
	return check(nnSetInput(c, i, t))
}
func callGetOutput(c, i uint32, b unsafe.Pointer, max uint32, n unsafe.Pointer) error {
	return check(nnGetOutput(c, i, b, max, n))
}
func callCompute(c uint32) error { return check(nnCompute(c)) }
