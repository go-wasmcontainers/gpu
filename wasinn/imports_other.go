//go:build !wasip1

package wasinn

import "unsafe"

func callLoad(unsafe.Pointer, uint32, uint32, uint32, unsafe.Pointer) error { return ErrUnavailable }
func callLoadByName(unsafe.Pointer, uint32, unsafe.Pointer) error           { return ErrUnavailable }
func callInit(uint32, unsafe.Pointer) error                                 { return ErrUnavailable }
func callSetInput(uint32, uint32, unsafe.Pointer) error                     { return ErrUnavailable }
func callGetOutput(uint32, uint32, unsafe.Pointer, uint32, unsafe.Pointer) error {
	return ErrUnavailable
}
func callCompute(uint32) error { return ErrUnavailable }
