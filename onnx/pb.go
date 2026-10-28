package onnx

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// A minimal protocol-buffers reader for the ONNX messages (wire format:
// https://protobuf.dev/programming-guides/encoding/). It reads from a byte
// slice with every length checked against what remains, so a hostile model
// can make it fail but never read out of bounds or allocate more than the
// input's size.

const (
	wireVarint = 0
	wireI64    = 1
	wireLen    = 2
	wireI32    = 5
)

var errTruncated = errors.New("truncated")

type pbReader struct {
	b   []byte
	off int
}

func (r *pbReader) done() bool { return r.off >= len(r.b) }

func (r *pbReader) varint() (uint64, error) {
	var v uint64
	for i := 0; i < 10; i++ {
		if r.off >= len(r.b) {
			return 0, errTruncated
		}
		c := r.b[r.off]
		r.off++
		if i == 9 && c > 1 {
			return 0, errors.New("varint overflows 64 bits")
		}
		v |= uint64(c&0x7f) << (7 * i)
		if c < 0x80 {
			return v, nil
		}
	}
	return 0, errors.New("varint longer than 10 bytes")
}

// key reads a field key: its number and wire type.
func (r *pbReader) key() (int, int, error) {
	k, err := r.varint()
	if err != nil {
		return 0, 0, err
	}
	field := k >> 3
	if field == 0 || field > math.MaxInt32 {
		return 0, 0, fmt.Errorf("bad field number %d", field)
	}
	return int(field), int(k & 7), nil
}

func (r *pbReader) bytes() ([]byte, error) {
	n, err := r.varint()
	if err != nil {
		return nil, err
	}
	if n > uint64(len(r.b)-r.off) {
		return nil, errTruncated
	}
	b := r.b[r.off : r.off+int(n)]
	r.off += int(n)
	return b, nil
}

func (r *pbReader) fixed32() (uint32, error) {
	if len(r.b)-r.off < 4 {
		return 0, errTruncated
	}
	v := binary.LittleEndian.Uint32(r.b[r.off:])
	r.off += 4
	return v, nil
}

func (r *pbReader) fixed64() (uint64, error) {
	if len(r.b)-r.off < 8 {
		return 0, errTruncated
	}
	v := binary.LittleEndian.Uint64(r.b[r.off:])
	r.off += 8
	return v, nil
}

// skip passes over a field of wire type w. Groups (deprecated, unused by
// ONNX) are refused.
func (r *pbReader) skip(w int) error {
	var err error
	switch w {
	case wireVarint:
		_, err = r.varint()
	case wireI64:
		_, err = r.fixed64()
	case wireLen:
		_, err = r.bytes()
	case wireI32:
		_, err = r.fixed32()
	default:
		err = fmt.Errorf("unsupported wire type %d", w)
	}
	return err
}

// str reads a LEN field as a string, bounded by max bytes.
func (r *pbReader) str(w, max int) (string, error) {
	if w != wireLen {
		return "", fmt.Errorf("wire type %d for a string", w)
	}
	b, err := r.bytes()
	if err != nil {
		return "", err
	}
	if len(b) > max {
		return "", fmt.Errorf("string of %d bytes (limit %d)", len(b), max)
	}
	return string(b), nil
}

// int64s appends a repeated int64 field, packed or not.
func (r *pbReader) int64s(w int, dst []int64, max int) ([]int64, error) {
	switch w {
	case wireVarint:
		v, err := r.varint()
		if err != nil {
			return dst, err
		}
		if len(dst) >= max {
			return dst, fmt.Errorf("more than %d values", max)
		}
		return append(dst, int64(v)), nil
	case wireLen:
		b, err := r.bytes()
		if err != nil {
			return dst, err
		}
		sub := pbReader{b: b}
		for !sub.done() {
			v, err := sub.varint()
			if err != nil {
				return dst, err
			}
			if len(dst) >= max {
				return dst, fmt.Errorf("more than %d values", max)
			}
			dst = append(dst, int64(v))
		}
		return dst, nil
	}
	return dst, fmt.Errorf("wire type %d for int64 values", w)
}

// float32s appends a repeated float field, packed or not.
func (r *pbReader) float32s(w int, dst []float32, max int) ([]float32, error) {
	switch w {
	case wireI32:
		v, err := r.fixed32()
		if err != nil {
			return dst, err
		}
		if len(dst) >= max {
			return dst, fmt.Errorf("more than %d values", max)
		}
		return append(dst, math.Float32frombits(v)), nil
	case wireLen:
		b, err := r.bytes()
		if err != nil {
			return dst, err
		}
		if len(b)%4 != 0 {
			return dst, errors.New("packed floats not a multiple of 4 bytes")
		}
		if len(dst)+len(b)/4 > max {
			return dst, fmt.Errorf("more than %d values", max)
		}
		for i := 0; i < len(b); i += 4 {
			dst = append(dst, math.Float32frombits(binary.LittleEndian.Uint32(b[i:])))
		}
		return dst, nil
	}
	return dst, fmt.Errorf("wire type %d for float values", w)
}

// float64s appends a repeated double field, packed or not.
func (r *pbReader) float64s(w int, dst []float64, max int) ([]float64, error) {
	switch w {
	case wireI64:
		v, err := r.fixed64()
		if err != nil {
			return dst, err
		}
		if len(dst) >= max {
			return dst, fmt.Errorf("more than %d values", max)
		}
		return append(dst, math.Float64frombits(v)), nil
	case wireLen:
		b, err := r.bytes()
		if err != nil {
			return dst, err
		}
		if len(b)%8 != 0 {
			return dst, errors.New("packed doubles not a multiple of 8 bytes")
		}
		if len(dst)+len(b)/8 > max {
			return dst, fmt.Errorf("more than %d values", max)
		}
		for i := 0; i < len(b); i += 8 {
			dst = append(dst, math.Float64frombits(binary.LittleEndian.Uint64(b[i:])))
		}
		return dst, nil
	}
	return dst, fmt.Errorf("wire type %d for double values", w)
}
