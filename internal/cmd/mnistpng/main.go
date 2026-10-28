// Command mnistpng extracts one picture of the MNIST test set (IDX files,
// gzipped, as published by Y. LeCun, C. Cortes and C. Burges; CC BY-SA
// 3.0) to a 28x28 grayscale PNG and prints its label, so the mnist example
// classifies a real handwritten digit whose answer is known.
//
// The IDX format: big-endian int32 magic (2051 images, 2049 labels), the
// count, then for images the rows and columns, then one unsigned byte per
// pixel (0 background, 255 ink) or per label.
//
//	go run ./internal/cmd/mnistpng --images t10k-images-idx3-ubyte.gz --labels t10k-labels-idx1-ubyte.gz --index 0 --out examples/mnist/digit.png
package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"image"
	"image/png"
	"io"
	"os"

	"github.com/spf13/cobra"
)

var osExit = os.Exit

func main() {
	if err := RootCmd().Execute(); err != nil {
		osExit(1)
	}
}

// RootCmd is the command, for tests.
func RootCmd() *cobra.Command {
	var images, labels, out string
	var index int
	cmd := &cobra.Command{
		Use:          "mnistpng --images FILE --labels FILE --index N --out digit.png",
		Short:        "Write one MNIST test picture as a PNG and print its label",
		SilenceUsage: true,
		Args:         cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			img, err := readImage(images, index)
			if err != nil {
				return err
			}
			label, err := readLabel(labels, index)
			if err != nil {
				return err
			}
			f, err := os.Create(out)
			if err != nil {
				return err
			}
			if err := png.Encode(f, img); err != nil {
				f.Close()
				return err
			}
			if err := f.Close(); err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "wrote %s: test picture %d, label %d\n", out, index, label)
			return nil
		},
	}
	f := cmd.Flags()
	f.StringVar(&images, "images", "", "t10k-images-idx3-ubyte.gz")
	f.StringVar(&labels, "labels", "", "t10k-labels-idx1-ubyte.gz")
	f.IntVar(&index, "index", 0, "picture number")
	f.StringVar(&out, "out", "digit.png", "PNG to write")
	_ = cmd.MarkFlagRequired("images")
	_ = cmd.MarkFlagRequired("labels")
	return cmd
}

// open returns the decompressed IDX stream after checking its magic, and
// the header's further int32s.
func open(path string, magic uint32, more int) (io.Reader, []uint32, func() error, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, err
	}
	z, err := gzip.NewReader(f)
	if err != nil {
		f.Close()
		return nil, nil, nil, err
	}
	hdr := make([]uint32, 1+more)
	if err := binary.Read(z, binary.BigEndian, hdr); err != nil {
		f.Close()
		return nil, nil, nil, err
	}
	if hdr[0] != magic {
		f.Close()
		return nil, nil, nil, fmt.Errorf("%s: magic %d, want %d", path, hdr[0], magic)
	}
	return z, hdr[1:], f.Close, nil
}

func readImage(path string, index int) (*image.Gray, error) {
	z, hdr, closeFn, err := open(path, 2051, 3)
	if err != nil {
		return nil, err
	}
	defer closeFn()
	n, rows, cols := hdr[0], hdr[1], hdr[2]
	if rows != 28 || cols != 28 || index < 0 || uint32(index) >= n {
		return nil, fmt.Errorf("%s: %d pictures of %dx%d, picture %d asked", path, n, rows, cols, index)
	}
	if _, err := io.CopyN(io.Discard, z, int64(index)*28*28); err != nil {
		return nil, err
	}
	img := image.NewGray(image.Rect(0, 0, 28, 28))
	if _, err := io.ReadFull(z, img.Pix); err != nil {
		return nil, err
	}
	return img, nil
}

func readLabel(path string, index int) (int, error) {
	z, hdr, closeFn, err := open(path, 2049, 1)
	if err != nil {
		return 0, err
	}
	defer closeFn()
	if index < 0 || uint32(index) >= hdr[0] {
		return 0, fmt.Errorf("%s: %d labels, label %d asked", path, hdr[0], index)
	}
	if _, err := io.CopyN(io.Discard, z, int64(index)); err != nil {
		return 0, err
	}
	var b [1]byte
	if _, err := io.ReadFull(z, b[:]); err != nil {
		return 0, err
	}
	return int(b[0]), nil
}
