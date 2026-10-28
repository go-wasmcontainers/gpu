// Command mnist is a WASI container that reads a handwritten digit: it
// embeds MNIST-12 (ONNX model zoo, MIT) and a 28x28 picture, runs the
// model through wasi-nn, and prints the digit with its probability.
//
//	GOOS=wasip1 GOARCH=wasm go build -o mnist.wasm ./examples/mnist
//	wctr run [--gpu] IMAGE [cpu|gpu] [picture.png]
//
// The target is "cpu" unless the first argument says "gpu"; a GPU target
// on an engine without one fails with "unsupported operation", it is not
// run on the CPU instead.
package main

import (
	"bytes"
	_ "embed"
	"fmt"
	"image/color"
	"image/png"
	"math"
	"os"

	"github.com/go-wasmcontainers/gpu/wasinn"
)

//go:embed mnist-12.onnx
var model []byte

//go:embed digit.png
var digit []byte

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "mnist:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	target := wasinn.CPU
	if len(args) > 0 && (args[0] == "gpu" || args[0] == "cpu") {
		if args[0] == "gpu" {
			target = wasinn.GPU
		}
		args = args[1:]
	}
	pic := digit
	if len(args) > 0 {
		b, err := os.ReadFile(args[0])
		if err != nil {
			return err
		}
		pic = b
	}
	x, err := pixels(pic)
	if err != nil {
		return err
	}
	g, err := wasinn.Load([][]byte{model}, wasinn.ONNX, target)
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}
	c, err := g.NewContext()
	if err != nil {
		return err
	}
	if err := c.SetInputF32(0, []uint32{1, 1, 28, 28}, x); err != nil {
		return fmt.Errorf("input: %w", err)
	}
	if err := c.Compute(); err != nil {
		return fmt.Errorf("compute: %w", err)
	}
	scores := make([]float32, 10)
	n, err := c.OutputF32(0, scores)
	if err != nil {
		return fmt.Errorf("output: %w", err)
	}
	if n != 10 {
		return fmt.Errorf("%d scores, want 10", n)
	}
	best, p := classify(scores)
	fmt.Printf("digit %d (p=%.3f)\n", best, p)
	return nil
}

// pixels decodes a picture to 28x28 gray values in [0, 1], white digit on
// black (the model's input).
func pixels(b []byte) ([]float32, error) {
	img, err := png.Decode(bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	r := img.Bounds()
	if r.Dx() != 28 || r.Dy() != 28 {
		return nil, fmt.Errorf("picture is %dx%d, want 28x28", r.Dx(), r.Dy())
	}
	x := make([]float32, 0, 28*28)
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for xx := r.Min.X; xx < r.Max.X; xx++ {
			g := color.GrayModel.Convert(img.At(xx, y)).(color.Gray)
			x = append(x, float32(g.Y)/255)
		}
	}
	return x, nil
}

// classify returns the best class and its softmax probability.
func classify(scores []float32) (int, float64) {
	best := 0
	for i, s := range scores {
		if s > scores[best] {
			best = i
		}
	}
	var sum float64
	for _, s := range scores {
		sum += math.Exp(float64(s - scores[best]))
	}
	return best, 1 / sum
}
