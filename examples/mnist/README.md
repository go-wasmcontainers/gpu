# mnist: wasi-nn in a container

Reads a handwritten digit with MNIST-12 through wasi-nn
(`wasi_ephemeral_nn`), on the engine's pure-Go CPU backend.

```
GOOS=wasip1 GOARCH=wasm go build -o mnist.wasm ./examples/mnist
wctr run IMAGE              # digit 7 (p=1.000)
wctr run IMAGE cpu /pic.png # a 28x28 PNG of your own, white on black
wctr run --gpu IMAGE gpu    # needs a GPU backend; fails plainly without one
```

- `mnist-12.onnx`: ONNX Model Zoo, `validated/vision/classification/mnist`,
  MIT (model card).
- `digit.png`: picture 0 of the MNIST test set (label 7), Y. LeCun,
  C. Cortes, C. J. C. Burges, CC BY-SA 3.0; extracted unmodified by
  `internal/cmd/mnistpng`.
