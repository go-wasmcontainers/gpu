MNIST-12 from the ONNX Model Zoo, github.com/onnx/models,
`validated/vision/classification/mnist/model/mnist-12.tar.gz` (model and
test_data_set_0, unmodified). The model card gives its license as MIT.

Input: float32 1x1x28x28, grayscale in [0, 1], white digit on black.
Output: 1x10 scores before softmax.
