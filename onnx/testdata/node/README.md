Node tests from github.com/onnx/onnx at tag v1.22.0 (commit
2bb50465112feca9003e1ed654d77f01ff1415ca), `onnx/backend/test/data/node`,
under the Apache License 2.0 (LICENSE here). v1.22.0 is the last tag that
ships this directory.

The selection is derived, not written: `go run ./internal/cmd/onnxvendor
--from <checkout>/onnx/backend/test/data/node` keeps every test whose model
package onnx loads (supported operators, float/int64 graph values). Files
are unmodified.
