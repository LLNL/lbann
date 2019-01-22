# lbann-onnx
This tool provides a way to convert [LBANN](https://github.com/LLNL/lbann) models from/to [ONNX](https://github.com/onnx/onnx) models.
* `LBANN_ROOT` environment variable may be used to explicitly specify a LBANN root directory (otherwise, `git rev-parse --show-toplevel` is used).

## Requirements
* [LBANN](https://github.com/LLNL/lbann)
* Python >= 3.7.2
* [ONNX](https://github.com/onnx/onnx) >= 1.3.0
* [NumPy](http://www.numpy.org/) >= 1.16.0
* [Protobuf](https://github.com/protocolbuffers/protobuf) >= 3.6.1

The Python packages of ONNX, NumPy and Prootbuf can be install via `pip3 install onnx numpy protobuf`.

## How to Setup
1. Run `pip3 install -e .`
2. Run `python3 -c "import lbann_onnx"` to verify that the package has been installed.
3. Run `python3 setup.py test` to verify the generated Protobuf definition.
   * This will generate converted ONNX models if you set `SAVE_ONNX=True`.

## How to Use
See [`example/lbann2onnx.py`](example/lbann2onnx.py) and [`example/onnx2lbann.py`](example/onnx2lbann.py) for details.

## Support Status
See [`docs/l2o.md`](docs/l2o.md) and [`docs/o2l.md`](docs/o2l.md) for details.

## Example: Converting [the MNIST model](/model_zoo/models/simple_mnist/model_mnist_simple_1.prototext) from LBANN to ONNX
See [`viz/l2o/`](viz/l2o/) for more details.

### LBANN (vizualized with [viz.py](/viz/viz.py))
<img src="viz/l2o/mnist/mnist_lbann.png" width="200" />

### ONNX (vizualized with [Netron](https://github.com/lutzroeder/netron))
<img src="viz/l2o/mnist/mnist_onnx_netron.png" width="200" />

### ONNX with the original node names
<img src="viz/l2o/mnist/mnist_onnx_netron_name.png" width="200" />
