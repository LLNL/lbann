# lbann-onnx
This tool provides a way to convert [LBANN](https://github.com/LLNL/lbann) models from/to [ONNX](https://github.com/onnx/onnx) models.
* `LBANN_ROOT` environment variable may be used to explicitly specify a LBANN root directory.
   * Otherwise, `git rev-parse --show-toplevel` is used.

## Requirements
* [LBANN](https://github.com/LLNL/lbann)
* Python >= 3.7.2
* [ONNX](https://github.com/onnx/onnx) >= 1.3.0
* [NumPy](http://www.numpy.org/) >= 1.16.0
* [Protobuf](https://github.com/protocolbuffers/protobuf) >= 3.6.1

## How to Setup
1. Run `pip3 install -e .`
   * `pip3` tries to install the dependent libralies if you don't have.
2. Run `python3 -c "import lbann_onnx"` to verify that the package has been installed.
3. Run `python3 setup.py test` to verify generated Protobuf/ONNX files.
   * You may need to run [`scripts/download_onnx_model_zoo.sh`](scripts/download_onnx_model_zoo.sh) to get pre-trained ONNX models.
   * Converted Protobuf/ONNX models will be generated if `LBANN_ONNX_DUMP_MODELS=1` is set.

## How to Use
See [`examples/lbann2onnx.py`](examples/lbann2onnx.py) and [`examples/onnx2lbann.py`](examples/onnx2lbann.py) for details.
* Set `LBANN_ONNX_VERBOSE=1` to show detailed conversion warnings.

## Support Status
See the following documentation for details.
* [Operators/Layers Support Status](docs/support_status.md)
* [Details of the LBANN -> ONNX Conversion](docs/l2o.md)
* [Details of the ONNX -> LBANN Conversion](docs/o2l.md)

## Example: Converting [the MNIST model](/model_zoo/models/simple_mnist/model_mnist_simple_1.prototext) from LBANN to ONNX
See [`viz/l2o/`](viz/l2o/) for more details.

### LBANN (vizualized with [viz.py](/viz/viz.py))
<img src="viz/l2o/mnist/mnist_lbann.png" width="200" />

### ONNX (vizualized with [Netron](https://github.com/lutzroeder/netron))
<img src="viz/l2o/mnist/mnist_onnx_netron.png" width="200" />

### ONNX with the original node names
<img src="viz/l2o/mnist/mnist_onnx_netron_name.png" width="200" />
