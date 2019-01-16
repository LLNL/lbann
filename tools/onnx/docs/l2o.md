# Support Status of LBANN->ONNX Conversion

## Supported Layers
* I/O Layers
  * An `io` layer is converted to ONNX's input tensor(s) since ONNX models cannot represent the abstraction of I/O.
* [Learning Layers](../lbann_onnx/l2o/layers/learnings.py)
  * `fully_connected`
  * `convolution`
* [Regularizer Layers](../lbann_onnx/l2o/layers/regularizers.py)
  * `batch_normalization`
  * `local_response_normalization`
  * `dropout`
* [Transform Layers](../lbann_onnx/l2o/layers/transforms.py)
  * `pooling`
  * `unpooling`
  * `slice`
  * `concatenation`
  * `gaussian`
  * `reshape`
  * `reduction`
* [Math Layers](../lbann_onnx/l2o/layers/math.py)
  * `relu`
  * `leaky_relu`
  * `sigmoid`
  * `tanh`
  * `softmax`
  * `exp`
  * `add`
  * `sum`
  * `hadamard`
  * `abs`
  * `weighted_sum` (partially supported)
  * `constant`

## Dummy/Non-supported Layers
Some LBANN layers are not supported since there is no equivalent operation in ONNX.

* [Transform Layers](../lbann_onnx/l2o/layers/transforms.py)
  * `evaluation`
  * `zero`
* [Math Layers](../lbann_onnx/l2o/layers/math.py)
  * `square`
  * `rsqrt`
* [Loss Functions](../lbann_onnx/l2o/layers/losses.py)
  * Since ONNX is intended for describing DNN models, loss functions cannot be represented directly. They still can be described by combining ONNX's arithemtic operations.

## Difference between LBANN/ONNX Models
* The following attributes are stored in ONNX nodes to keep the original information of LBANN models.
  * `lbannOp`: The original layer type as a string
  * `lbannDataLayout`: The `data_layout` attribute of the original layer as a string
* An ONNX's `Reshape` node is inserted before a `Gemm` node if the input dimension is not 2D.

## Example: `fc6` of [AlexNet](../../../model_zoo/models/alexnet/model_alexnet.prototext)
### `fc6` in LBANN:
```
layer {
  name: "fc6"
  parents: "pool5"
  data_layout: "model_parallel"
  fully_connected {
    num_neurons: 4096
    has_bias: true
  }
}
```

### `fc6` in ONNX:
```
node {
  input: "pool5_0_reshaped_15"
  input: "fc6_p0"
  input: "fc6_p1"
  output: "fc6_0"
  name: "fc6"
  op_type: "Gemm"
  attribute {
    name: "lbannDataLayout"
    s: "model_parallel"
    type: STRING
  }
  attribute {
    name: "lbannOp"
    s: "fully_connected"
    type: STRING
  }
  attribute {
    name: "transB"
    i: 1
    type: INT
  }
}
```
`fc6_p0` and `fc6_p1` represent its two parameter tensors (weights and biases).
