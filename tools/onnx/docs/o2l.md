# Support Status of ONNX->LBANN Conversion
## Example: `fc6` of [AlexNet](https://github.com/onnx/models/tree/master/bvlc_alexnet)
### `fc6` in ONNX:
```
node {
  input: "OC2_DUMMY_0"
  input: "fc6_w_0"
  input: "fc6_b_0"
  output: "fc6_1"
  op_type: "Gemm"
  attribute {
    name: "transB"
    i: 1
    type: INT
  }
}
```
```
initializer {
  dims: 4096
  data_type: FLOAT
  name: "fc6_b_0"
  raw_data: "..."
}
initializer {
  dims: 4096
  dims: 9216
  data_type: FLOAT
  name: "fc6_w_0"
  raw_data: "..."
}
```
```
input {
  name: "fc6_w_0"
  type {
    tensor_type {
      elem_type: FLOAT
      shape {
        dim {
          dim_value: 4096
        }
        dim {
          dim_value: 9216
        }
      }
    }
  }
}
input {
  name: "fc6_b_0"
  type {
    tensor_type {
      elem_type: FLOAT
      shape {
        dim {
          dim_value: 4096
        }
      }
    }
  }
}
```

### `fc6` in LBANN:
```
layer {
  fully_connected {
    num_neurons: 4096
    has_bias: true
  }
  name: "Gemm_16"
  data_layout: "model_parallel"
  parents: "Reshape_15"
}
```
