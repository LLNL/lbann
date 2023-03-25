////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

std::string const model_prototext = R"ptext(
model {
  objective_function {
    layer_term {
      scale_factor: 1.0
      layer: "layer16"
    }
  }
  metric {
    layer_metric {
      layer: "layer17"
      name: "accuracy"
      unit: "%"
    }
  }
  num_epochs: 20
  layer {
    name: "image"
    data_layout: "data_parallel"
    input {
      data_field: "samples"
    }
  }

  layer {
    name: "label"
    data_layout: "data_parallel"
    input {
      data_field: "labels"
    }
  }
  layer {
    name: "layer4"
    parents: "image"
    children: "layer5"
    convolution {
      num_dims: 2
      out_channels: 6
      kernel_size: 5
      stride: 1
      dilation: 1
      groups {
        value: 1
      }
      has_bias {
        value: true
      }
    }
  }
  layer {
    name: "layer5"
    parents: "layer4"
    children: "layer6"
    relu {
    }
  }
  layer {
    name: "layer6"
    parents: "layer5"
    children: "layer7"
    pooling {
      num_dims: 2
      pool_dims_i: 2
      pool_strides_i: 2
      pool_mode: "max"
    }
  }
  layer {
    name: "layer7"
    parents: "layer6"
    children: "layer8"
    convolution {
      num_dims: 2
      out_channels: 16
      kernel_size: 5
      stride: 1
      dilation: 1
      groups {
        value: 1
      }
      has_bias {
        value: true
      }
    }
  }
  layer {
    name: "layer8"
    parents: "layer7"
    children: "layer9"
    relu {
    }
  }
  layer {
    name: "layer9"
    parents: "layer8"
    children: "layer10"
    pooling {
      num_dims: 2
      pool_dims_i: 2
      pool_strides_i: 2
      pool_mode: "max"
    }
  }
  layer {
    name: "layer10"
    parents: "layer9"
    children: "layer11"
    fully_connected {
      num_neurons: 120
      has_bias: true
    }
  }
  layer {
    name: "layer11"
    parents: "layer10"
    children: "layer12"
    relu {
    }
  }
  layer {
    name: "layer12"
    parents: "layer11"
    children: "layer13"
    fully_connected {
      num_neurons: 84
      has_bias: true
    }
  }
  layer {
    name: "layer13"
    parents: "layer12"
    children: "layer14"
    relu {
    }
  }
  layer {
    name: "layer14"
    parents: "layer13"
    children: "layer15"
    fully_connected {
      num_neurons: 10
      has_bias: true
    }
  }
  layer {
    name: "layer15"
    parents: "layer14"
    children: "layer16"
    children: "layer17"
    softmax {
    }
  }
  layer {
    name: "layer17"
    parents: "layer15"
    parents: "label"
    categorical_accuracy {
    }
  }
  layer {
    name: "layer16"
    parents: "layer15"
    parents: "label"
    cross_entropy {
    }
  }
}
optimizer {
  sgd {
    learn_rate: 0.01
    momentum: 0.9
  }
}
trainer {
  mini_batch_size: 64
}
)ptext";
