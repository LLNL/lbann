////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include "lbann/layers/activations/relu.hpp"
#include "lbann/utils/cuda.hpp"

namespace lbann {

namespace {

/** Entry-wise operator. */
struct op {
  __device__ DataType operator()(DataType x) const {
    return x > DataType(0) ? x : DataType(0);
  }
};
  
/** Entry-wise operator for backprop.
 *  If the forward propagation step computes \f$ y = f(x) \f$, the
 *  backward propagation step computes
 *  \f$ \frac{dL}{dx} = \frac{dL}{dy} f'(x) \f$.
 */
struct op_backprop {
  __device__ DataType operator()(DataType x, DataType dy) const {
    return x > DataType(0) ? dy : DataType(0);
  }
};
  
} // namespace

// Template instantiation
template <>
void relu_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {
  cuda::apply_entrywise_unary_operator<op>(get_prev_activations(),
                                           get_activations());
}
template <>
void relu_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {
  cuda::apply_entrywise_binary_operator<op_backprop>(get_prev_activations(),
                                                     get_prev_error_signals(),
                                                     get_error_signals());
}
template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
  cuda::apply_entrywise_unary_operator<op>(get_prev_activations(),
                                           get_activations());
}
template <>
void relu_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
  cuda::apply_entrywise_binary_operator<op_backprop>(get_prev_activations(),
                                                     get_prev_error_signals(),
                                                     get_error_signals());
}
  
} // namespace lbann
