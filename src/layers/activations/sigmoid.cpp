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

#include "lbann/layers/activations/sigmoid.hpp"
#include "lbann/utils/entrywise_operator.hpp"

namespace lbann {

namespace {

// Useful constants
constexpr DataType zero = 0;
constexpr DataType one = 1;
constexpr DataType eps = std::numeric_limits<DataType>::epsilon();

/** Entry-wise operator. */
struct op {
  inline DataType operator()(DataType x) const {
    const DataType y = 1 / (one + std::exp(-x));
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    if (y <= eps)            { return eps; }
    else if (y >= one - eps) { return one - eps; }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return y;
  }
};
  
/** Entry-wise operator for backprop.
 *  If the forward propagation step computes \f$ y = f(x) \f$, the
 *  backward propagation step computes
 *  \f$ \frac{dL}{dx} = \frac{dL}{dy} f'(x) \f$.
 */
struct op_backprop {
  inline DataType operator()(DataType x, DataType dy) const {
    const auto& y = op()(x);
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    if (y <= eps || y >= one - eps) { return zero; }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return dy * y * (one - y);
  }
};
  
} // namespace

// Template instantiation
template <>
void sigmoid_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::fp_compute() {
  apply_entrywise_unary_operator<op>(get_prev_activations(),
                                     get_activations());
}
template <>
void sigmoid_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>::bp_compute() {
  apply_entrywise_binary_operator<op_backprop>(get_prev_activations(),
                                               get_prev_error_signals(),
                                               get_error_signals());
}
template <>
void sigmoid_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {
  apply_entrywise_unary_operator<op>(get_prev_activations(),
                                     get_activations());
}
template <>
void sigmoid_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::bp_compute() {
  apply_entrywise_binary_operator<op_backprop>(get_prev_activations(),
                                               get_prev_error_signals(),
                                               get_error_signals());
}
  
} // namespace lbann
