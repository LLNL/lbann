////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYER_MATH_MATMUL_HPP_INCLUDED
#define LBANN_LAYER_MATH_MATMUL_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Matrix multiplication.
 *
 *  Takes two 2D input tensors and outputs their matrix product.
 *  Matrix products are computed independently for each mini-batch
 *  sample, in a similar manner as NumPy's matmul function.
 *
 *  @todo Support >2 dimensions, transposes, matvecs, and dot products
 *
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class matmul_layer : public data_type_layer<TensorDataType> {
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "matmul_layer only supports "
                "data-parallel data layout");

public:

  matmul_layer(lbann_comm *comm);
  matmul_layer(const matmul_layer& other) = default;
  matmul_layer& operator=(const matmul_layer& other) = default;
  matmul_layer* copy() const override;
  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

protected:

  void setup_dims() override;
  void fp_compute() override;
  void bp_compute() override;

};

// =========================================================
// Implementation
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
matmul_layer<TensorDataType, Layout,Device>::matmul_layer(lbann_comm *comm)
  : data_type_layer<TensorDataType>(comm) {
  this->m_expected_num_parent_layers = 2;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
matmul_layer<TensorDataType, Layout,Device>* matmul_layer<TensorDataType,Layout,Device>::copy() const {
  return new matmul_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string matmul_layer<TensorDataType,Layout,Device>::get_type() const {
  return "matrix multiply";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout matmul_layer<TensorDataType,Layout,Device>::get_data_layout() const {
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device matmul_layer<TensorDataType,Layout,Device>::get_device_allocation() const {
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void matmul_layer<TensorDataType,Layout,Device>::setup_dims() {
  data_type_layer<TensorDataType>::setup_dims();

  // Input dimensions
  const auto& input0_dims = this->get_input_dims(0);
  const auto& input1_dims = this->get_input_dims(1);

  // Lambdas to help print error messages
  auto print_name = [this] () -> std::string {
    return this->get_type() + " layer \"" + this->get_name() + "\"";
  };
  auto print_inputs = [this, &input0_dims, &input1_dims] () -> std::string {
    auto print_dims = [] (const decltype(input0_dims)& dims) -> std::string {
      std::ostringstream ss;
      for (size_t i = 0; i < dims.size(); ++i) {
        ss << (i > 0 ? "x" : "") << dims[i];
      }
      return ss.str();
    };
    const auto& parents = this->get_parent_layers();
    return lbann::build_string(
      parents[0]->get_type()," layer \"",parents[0]->get_name(),"\" ",
      "outputs ",print_dims(input0_dims),", ",
      parents[1]->get_type()," layer \"",parents[1]->get_name(),"\" ",
      "outputs ",print_dims(input1_dims));
  };

  // Check input dimensions
  if (input0_dims.size() != input1_dims.size()) {
    LBANN_ERROR("input tensors in ",print_name()," "
                "have different numbers of dimensions ",
                "(",print_inputs(),")");
  }
  if (input0_dims.size() != 2) {
    LBANN_ERROR("input tensors in ",print_name()," are not 2D ",
                "(",print_inputs(),")");
  }

  // Get dimensions for matrix multiply
  const auto m = *(input0_dims.rbegin()+1);
  const auto n = *(input1_dims.rbegin());
  const auto k = *(input0_dims.rbegin());
  if (*(input1_dims.rbegin()+1) != k || m < 1 || n < 1 || k < 1) {
    LBANN_ERROR("input tensors in ",print_name()," ",
                "are not compatible with matrix multiplication ",
                "(",print_inputs(),")");
  }

  // Set output dimensions
  std::vector<int> output_dims(input0_dims);
  *(output_dims.rbegin()+1) = m;
  *(output_dims.rbegin()) = n;
  this->set_output_dims(output_dims);

}

// =========================================================
// Explicit template instantiation
// =========================================================

#ifndef LBANN_MATMUL_LAYER_INSTANTIATE
extern template class matmul_layer<
  float, data_layout::DATA_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class matmul_layer<
  float, data_layout::DATA_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_MATMUL_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_MATH_MATMUL_HPP_INCLUDED
