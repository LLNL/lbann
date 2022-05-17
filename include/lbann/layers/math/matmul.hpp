////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
#include "lbann/proto/datatype_helpers.hpp"

namespace lbann {

/** @brief Matrix multiplication.
 *
 *  Performs matrix product of two 2D input tensors. If the input
 *  tensors are 3D, then matrix products are computed independently
 *  over the first dimension, in a similar manner as NumPy's matmul
 *  function.
 *
 *  @todo Support >3 dimensions, matvecs, and dot products
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

  matmul_layer(lbann_comm *comm,
               bool transpose_a = false,
               bool transpose_b = false);
  matmul_layer(const matmul_layer& other) = default;
  matmul_layer& operator=(const matmul_layer& other) = default;
  matmul_layer* copy() const override;

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

  description get_description() const override;

  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

protected:
  friend class cereal::access;
  matmul_layer()
    : matmul_layer(nullptr, false, false)
  {}


  void setup_dims(DataReaderMetaData& dr_metadata) override;
  void fp_compute() override;
  void bp_compute() override;

private:

  /** If true, matrices from the first input tensor are transposed
   *  before multiplication. */
  bool m_transpose_a;
  /** If true, matrices from the second input tensor are transposed
   *  before multiplication. */
  bool m_transpose_b;

  template <typename U>
  friend void fp_compute_impl(matmul_layer<U, Layout, Device>&, bool, bool);
  template <typename U>
  friend void bp_compute_impl(matmul_layer<U, Layout, Device>&, bool, bool);
};

// =========================================================
// Implementation
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
matmul_layer<TensorDataType, Layout,Device>::matmul_layer(lbann_comm *comm, bool transpose_a, bool transpose_b)
  : data_type_layer<TensorDataType>(comm),
    m_transpose_a{transpose_a},
    m_transpose_b{transpose_b} {
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
description matmul_layer<TensorDataType,Layout,Device>::get_description() const {
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Transpose A", m_transpose_a);
  desc.add("Transpose B", m_transpose_b);
  return desc;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void matmul_layer<TensorDataType,Layout,Device>::setup_dims(DataReaderMetaData& dr_metadata) {
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);

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
  if (input0_dims.size() != 2 && input0_dims.size() != 3) {
    LBANN_ERROR("input tensors in ",print_name()," are not 2D or 3D",
                "(",print_inputs(),")");
  }

  // Get matrix dimensions
  const auto input0_height = *(input0_dims.rbegin()+1);
  const auto input0_width = *(input0_dims.rbegin());
  const auto input1_height = *(input1_dims.rbegin()+1);
  const auto input1_width = *(input1_dims.rbegin());
  if ((m_transpose_a ? input0_height : input0_width)
      != (m_transpose_b ? input1_width : input1_height)) {
    LBANN_ERROR("input tensors in ",print_name()," ",
                "are not compatible with ",
                (m_transpose_a ? "T" : "N"), (m_transpose_b ? "T" : "N"),
                " matrix multiplication ",
                "(",print_inputs(),")");
  }

  // Set output dimensions
  std::vector<int> output_dims(input0_dims);
  *(output_dims.rbegin()+1) = (m_transpose_a ? input0_width : input0_height);
  *(output_dims.rbegin()) = (m_transpose_b ? input1_height : input1_width);
  this->set_output_dims(output_dims);

}

// =========================================================
// Explicit template instantiation
// =========================================================

#ifndef LBANN_MATMUL_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device) \
  extern template class matmul_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_MATMUL_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_MATH_MATMUL_HPP_INCLUDED
