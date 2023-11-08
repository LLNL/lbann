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

#ifndef LBANN_LAYER_MATH_MATMUL_HPP_INCLUDED
#define LBANN_LAYER_MATH_MATMUL_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/layers/data_type_distconv_adapter.hpp"
#include "lbann/layers/math/distconv/distconv_matmul.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
namespace dc {
using Backend = ::distconv::BackendDNNLib;
template <typename TensorDataType>
using MatMul = ::distconv::MatMul<Backend, TensorDataType>;
} // namespace dc

template <typename TensorDataType, data_layout Layout, El::Device Device>
class matmul_distconv_adapter
  : public data_type_distconv_adapter<TensorDataType>
{
public:
  using TensorDevType =
    typename data_type_distconv_adapter<TensorDataType>::TensorDevType;

  matmul_distconv_adapter(Layer& layer)
    : data_type_distconv_adapter<TensorDataType>(layer)
  {}

  virtual ~matmul_distconv_adapter() = default;
  void setup_distributions(tensor_overlap_constraints& constraints) override;
  void setup_layer(size_t workspace_capacity) override;
  void fp_compute();
  void bp_compute();
  dc::Shape get_activations_local_shape(int index = 0) const override;
  std::unique_ptr<dc::MatMul<TensorDataType>> m_matmul_operator;
}; // class definition matmul_distconv_adapter

#endif // LBANN_HAS_DISTCONV

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
class matmul_layer : public data_type_layer<TensorDataType>
{
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "matmul_layer only supports "
                "data-parallel data layout");

public:
  matmul_layer(lbann_comm* comm,
               bool transpose_a = false,
               bool transpose_b = false);
  matmul_layer(const matmul_layer& other) = default;
  matmul_layer& operator=(const matmul_layer& other) = default;
  matmul_layer* copy() const override;

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS;
  }

  description get_description() const override;

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  matmul_layer() : matmul_layer(nullptr, false, false) {}

  void setup_dims() override;
  void fp_compute() override;
  void bp_compute() override;

#ifdef LBANN_HAS_DISTCONV
  friend class matmul_distconv_adapter<TensorDataType, Layout, Device>;

protected:
  void setup_distconv_adapter() override;
  bool is_distconv_supported() const override;
  matmul_distconv_adapter<TensorDataType, Layout, Device>&
  get_distconv_adapter() override;
  const matmul_distconv_adapter<TensorDataType, Layout, Device>&
  get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV

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
matmul_layer<TensorDataType, Layout, Device>::matmul_layer(lbann_comm* comm,
                                                           bool transpose_a,
                                                           bool transpose_b)
  : data_type_layer<TensorDataType>(comm),
    m_transpose_a{transpose_a},
    m_transpose_b{transpose_b}
{
  this->m_expected_num_parent_layers = 2;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
matmul_layer<TensorDataType, Layout, Device>*
matmul_layer<TensorDataType, Layout, Device>::copy() const
{
  return new matmul_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string matmul_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "matrix multiply";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
matmul_layer<TensorDataType, Layout, Device>::get_data_layout() const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device
matmul_layer<TensorDataType, Layout, Device>::get_device_allocation() const
{
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
description
matmul_layer<TensorDataType, Layout, Device>::get_description() const
{
  auto desc = data_type_layer<TensorDataType>::get_description();
  desc.add("Transpose A", m_transpose_a);
  desc.add("Transpose B", m_transpose_b);
  return desc;
}

// =========================================================
// Explicit template instantiation
// =========================================================

#ifndef LBANN_MATMUL_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                                                \
  extern template class matmul_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_MATMUL_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_MATH_MATMUL_HPP_INCLUDED
