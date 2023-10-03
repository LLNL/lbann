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

#ifndef LBANN_LAYER_POOLING_HPP_INCLUDED
#define LBANN_LAYER_POOLING_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/dnn_enums.hpp"
#ifdef LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/helpers.hpp"
#include "lbann/utils/dnn_lib/pooling.hpp"
#endif // LBANN_HAS_DNN_LIB
#include "lbann/utils/exception.hpp"
#include "lbann/utils/im2col.hpp"

#include <utility>
#include <vector>

#ifdef LBANN_HAS_DISTCONV
#include "distconv/dnn_backend/pooling.hpp"
#include "lbann/utils/distconv.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

inline pooling_mode to_pool_mode(std::string m)
{
#ifdef LBANN_DETERMINISTIC
  if (m == "max")
    return pooling_mode::MAX_DETERMINISTIC;
#else
  if (m == "max")
    return pooling_mode::MAX;
#endif // LBANN_DETERMINISTIC
  if (m == "average")
    return pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING;
  if (m == "average_no_pad")
    return pooling_mode::AVERAGE_COUNT_EXCLUDE_PADDING;
  else {
    LBANN_ERROR("Invalid pooling mode requested.");
  }
}

#ifdef LBANN_HAS_DISTCONV

namespace dc {
using Shape = ::distconv::tensor::Shape;
using Backend = ::distconv::BackendDNNLib;
template <typename TensorDataType>
using Pooling = ::distconv::Pooling<Backend, TensorDataType>;
} // namespace dc

template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class pooling_distconv_adapter
  : public data_type_distconv_adapter<TensorDataType>
{
public:
  using TensorDevType =
    typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  pooling_distconv_adapter(Layer& layer)
    : data_type_distconv_adapter<TensorDataType>(layer)
  {}
  virtual ~pooling_distconv_adapter() = default;
  void setup_distributions(tensor_overlap_constraints& constraints) override;
  dc::Shape get_activations_local_shape(int index = 0) const override;
  void setup_layer(size_t workspace_capacity) override;
  void
  fp_compute(bool training = true); // training=true for max back-compatibility.
  void bp_compute();
  std::unique_ptr<dc::Pooling<TensorDataType>> m_pooling;
};
#endif // LBANN_HAS_DISTCONV

// Forward declaration
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class unpooling_layer;

template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class pooling_layer : public data_type_layer<TensorDataType>
{
  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "pooling only supports DATA_PARALLEL");

private:
  /** Pooling mode. */
  pooling_mode m_pool_mode;

  /** Pooling window dimensions. */
  std::vector<int> m_pool_dims;
  /** Size of pooling window. */
  int m_pool_size;
  /** Pooling padding. */
  std::vector<int> m_pads;
  /** Pooling strides. */
  std::vector<int> m_strides;

  /** Input indices for max pooling.
   *  Each entry corresponds to a local entry in the activations
   *  matrix. The entry gives the index of the maximum entry within
   *  the pooling window.
   */
  std::vector<int> m_max_pool_indices;

#ifdef LBANN_HAS_DNN_LIB
  /** Pooling descriptor. */
  dnn_lib::PoolingDescriptor m_pooling_dnn_desc;
  /** Tensor DNN library descriptors. */
  dnn_lib::data_parallel_layer_tensor_manager<TensorDataType>
    m_tensors_dnn_desc;
#endif // LBANN_HAS_DNN_LIB

  friend class unpooling_layer<TensorDataType, T_layout, Dev>;

public:
  pooling_layer(lbann_comm* comm,
                int num_data_dims,
                int pool_dim,
                int pad,
                int stride,
                pooling_mode mode)
    : pooling_layer(comm,
                    num_data_dims,
                    std::vector<int>(num_data_dims, pool_dim),
                    std::vector<int>(num_data_dims, pad),
                    std::vector<int>(num_data_dims, stride),
                    mode)
  {}

  pooling_layer(lbann_comm* comm,
                int num_data_dims,
                std::vector<int> pool_dims,
                std::vector<int> pads,
                std::vector<int> strides,
                pooling_mode mode)
    : data_type_layer<TensorDataType>(comm),
      m_pool_mode(mode),
      m_pool_dims(pool_dims),
      m_pads(pads),
      m_strides(strides)
#ifdef LBANN_HAS_DNN_LIB
      ,
      m_tensors_dnn_desc(this)
#endif // LBANN_HAS_DNN_LIB
  {
    // Initialize input dimensions and pooling parameters
    m_pool_size = get_linear_size(m_pool_dims);
  }

  pooling_layer(const pooling_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_pool_mode(other.m_pool_mode),
      m_pool_dims(other.m_pool_dims),
      m_pool_size(other.m_pool_size),
      m_pads(other.m_pads),
      m_strides(other.m_strides),
      m_max_pool_indices(other.m_max_pool_indices)
#ifdef LBANN_HAS_DNN_LIB
      ,
      m_pooling_dnn_desc(other.m_pooling_dnn_desc),
      m_tensors_dnn_desc(other.m_tensors_dnn_desc)
#endif // LBANN_HAS_DNN_LIB
  {
#ifdef LBANN_HAS_DNN_LIB
    m_tensors_dnn_desc.set_layer(this);
#endif // LBANN_HAS_DNN_LIB
  }

  pooling_layer& operator=(const pooling_layer& other)
  {
    data_type_layer<TensorDataType>::operator=(other);
    m_pool_mode = other.m_pool_mode;
    m_pool_dims = other.m_pool_dims;
    m_pool_size = other.m_pool_size;
    m_pads = other.m_pads;
    m_strides = other.m_strides;
    m_max_pool_indices = other.m_max_pool_indices;
#ifdef LBANN_HAS_DNN_LIB
    m_pooling_dnn_desc = other.m_pooling_dnn_desc;
    m_tensors_dnn_desc = other.m_tensors_dnn_desc;
    m_tensors_dnn_desc.set_layer(this);
#endif // LBANN_HAS_DNN_LIB
    return *this;
  }

  ~pooling_layer() override = default;

  pooling_layer* copy() const override { return new pooling_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "pooling"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS | ACTIVATIONS;
  }

#ifdef LBANN_HAS_ONNX
  void fill_onnx_node(onnx::GraphProto& graph) const override;
#endif // LBANN_HAS_ONNX

  description get_description() const override
  {
    auto desc = data_type_layer<TensorDataType>::get_description();
    std::stringstream ss;

    // Pool mode
    ss.str(std::string{});
    ss.clear();
    switch (m_pool_mode) {
    case pooling_mode::MAX:
      ss << "max";
      break;
    case pooling_mode::MAX_DETERMINISTIC:
      ss << "max (deterministic)";
      break;
    case pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING:
      ss << "average";
      break;
    case pooling_mode::AVERAGE_COUNT_EXCLUDE_PADDING:
      ss << "average (no pad)";
      break;
    default:
      ss << "invalid";
    }
    desc.add("Pool mode", ss.str());

    // Pool dimensions
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < m_pool_dims.size(); ++i) {
      ss << (i > 0 ? ", " : "") << m_pool_dims[i];
    }
    desc.add("Pool dimensions", ss.str());

    // Strides
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < m_strides.size(); ++i) {
      ss << (i > 0 ? ", " : "") << m_strides[i];
    }
    desc.add("Strides", ss.str());

    // Pads
    ss.str(std::string{});
    ss.clear();
    for (size_t i = 0; i < m_pads.size(); ++i) {
      ss << (i > 0 ? ", " : "") << m_pads[i];
    }
    desc.add("Pads", ss.str());

    // Result
    return desc;
  }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  pooling_layer() : pooling_layer(nullptr, 1, 1, 1, 1, pooling_mode::MAX) {}

  void setup_dims() override
  {
    data_type_layer<TensorDataType>::setup_dims();
    const auto& input_dims = this->get_input_dims();
    auto output_dims = input_dims;
    for (size_t i = 0; i < output_dims.size() - 1; ++i) {
      const int effective_dim =
        (input_dims[i + 1] + 2 * m_pads[i] - m_pool_dims[i] + 1);
      output_dims[i + 1] = (effective_dim + m_strides[i] - 1) / m_strides[i];
    }
    this->set_output_dims(output_dims);
  }

  /// Initialize GPU objects
  void setup_gpu() override
  {
    data_type_layer<TensorDataType>::setup_gpu();
#ifndef LBANN_HAS_DNN_LIB
    LBANN_ERROR("DNN library not detected");
#else

    // Set pooling descriptor
    m_pooling_dnn_desc.set(m_pool_mode,
                           dnn_lib::DNN_PROPAGATE_NAN,
                           m_pool_dims.size(),
                           m_pool_dims.data(),
                           m_pads.data(),
                           m_strides.data());

#endif // #ifndef LBANN_HAS_DNN_LIB
  }

  void fp_compute() override;

  void bp_compute() override;

private:
  /// Pooling forward propagation with DNN library
  void fp_compute_dnn();

  /// Pooling backward propagation with DNN library
  void bp_compute_dnn();

  /// Pooling forward propagation with im2col
  void fp_compute_im2col();

  /// Pooling forward propagation with im2col
  void bp_compute_im2col();

#ifdef LBANN_HAS_DISTCONV
  friend class pooling_distconv_adapter<TensorDataType, T_layout, Dev>;

protected:
  bool is_distconv_supported() const override;
  void setup_distconv_adapter() override
  {
    this->get_distconv_adapter_ptr() =
      std::make_unique<pooling_distconv_adapter<TensorDataType, T_layout, Dev>>(
        *this);
  }
  pooling_distconv_adapter<TensorDataType, T_layout, Dev>&
  get_distconv_adapter() override;
  const pooling_distconv_adapter<TensorDataType, T_layout, Dev>&
  get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV
};

#ifdef LBANN_HAS_ONNX
template <typename T, data_layout L, El::Device D>
void pooling_layer<T, L, D>::fill_onnx_node(onnx::GraphProto& graph) const
{
  auto* pool = graph.add_node();

  // Get the attributes setup first
  {
    auto* kernel_shape = pool->add_attribute();
    kernel_shape->set_name("kernel_shape");
    kernel_shape->set_type(onnx::AttributeProto::INTS);
    for (auto const& k : this->m_pool_dims)
      kernel_shape->add_ints(k);
  }
  if (!this->m_strides.empty()) {
    auto* strides = pool->add_attribute();
    strides->set_name("strides");
    strides->set_type(onnx::AttributeProto::INTS);
    for (auto const& s : this->m_strides)
      strides->add_ints(s);
  }
  if (!this->m_pads.empty()) {
    auto* pads = pool->add_attribute();
    pads->set_name("pads");
    pads->set_type(onnx::AttributeProto::INTS);
    for (auto const& p : this->m_pads) {
      pads->add_ints(p);
      pads->add_ints(p);
    }
  }
  // FIXME: This is missing "dilations". However, they're only a valid
  // attribute for MaxPool, not AveragePool.

  for (auto const* parent : this->get_parent_layers()) {
    size_t idx = parent->find_child_layer_index(*this);
    pool->add_input(parent->get_name() + "_" + std::to_string(idx));
  }
  for (size_t ii = 0; ii < this->num_weights(); ii++)
    pool->add_input(this->get_weights(ii).get_name());
  for (auto const* child : this->get_child_layers()) {
    size_t idx = this->find_child_layer_index(*child);
    pool->add_output(this->get_name() + "_" + std::to_string(idx));
  }
  pool->set_name(this->get_name());

  switch (m_pool_mode) {
  case pooling_mode::MAX:
    pool->set_op_type("MaxPool");
    break;
  case pooling_mode::MAX_DETERMINISTIC:
    pool->set_op_type("MaxPool");
    break;
  case pooling_mode::AVERAGE_COUNT_INCLUDE_PADDING:
    pool->set_op_type("AveragePool");
    break;
  case pooling_mode::AVERAGE_COUNT_EXCLUDE_PADDING:
    pool->set_op_type("AveragePool");
    break;
  default:
    LBANN_ERROR("pooling_layer: no ONNX implementation for pooling mode");
  }

  pool->set_domain("");
  pool->set_doc_string(this->get_type());
}
#endif

#ifndef LBANN_POOLING_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class pooling_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_POOLING_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_POOLING_HPP_INCLUDED
