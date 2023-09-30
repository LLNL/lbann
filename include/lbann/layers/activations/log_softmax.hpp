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

#ifndef LBANN_LAYERS_ACTIVATIONS_LOG_SOFTMAX_HPP_INCLUDED
#define LBANN_LAYERS_ACTIVATIONS_LOG_SOFTMAX_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#if defined LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/helpers.hpp"
#endif // LBANN_HAS_DNN_LIB
#include "lbann/proto/layers.pb.h"

namespace lbann {

/** @brief Logarithm of softmax function
 *
 *  @f[ \log \text{softmax}(x)_i = x_i - \log \sum_j e^{x_j} @f]
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class log_softmax_layer : public data_type_layer<TensorDataType>
{
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:
  log_softmax_layer() : log_softmax_layer(nullptr) {}
  log_softmax_layer(lbann_comm* comm)
    : data_type_layer<TensorDataType>(comm)
#ifdef LBANN_HAS_DNN_LIB
      ,
      m_tensors_dnn_desc(this)
#endif // LBANN_HAS_DNN_LIB
  {}

  log_softmax_layer(const log_softmax_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_workspace(other.m_workspace ? other.m_workspace->Copy() : nullptr)
#ifdef LBANN_HAS_DNN_LIB
      ,
      m_tensors_dnn_desc(other.m_tensors_dnn_desc)
#endif // LBANN_HAS_DNN_LIB
  {
#ifdef LBANN_HAS_DNN_LIB
    m_tensors_dnn_desc.set_layer(this);
#endif // LBANN_HAS_DNN_LIB
  }

  log_softmax_layer& operator=(const log_softmax_layer& other)
  {
    data_type_layer<TensorDataType>::operator=(other);
    m_workspace.reset(other.m_workspace ? other.m_workspace->Copy() : nullptr);
#ifdef LBANN_HAS_DNN_LIB
    m_tensors_dnn_desc = other.m_tensors_dnn_desc;
    m_tensors_dnn_desc.set_layer(this);
#endif // LBANN_HAS_DNN_LIB
    return *this;
  }

  ~log_softmax_layer() = default;

  log_softmax_layer* copy() const override
  {
    return new log_softmax_layer(*this);
  }
  std::string get_type() const override { return "log softmax"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  // Multi-stage log softmax can run in-place (local workspace acts as an
  // intermediate buffer)
  bool can_run_inplace() const override { return true; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | ACTIVATIONS;
  }

  void setup_dims() override
  {
    data_type_layer<TensorDataType>::setup_dims();
    this->set_output_dims(this->get_input_dims());
  }

  void setup_data(size_t max_mini_batch_size) override
  {
    data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);
    auto dist = this->get_prev_activations().DistData();
    dist.colDist = El::STAR;
    m_workspace.reset(AbsDistMatrixType::Instantiate(dist));
#ifdef HYDROGEN_HAVE_CUB
    if (m_workspace->GetLocalDevice() == El::Device::GPU) {
      m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // HYDROGEN_HAVE_CUB
  }

  void fp_compute() override;
  void bp_compute() override;

  template <typename U>
  friend void fp_compute_impl(log_softmax_layer<U, Layout, Device>& l);
  template <typename U>
  friend void bp_compute_impl(log_softmax_layer<U, Layout, Device>& l);

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

private:
  /** Workspace for column-wise reductions. */
  std::unique_ptr<AbsDistMatrixType> m_workspace;

#ifdef LBANN_HAS_DNN_LIB
  /** Tensor DNN library descriptors. */
  dnn_lib::data_parallel_layer_tensor_manager<TensorDataType>
    m_tensors_dnn_desc;
#endif // LBANN_HAS_DNN_LIB
};

template <typename T, data_layout L, El::Device D>
void log_softmax_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_log_softmax();
}

#ifndef LBANN_LOG_SOFTMAX_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class log_softmax_layer<T,                                   \
                                          data_layout::DATA_PARALLEL,          \
                                          Device>;                             \
  extern template class log_softmax_layer<T,                                   \
                                          data_layout::MODEL_PARALLEL,         \
                                          Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_LOG_SOFTMAX_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_ACTIVATIONS_LOG_SOFTMAX_HPP_INCLUDED
