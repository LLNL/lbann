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

#ifndef LBANN_LAYERS_ACTIVATIONS_SOFTMAX_HPP_INCLUDED
#define LBANN_LAYERS_ACTIVATIONS_SOFTMAX_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/utils/dnn_enums.hpp"
#if defined LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/helpers.hpp"
#include "lbann/utils/dnn_lib/softmax.hpp"
#endif // defined LBANN_HAS_DNN_LIB
#include "lbann/proto/layers.pb.h"
#include "lbann/utils/dnn_lib/softmax.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "distconv/dnn_backend/softmax.hpp"
#include "lbann/utils/distconv.hpp"
#endif

// Threshold outputs to a minimum value.

// If enabled, the minimum output value is sqrt(min), where min is the
// minimum, normalized, positive value (~1e-19 for float and ~1e-154
// for double). During backprop, gradients are computed as if
// thresholding did not occur, so there will be a discrepancy for
// values that are thresholded.
#define LBANN_ENABLE_SOFTMAX_THRESHOLD

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
namespace dc {
using Backend = ::distconv::BackendDNNLib;
using Softmax = ::distconv::Softmax<Backend>;
} // namespace dc

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class softmax_distconv_adapter
  : public data_type_distconv_adapter<TensorDataType>
{
public:
  using TensorDevType =
    typename data_type_distconv_adapter<TensorDataType>::TensorDevType;

  softmax_distconv_adapter(Layer& layer)
    : data_type_distconv_adapter<TensorDataType>(layer)
  {}
  virtual ~softmax_distconv_adapter() = default;

  void setup_distributions(tensor_overlap_constraints& constraints) override;
  void setup_layer(size_t workspace_capacity) override;

  std::unique_ptr<dc::Softmax> m_softmax;
};
#endif // LBANN_HAS_DISTCONV

/**
 *  @f[ \text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} @f]
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class softmax_layer : public data_type_layer<TensorDataType>
{
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:
  softmax_layer(lbann_comm* comm, softmax_mode mode)
    : data_type_layer<TensorDataType>(comm),
      m_mode(mode)
#ifdef LBANN_HAS_DNN_LIB
      ,
      m_tensors_dnn_desc(this)
#endif // LBANN_HAS_DNN_LIB
  {
    if (mode == softmax_mode::INVALID) {
      LBANN_ERROR("invalid softmax mode");
    }
  }

  softmax_layer(const softmax_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_mode(other.m_mode),
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

  ~softmax_layer() = default;

  softmax_layer* copy() const final { return new softmax_layer(*this); }
  std::string get_type() const final { return "softmax"; }
  data_layout get_data_layout() const final { return Layout; }
  El::Device get_device_allocation() const final { return Device; }

  // Softmax can run in-place (local workspace acts as an
  // intermediate buffer)
  bool can_run_inplace() const override { return true; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | ACTIVATIONS;
  }

#ifdef LBANN_HAS_ONNX
  std::string get_onnx_op_type() const override { return "Softmax"; }
#endif // LBANN_HAS_ONNX

  void setup_dims(DataReaderMetaData& dr_metadata) final
  {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
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
#ifdef LBANN_HAS_DNN_LIB
    if (!m_tensors_dnn_desc.get_layer())
      m_tensors_dnn_desc.set_layer(this);
#endif // LBANN_HAS_DNN_LIB
  }

  void fp_compute() final;
  void bp_compute() final;

  template <typename U>
  friend void fp_compute_impl(softmax_layer<U, Layout, Device>& l);
  template <typename U>
  friend void bp_compute_impl(softmax_layer<U, Layout, Device>& l);

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

private:
  /** @name DNN library stuff */
  ///@{

  // using dnn_backend = dnn_lib::get_backend<Device>;
#ifdef LBANN_HAS_ONEDNN_CPU
  using dnn_backend = onednn_backend<Device>;
#else
  using dnn_backend = openmp_backend;
#endif
  using dnnTensorDescriptor = typename dnn_backend::TensorDescriptor;

  /** @brief Descriptor for local input tensor
   *  @details Only used for data-parallel, CPU implementation.
   */
  dnnTensorDescriptor input_descriptor_;
  /** @brief Descriptor for local output tensor
   *  @details Only used for data-parallel, CPU implementation.
   */
  dnnTensorDescriptor output_descriptor_;
  /** @brief Descriptor for local input gradient tensor
   *  @details Only used for data-parallel, CPU implementation.
   */
  dnnTensorDescriptor grad_wrt_input_descriptor_;
  /** @brief Descriptor for local output gradient tensor
   *  @details Only used for data-parallel, CPU implementation.
   */
  dnnTensorDescriptor grad_wrt_output_descriptor_;

  ///@}

  friend cereal::access;
  softmax_layer() : data_type_layer<TensorDataType>(nullptr) {}

  /** Softmax mode. */
  softmax_mode m_mode;

  /** @brief Workspace for column-wise reductions
   *
   *  Only used for model-parallel implementation.
   */
  std::unique_ptr<AbsDistMatrixType> m_workspace;

#ifdef LBANN_HAS_DNN_LIB
  /** Tensor DNN library descriptors. */
  dnn_lib::data_parallel_layer_tensor_manager<TensorDataType>
    m_tensors_dnn_desc;
#endif // LBANN_HAS_DNN_LIB

// Minimum output value to avoid denormalized floats
#ifdef LBANN_ENABLE_SOFTMAX_THRESHOLD
  TensorDataType threshold_val = static_cast<TensorDataType>(
    El::Sqrt(std::numeric_limits<TensorDataType>::min()));
#else
  TensorDataType threshold_val = El::TypeTraits<TensorDataType>::Zero();
#endif // LBANN_ENABLE_SOFTMAX_THRESHOLD

#ifdef LBANN_HAS_DISTCONV
  friend class softmax_distconv_adapter<TensorDataType, Layout, Device>;

protected:
  bool is_distconv_supported() const final
  {
    return Device == El::Device::GPU && Layout == data_layout::DATA_PARALLEL;
  }
  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) final
  {
    this->get_distconv_adapter_ptr() = std::make_unique<
      softmax_distconv_adapter<TensorDataType, Layout, Device>>(*this);
  }
  softmax_distconv_adapter<TensorDataType, Layout, Device>&
  get_distconv_adapter() final;
  const softmax_distconv_adapter<TensorDataType, Layout, Device>&
  get_distconv_adapter() const final;
#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_SOFTMAX_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class softmax_layer<T, data_layout::DATA_PARALLEL, Device>;  \
  extern template class softmax_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_SOFTMAX_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_ACTIVATIONS_SOFTMAX_HPP_INCLUDED
