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

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/distconv.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/im2col.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/timer.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/variance_scaling_initializers.hpp"

#include <omp.h>

#include <sstream>
#include <string>
#include <vector>

namespace lbann {

template <typename TensorDataType, El::Device Device>
base_convolution_layer<TensorDataType,Device>::base_convolution_layer(
  lbann_comm* comm,
  int num_data_dims,
  int output_channels,
  std::vector<int> conv_dims,
  std::vector<int> pads,
  std::vector<int> strides,
  std::vector<int> dilations,
  int groups,
  bool has_bias)
  : data_type_layer<TensorDataType>(comm),
  m_output_channels(output_channels),
  m_conv_dims(std::move(conv_dims)),
  m_pads(std::move(pads)),
  m_strides(std::move(strides)),
  m_dilations(std::move(dilations)),
  m_groups(groups),
  m_bias_scaling_factor(has_bias
                        ? El::TypeTraits<ScalingType>::One()
                        : El::TypeTraits<ScalingType>::Zero())
#ifdef LBANN_HAS_CUDNN
  , m_tensors_cudnn_desc(this)
#endif // LBANN_HAS_CUDNN
{}
template <typename TensorDataType, El::Device Device>
base_convolution_layer<TensorDataType,Device>::base_convolution_layer(
  const base_convolution_layer& other)
  : data_type_layer<TensorDataType>(other),
  m_output_channels(other.m_output_channels),
  m_conv_dims(other.m_conv_dims),
  m_pads(other.m_pads),
  m_strides(other.m_strides),
  m_dilations(other.m_dilations),
  m_groups(other.m_groups),
  m_bias_scaling_factor(other.m_bias_scaling_factor)
#ifdef LBANN_HAS_CUDNN
  , m_convolution_math_type(other.m_convolution_math_type),
  m_tensors_cudnn_desc(other.m_tensors_cudnn_desc),
  m_fwd_cudnn_algos(other.m_fwd_cudnn_algos),
  m_bwd_data_cudnn_algos(other.m_bwd_data_cudnn_algos),
  m_bwd_filter_cudnn_algos(other.m_bwd_filter_cudnn_algos)
#endif // LBANN_HAS_CUDNN
{
#ifdef LBANN_HAS_CUDNN
  m_kernel_cudnn_desc = other.m_kernel_cudnn_desc;
  m_convolution_cudnn_desc = other.m_convolution_cudnn_desc;
  if (other.m_bias_scaling_factor != El::TypeTraits<ScalingType>::Zero()) {
    m_bias_cudnn_desc = other.m_bias_cudnn_desc;
  }
  m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
}

template <typename TensorDataType, El::Device Device>
auto
base_convolution_layer<TensorDataType,Device>
::operator=(const base_convolution_layer& other)
  -> base_convolution_layer& {
  data_type_layer<TensorDataType>::operator=(other);
  m_output_channels = other.m_output_channels;
  m_conv_dims = other.m_conv_dims;
  m_pads = other.m_pads;
  m_strides = other.m_strides;
  m_dilations = other.m_dilations;
  m_groups = other.m_groups;
  m_bias_scaling_factor = other.m_bias_scaling_factor;

#ifdef LBANN_HAS_CUDNN
  // Copy cuDNN objects
  m_convolution_math_type = other.m_convolution_math_type;
  m_kernel_cudnn_desc = other.m_kernel_cudnn_desc;
  m_convolution_cudnn_desc = other.m_convolution_cudnn_desc;
  if (other.m_bias_scaling_factor != El::TypeTraits<ScalingType>::Zero()) {
    m_bias_cudnn_desc = other.m_bias_cudnn_desc;
  }
  m_tensors_cudnn_desc = other.m_tensors_cudnn_desc;
  m_tensors_cudnn_desc.set_layer(this);
  m_fwd_cudnn_algos = other.m_fwd_cudnn_algos;
  m_bwd_data_cudnn_algos = other.m_bwd_data_cudnn_algos;
  m_bwd_filter_cudnn_algos = other.m_bwd_filter_cudnn_algos;
#endif // LBANN_HAS_CUDNN

  return *this;
}

template <typename TensorDataType, El::Device Device>
base_convolution_layer<TensorDataType,Device>::~base_convolution_layer() {}

#ifdef LBANN_HAS_CUDNN
template <typename TensorDataType, El::Device Device>
void
base_convolution_layer<TensorDataType,Device>
::set_cudnn_math_mode(cudnnMathType_t math_type) noexcept {
  m_convolution_math_type = math_type;
}
#endif // LBANN_HAS_CUDNN

template <typename TensorDataType, El::Device Device>
description
base_convolution_layer<TensorDataType,Device>::get_description() const {
  auto desc = data_type_layer<TensorDataType>::get_description();
  std::ostringstream ss;

  // Convolution dimensions
  ss.str(std::string{});
  ss.clear();
  for (size_t i = 0; i < m_conv_dims.size(); ++i) {
    ss << (i > 0 ? ", " : "" ) << m_conv_dims[i];
  }
  desc.add("Convolution dimensions", ss.str());

  // Strides
  ss.str(std::string{});
  ss.clear();
  for (size_t i = 0; i < m_strides.size(); ++i) {
    ss << (i > 0 ? ", " : "" ) << m_strides[i];
  }
  desc.add("Strides", ss.str());

  // Pads
  ss.str(std::string{});
  ss.clear();
  for (size_t i = 0; i < m_pads.size(); ++i) {
    ss << (i > 0 ? ", " : "" ) << m_pads[i];
  }
  desc.add("Pads", ss.str());

  // Dilation
  ss.str(std::string{});
  ss.clear();
  for (size_t i = 0; i < m_dilations.size(); ++i) {
    ss << (i > 0 ? ", " : "" ) << m_dilations[i];
  }
  desc.add("Dilations", ss.str());

  // Groups
  desc.add("Groups", m_groups);

  // Bias
  ss.str(std::string{});
  ss.clear();
  ss << (m_bias_scaling_factor == El::TypeTraits<ScalingType>::Zero() ?
         "disabled" : "enabled");
  desc.add("Bias", ss.str());

#ifdef LBANN_HAS_CUDNN
  if (Device == El::Device::GPU) {
    desc.add("cuDNN Math Mode",
             (m_convolution_math_type == CUDNN_DEFAULT_MATH
              ? "NO tensor cores."
              : "USE tensor cores."));
  }
#endif // LBANN_HAS_CUDNN

  // Result
  return desc;
}

template <typename TensorDataType, El::Device Device>
void
base_convolution_layer<TensorDataType,Device>
::setup_dims(DataReaderMetaData& dr_metadata) {
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  std::ostringstream err;

  // Check number of channels and channel groups
  const auto& input_dims = this->get_input_dims();
  if (m_output_channels < 1) {
    err << this->get_type() << " layer \"" << this->get_name() << "\" "
        << "has an invalid number of output channels "
        << "(" << m_output_channels << ")";
    LBANN_ERROR(err.str());
  } else if (m_groups < 1) {
    err << this->get_type() << " layer \"" << this->get_name() << "\" "
        << "has an invalid number of groups (" << m_groups << ")";
    LBANN_ERROR(err.str());
  } else if (input_dims[0] % m_groups != 0
             || m_output_channels % m_groups != 0) {
    err << this->get_type() << " layer \"" << this->get_name() << "\" "
        << "has " << m_groups << " groups, which does not divide "
        << "the input channels (" << input_dims[0] << ") or "
        << "the output channels (" << m_output_channels << ")";
    LBANN_ERROR(err.str());
  }

  // Check kernel dims, pads, stride, dilations
  const auto& num_spatial_dims = input_dims.size() - 1;
  if (m_conv_dims.size() != num_spatial_dims
      || std::any_of(m_conv_dims.begin(), m_conv_dims.end(),
                     [](El::Int d) { return d < 1; })) {
    err << this->get_type() << " layer \"" << this->get_name() << "\" "
        << "has invalid spatial dimensions for convolution kernel (";
    if (m_conv_dims.empty()) { err << "no dimensions"; }
    for (size_t i = 0; i < m_conv_dims.size(); ++i) {
      err << (i > 0 ? "x" : "") << m_conv_dims[i];
    }
    err << ", expected " << num_spatial_dims << " spatial dimensions)";
    LBANN_ERROR(err.str());
  } else if (m_pads.size() != num_spatial_dims) {
    err << this->get_type() << " layer \"" << this->get_name() << "\" "
        << "has invalid convolution pads ((";
    for (size_t i = 0; i < m_pads.size(); ++i) {
      err << (i > 0 ? "," : "") << m_pads[i];
    }
    err << "), expected " << num_spatial_dims << " spatial dimensions)";
    LBANN_ERROR(err.str());
  } else if (m_strides.size() != num_spatial_dims
             || std::any_of(m_strides.begin(), m_strides.end(),
                            [](El::Int d) { return d < 1; })) {
    err << this->get_type() << " layer \"" << this->get_name() << "\" "
        << "has invalid convolution strides ((";
    for (size_t i = 0; i < m_strides.size(); ++i) {
      err << (i > 0 ? "," : "") << m_strides[i];
    }
    err << "), expected " << num_spatial_dims << " spatial dimensions)";
    LBANN_ERROR(err.str());
  } else if (m_dilations.size() != num_spatial_dims
             || std::any_of(m_dilations.begin(), m_dilations.end(),
                            [](El::Int d) { return d < 1; })) {
    err << this->get_type() << " layer \"" << this->get_name() << "\" "
        << "has invalid convolution dilations ((";
    for (size_t i = 0; i < m_dilations.size(); ++i) {
      err << (i > 0 ? "," : "") << m_dilations[i];
    }
    err << "), expected " << num_spatial_dims << " spatial dimensions)";
    LBANN_ERROR(err.str());
  }

  // Make sure that configuration is supported
  if (Device == El::Device::CPU
      && std::any_of(m_dilations.begin(), m_dilations.end(),
                     [](El::Int d) { return d != 1; })) {
    err << this->get_type() << " layer \"" << this->get_name() << "\" "
        << "has non-unit dilation, which is not yet supported on CPU";
    LBANN_ERROR(err.str());
  }
  if (Device == El::Device::CPU && m_groups != 1) {
    err << this->get_type() << " layer \"" << this->get_name() << "\" "
        << "has " << m_groups << " groups, "
        << "but only one group is currently supported on CPU";
    LBANN_ERROR(err.str());
  }

}

template <typename TensorDataType, El::Device Device>
void
base_convolution_layer<TensorDataType,Device>
::setup_data(size_t max_mini_batch_size) {
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

  // Tensor dimensions
  const auto& input_dims = this->get_input_dims();
  const auto& output_dims = this->get_output_dims();
  const auto& kernel_dims = this->get_kernel_dims();
  const auto& kernel_size = std::accumulate(kernel_dims.begin(),
                                            kernel_dims.end(),
                                            1, std::multiplies<int>());

  // Initialize default weights if none are provided
  if (this->num_weights() > 2) {
    LBANN_ERROR("attempted to setup layer \"", this->get_name(), "\" "
                "with an invalid number of weights "
                "(expected at most 2, "
                "found ", this->num_weights(), ")");
  }
  if (m_bias_scaling_factor != El::TypeTraits<ScalingType>::Zero()) {
    this->set_num_weights(2);
  } else {
    this->set_num_weights(1);
  }
  if (!this->has_weights(0)) {
    auto w = make_unique<WeightsType>(this->get_comm());
    auto init = make_unique<he_initializer<TensorDataType>>(probability_distribution::gaussian);
    auto opt = this->m_model->template create_optimizer<TensorDataType>();

    w->set_name(this->get_name() + "_kernel");
    w->set_initializer(std::move(init));
    w->set_optimizer(std::move(opt));
    this->set_weights(0, w.get());
    this->m_model->add_weights(std::move(w));
  }
  auto& kernel_weights = this->get_weights(0);

  // Initialize variance scaling initialization
  if (auto* initializer = kernel_weights.get_initializer()) {
    set_fan_in(*initializer, kernel_size / output_dims[0]);
    set_fan_out(*initializer, kernel_size / input_dims[0]);
  }

  // Initialize weight matrices
  auto dist = this->get_prev_activations().DistData();
  dist.colDist = El::STAR;
  dist.rowDist = El::STAR;
  kernel_weights.set_dims(kernel_dims);
  kernel_weights.set_matrix_distribution(dist);

  // Set up bias if needed.
  if (m_bias_scaling_factor != El::TypeTraits<ScalingType>::Zero()) {
    if (!this->has_weights(1)) {
      auto w = make_unique<WeightsType>(this->get_comm());
      auto opt = this->m_model->template create_optimizer<TensorDataType>();
      w->set_name(this->get_name() + "_bias");
      w->set_optimizer(std::move(opt));
      this->set_weights(1, w.get());
      this->m_model->add_weights(std::move(w));
    }
    auto& bias_weights = this->get_weights(1);
    bias_weights.set_dims(output_dims[0]);
    bias_weights.set_matrix_distribution(dist);
  }

  // Initialize freeze state
  auto const num_weights = this->num_weights();
  for (size_t ii = 0; ii < num_weights; ++ii) {
    auto& w = this->get_weights(ii);
    if (this->m_frozen) {
      w.freeze();
    } else {
      w.unfreeze();
    }
  }
  for (size_t ii = 0; ii < num_weights; ++ii) {
    auto& w = this->get_weights(ii);
    if (w.is_frozen() != this->m_frozen) {
      LBANN_ERROR((this->m_frozen ? "" : "un"), "frozen layer "
                  "\"", this->get_name(), "\" has ",
                  (w.is_frozen() ? "" : "un"), "frozen weights ",
                  "\"", w.get_name(), "\"");
    }
  }

}

template <typename TensorDataType, El::Device Device>
void base_convolution_layer<TensorDataType,Device>::setup_gpu() {
  data_type_layer<TensorDataType>::setup_gpu();
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else

  const auto& output_dims = this->get_output_dims();
  const auto& kernel_dims = this->get_kernel_dims();

  // Set kernel descriptor
  m_kernel_cudnn_desc.set(
    cudnn::get_data_type<TensorDataType>(),
    CUDNN_TENSOR_NCHW,
    kernel_dims);

  // Set convolution descriptor
  m_convolution_cudnn_desc.set(m_pads, m_strides, m_dilations,
                               cudnn::get_data_type<TensorDataType>(),
                               CUDNN_CROSS_CORRELATION);
  m_convolution_cudnn_desc.set_math_mode(m_convolution_math_type);
  m_convolution_cudnn_desc.set_group_count(m_groups);

  // Set bias tensor descriptor
  if (m_bias_scaling_factor != El::TypeTraits<ScalingType>::Zero()) {
    std::vector<int> bias_dims(output_dims.size() + 1, 1);
    bias_dims[1] = output_dims[0];
    m_bias_cudnn_desc.set(
      cudnn::get_data_type<TensorDataType>(), bias_dims);
  }

#endif // LBANN_HAS_CUDNN
}

template <typename TensorDataType, El::Device Device>
void
base_convolution_layer<TensorDataType,Device>
::apply_convolution_cudnn(bool during_forward_prop) {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else

  // Useful constants
  const auto zero = El::TypeTraits<ScalingType>::Zero();
  const auto one = El::TypeTraits<ScalingType>::One();

  // Matrices
  const auto& kernel = this->weights_values(0);
  const auto& input = (during_forward_prop ?
                       this->get_local_prev_activations() :
                       this->get_local_prev_error_signals());
  auto& output = (during_forward_prop ?
                  this->get_local_activations() :
                  this->get_local_error_signals());

  // Do nothing if there is no local data
  if (input.Height() < 1 || input.Width() < 1
      || output.Height() < 1 || output.Width() < 1) {
    return;
  }

  // Initialize GPU workspace
  El::Matrix<TensorDataType, El::Device::GPU> workspace;
#ifdef HYDROGEN_HAVE_CUB
  workspace.SetMemoryMode(1);
#endif // HYDROGEN_HAVE_CUB
  size_t workspace_size = 1 << 30; /// @todo Allocate largest free block
  workspace.Resize(workspace_size / sizeof(TensorDataType), 1);
  workspace_size = workspace.Height() * sizeof(TensorDataType);

  // Convolution parameters
  std::vector<int> input_dims, output_dims;
  cudnn::TensorDescriptor input_desc, output_desc;
  if (during_forward_prop) {
    input_dims = this->get_input_dims();
    output_dims = this->get_output_dims();
    input_desc = m_tensors_cudnn_desc.get_prev_activations();
    output_desc = m_tensors_cudnn_desc.get_activations();
  }
  else {
    input_dims = this->get_output_dims();
    output_dims = this->get_input_dims();
    input_desc = m_tensors_cudnn_desc.get_prev_error_signals();
    output_desc = m_tensors_cudnn_desc.get_error_signals();
  }

  // Perform convolution on the GPU
  // Determine convolution algorithm
  cudnnConvolutionFwdAlgo_t convolution_cudnn_algorithm
                       = get_forward_algo_cudnn(input.Width(), input_desc, input.LockedBuffer(),
                                                m_kernel_cudnn_desc, kernel.LockedBuffer(),
                                                m_convolution_cudnn_desc,
                                                output_desc, output.Buffer(),
                                                workspace_size, workspace.Buffer());

  // Apply convolution
  CHECK_CUDNN(cudnnConvolutionForward(cudnn::get_handle(),
                                      &one,
                                      input_desc,
                                      input.LockedBuffer(),
                                      m_kernel_cudnn_desc,
                                      kernel.LockedBuffer(),
                                      m_convolution_cudnn_desc,
                                      convolution_cudnn_algorithm,
                                      workspace.Buffer(),
                                      workspace_size,
                                      &zero,
                                      output_desc,
                                      output.Buffer()));

#endif // LBANN_HAS_CUDNN
}

template <typename TensorDataType, El::Device Device>
void
base_convolution_layer<TensorDataType,Device>::
apply_transposed_convolution_cudnn(bool during_forward_prop) {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else

  // Useful constants
  const auto zero = El::TypeTraits<ScalingType>::Zero();
  const auto one = El::TypeTraits<ScalingType>::One();

  // GPU data
  const auto& kernel = this->weights_values(0);
  const auto& input = (during_forward_prop ?
                       this->get_local_prev_activations() :
                       this->get_local_prev_error_signals());
  auto& output = (during_forward_prop ?
                  this->get_local_activations() :
                  this->get_local_error_signals());

  // Do nothing if there is no local data
  if (input.Height() < 1 || input.Width() < 1
      || output.Height() < 1 || output.Width() < 1) {
    return;
  }

  // Initialize GPU workspace
  // Note: Use CUB GPU memory pool if possible
  El::Matrix<TensorDataType, El::Device::GPU> workspace;
#ifdef HYDROGEN_HAVE_CUB
  workspace.SetMemoryMode(1);
#endif // HYDROGEN_HAVE_CUB
  size_t workspace_size = 1 << 30; /// @todo Allocate largest free block
  workspace.Resize(workspace_size / sizeof(TensorDataType), 1);
  workspace_size = workspace.Height() * sizeof(TensorDataType);

  // Convolution transpose parameters
  std::vector<int> input_dims, output_dims;
  cudnn::TensorDescriptor input_desc, output_desc;
  if (during_forward_prop) {
    input_dims = this->get_input_dims();
    output_dims = this->get_output_dims();
    input_desc = m_tensors_cudnn_desc.get_prev_activations();
    output_desc = m_tensors_cudnn_desc.get_activations();
  }
  else {
    input_dims = this->get_output_dims();
    output_dims = this->get_input_dims();
    input_desc = m_tensors_cudnn_desc.get_prev_error_signals();
    output_desc = m_tensors_cudnn_desc.get_error_signals();
  }

  // Perform transposed convolution on the GPU
  // Determine transposed convolution algorithm
  cudnnConvolutionBwdDataAlgo_t transposed_convolution_cudnn_algorithm
                       = get_backward_data_algo_cudnn(input.Width(),
                                                      m_kernel_cudnn_desc, kernel.LockedBuffer(),
                                                      input_desc, input.LockedBuffer(),
                                                      m_convolution_cudnn_desc,
                                                      output_desc, output.Buffer(),
                                                      workspace_size, workspace.Buffer());
  // Perform transposed convolution
  CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn::get_handle(),
                                           &one,
                                           m_kernel_cudnn_desc,
                                           kernel.LockedBuffer(),
                                           input_desc,
                                           input.LockedBuffer(),
                                           m_convolution_cudnn_desc,
                                           transposed_convolution_cudnn_algorithm,
                                           workspace.Buffer(),
                                           workspace_size,
                                           &zero,
                                           output_desc,
                                           output.Buffer()));


#endif // LBANN_HAS_CUDNN
}

template <typename TensorDataType, El::Device Device>
void base_convolution_layer<TensorDataType,Device>::apply_bias_cudnn() {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else
  auto& local_output = this->get_local_activations();
  if (m_bias_scaling_factor != El::TypeTraits<ScalingType>::Zero()
      && local_output.Height() > 0
      && local_output.Width() > 0) {
    const auto one = El::TypeTraits<ScalingType>::One();
    const auto& bias = this->weights_values(1);
    CHECK_CUDNN(cudnnAddTensor(cudnn::get_handle(),
                               &m_bias_scaling_factor,
                               m_bias_cudnn_desc,
                               bias.LockedBuffer(),
                               &one,
                               m_tensors_cudnn_desc.get_activations(),
                               local_output.Buffer()));
  }
#endif // LBANN_HAS_CUDNN
}

template <typename TensorDataType, El::Device Device>
void
base_convolution_layer<TensorDataType,Device>
::compute_gradients_cudnn(bool using_transposed_convolution) {
#ifndef LBANN_HAS_CUDNN
  LBANN_ERROR("cuDNN not detected");
#else

  // Matrices
  const auto& local_input = this->get_local_prev_activations();
  const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();

  const bool has_local_data = (local_input.Height() > 0
                               && local_input.Width() > 0
                               && local_gradient_wrt_output.Height() > 0
                               && local_gradient_wrt_output.Width() > 0);

  // Compute bias gradient
  if (m_bias_scaling_factor != El::TypeTraits<ScalingType>::Zero()
      && this->get_weights(1).get_optimizer() != nullptr) {
    auto* bias_optimizer = this->get_weights(1).get_optimizer();
    auto dst_scale_dt = El::TypeTraits<TensorDataType>::Zero(),
      gradient_scale_dt = El::TypeTraits<TensorDataType>::Zero();
    auto& bias_gradient = bias_optimizer->get_gradient_buffer(
      dst_scale_dt, gradient_scale_dt, true);
    if (has_local_data) {
      auto dst_scale = ScalingType(dst_scale_dt), gradient_scale = ScalingType(gradient_scale_dt);
      CHECK_CUDNN(cudnnConvolutionBackwardBias(
                    cudnn::get_handle(),
                    &gradient_scale,
                    m_tensors_cudnn_desc.get_prev_error_signals(),
                    local_gradient_wrt_output.LockedBuffer(),
                    &dst_scale,
                    m_bias_cudnn_desc,
                    bias_gradient.Buffer()));
    } else {
      El::Scale(dst_scale_dt, bias_gradient);
    }
  }

  // Compute kernel gradient
  auto* kernel_optimizer = this->get_weights(0).get_optimizer();
  if (kernel_optimizer != nullptr) {
    auto dst_scale_dt = El::TypeTraits<TensorDataType>::Zero(), gradient_scale_dt = El::TypeTraits<TensorDataType>::Zero();
    auto& kernel_gradient = kernel_optimizer->get_gradient_buffer(
      dst_scale_dt, gradient_scale_dt, true);
    if (has_local_data) {
      // Initialize GPU workspace
      El::Matrix<TensorDataType, El::Device::GPU> workspace;
#ifdef HYDROGEN_HAVE_CUB
      workspace.SetMemoryMode(1); // CUB GPU memory pool
#endif // HYDROGEN_HAVE_CUB
      size_t workspace_size = 1 << 30; /// @todo Allocate largest free block
      workspace.Resize(workspace_size / sizeof(TensorDataType), 1);
      workspace_size = workspace.Height() * sizeof(TensorDataType);

      // Initialize cuDNN objects
      auto&& input_desc = m_tensors_cudnn_desc.get_prev_activations();
      auto&& gradient_wrt_output_desc = m_tensors_cudnn_desc.get_prev_error_signals();

      auto dst_scale = ScalingType(dst_scale_dt), gradient_scale = ScalingType(gradient_scale_dt);

      // Determine algorithm and compute kernel gradient
      if (using_transposed_convolution) {
        cudnnConvolutionBwdFilterAlgo_t kernel_gradient_cudnn_algorithm
          = get_backward_filter_algo_cudnn(
            local_input.Width(),
            gradient_wrt_output_desc, local_gradient_wrt_output.LockedBuffer(),
            input_desc, local_input.LockedBuffer(),
            m_convolution_cudnn_desc,
            m_kernel_cudnn_desc,
            workspace_size, workspace.Buffer());
        CHECK_CUDNN(cudnnConvolutionBackwardFilter(
                      cudnn::get_handle(),
                      &gradient_scale,
                      gradient_wrt_output_desc,
                      local_gradient_wrt_output.LockedBuffer(),
                      input_desc,
                      local_input.LockedBuffer(),
                      m_convolution_cudnn_desc,
                      kernel_gradient_cudnn_algorithm,
                      workspace.Buffer(),
                      workspace_size,
                      &dst_scale,
                      m_kernel_cudnn_desc,
                      kernel_gradient.Buffer()));
      } else {
        cudnnConvolutionBwdFilterAlgo_t kernel_gradient_cudnn_algorithm
          = get_backward_filter_algo_cudnn(
            local_input.Width(),
            input_desc, local_input.LockedBuffer(),
            gradient_wrt_output_desc, local_gradient_wrt_output.LockedBuffer(),
            m_convolution_cudnn_desc,
            m_kernel_cudnn_desc,
            workspace_size, workspace.Buffer());
        CHECK_CUDNN(cudnnConvolutionBackwardFilter(
                      cudnn::get_handle(),
                      &gradient_scale,
                      input_desc,
                      local_input.LockedBuffer(),
                      gradient_wrt_output_desc,
                      local_gradient_wrt_output.LockedBuffer(),
                      m_convolution_cudnn_desc,
                      kernel_gradient_cudnn_algorithm,
                      workspace.Buffer(),
                      workspace_size,
                      &dst_scale,
                      m_kernel_cudnn_desc,
                      kernel_gradient.Buffer()));
      }
    } else {
      El::Scale(dst_scale_dt, kernel_gradient);
    }
  }

#endif // LBANN_HAS_CUDNN
}

template <typename TensorDataType, El::Device Device>
void
base_convolution_layer<TensorDataType,Device>
::apply_convolution_im2col(bool during_forward_prop) {

  // Local matrices
  const auto& local_kernel = this->weights_values(0).LockedMatrix();
  const auto& local_input = (during_forward_prop ?
                             this->get_local_prev_activations() :
                             this->get_local_prev_error_signals());
  auto& local_output = (during_forward_prop ?
                        this->get_local_activations() :
                        this->get_local_error_signals());

  // Matrix parameters
  const int output_size = local_output.Height();
  const El::Int local_width = local_input.Width();
  std::vector<int> input_dims, output_dims;
  if (during_forward_prop) {
    input_dims = this->get_input_dims();
    output_dims = this->get_output_dims();
  }
  else {
    input_dims = this->get_output_dims();
    output_dims = this->get_input_dims();
  }
  const auto& kernel_dims = this->get_kernel_dims();
  const auto& kernel_size = std::accumulate(kernel_dims.begin(),
                                            kernel_dims.end(),
                                            1, std::multiplies<int>());

  // Initialize matrices
  const int m = output_size / output_dims[0];
  const int n = output_dims[0];
  const int k = kernel_size / output_dims[0];
  DMatDT<Device> input_col, output_col;
  DMatDT<Device> im2col_matrix(k, m);
  const DMatDT<Device> kernel_matrix(k, n, local_kernel.LockedBuffer(), k);

  // Iterate through input columns
  for (El::Int col = 0; col < local_width; ++col) {

    // Construct im2col matrix from current input column
    El::LockedView(input_col, local_input, El::ALL, El::IR(col));
    im2col<TensorDataType>(input_col,
                           im2col_matrix,
                           input_dims[0],
                           input_dims.size() - 1,
                           &input_dims[1],
                           m_pads.data(),
                           &kernel_dims[2],
                           m_strides.data());

    // Apply convolution to current input column
    output_col.Attach(m, n, local_output.Buffer(0, col), m);
    El::Gemm(El::TRANSPOSE, El::NORMAL,
             El::TypeTraits<TensorDataType>::One(), im2col_matrix, kernel_matrix,
             El::TypeTraits<TensorDataType>::Zero(), output_col);

  }

}

template <typename TensorDataType, El::Device Device>
void
base_convolution_layer<TensorDataType,Device>
::apply_transposed_convolution_im2col(bool during_forward_prop) {

  // Local matrices
  const auto& local_kernel = this->weights_values(0).LockedMatrix();
  const auto& local_input = (during_forward_prop ?
                             this->get_local_prev_activations() :
                             this->get_local_prev_error_signals());
  DMatDT<Device>& local_output = (during_forward_prop ?
                                  this->get_local_activations() :
                                  this->get_local_error_signals());

  // Matrix parameters
  const int input_size = local_input.Height();
  const El::Int local_width = local_input.Width();
  std::vector<int> input_dims, output_dims;
  if (during_forward_prop) {
    input_dims = this->get_input_dims();
    output_dims = this->get_output_dims();
  }
  else {
    input_dims = this->get_output_dims();
    output_dims = this->get_input_dims();
  }
  const auto& kernel_dims = this->get_kernel_dims();
  const auto& kernel_size = std::accumulate(kernel_dims.begin(),
                                            kernel_dims.end(),
                                            1, std::multiplies<int>());

  // Initialize matrices
  const int m = kernel_size / input_dims[0];
  const int n = input_size / input_dims[0];
  const int k = input_dims[0];
  DMatDT<Device> input_col, output_col;
  DMatDT<Device> im2col_matrix(m, n);
  const DMatDT<Device> kernel_matrix(m, k, local_kernel.LockedBuffer(), m);

  // Iterate through input columns
  for (El::Int col = 0; col < local_width; ++col) {

    // Apply transposed convolution to current input column
    input_col.LockedAttach(n, k, local_input.LockedBuffer(0, col), n);
    El::Gemm(El::NORMAL, El::TRANSPOSE,
             El::TypeTraits<TensorDataType>::One(), kernel_matrix, input_col,
             El::TypeTraits<TensorDataType>::Zero(), im2col_matrix);

    // Perform col2im to accumulate contributions from each kernel
    // position
    El::View(output_col, local_output, El::ALL, El::IR(col));
    col2im<TensorDataType>(im2col_matrix,
                           output_col,
                           output_dims[0],
                           output_dims.size() - 1,
                           &output_dims[1],
                           m_pads.data(),
                           &kernel_dims[2],
                           m_strides.data());

  }

}

template <typename TensorDataType, El::Device Device>
void
base_convolution_layer<TensorDataType,Device>
::apply_bias_cpu() {

  // Return immediately if there is no bias
  if (m_bias_scaling_factor == El::TypeTraits<ScalingType>::Zero()) return;

  // Local matrices
  const auto& local_bias = this->weights_values(1).LockedMatrix();
  auto& local_output = this->get_local_activations();

  // Matrix parameters
  const El::Int local_width = local_output.Width();
  const auto& output_dims = this->get_output_dims();
  const El::Int num_output_channels = output_dims[0];
  const El::Int num_per_output_channel = this->get_output_size() / num_output_channels;

  // Apply bias to each output channel
  LBANN_OMP_PARALLEL_FOR
  for (El::Int channel = 0; channel < num_output_channels; ++channel) {
    const El::Int row_start = channel * num_per_output_channel;
    const El::Int row_end = (channel+1) * num_per_output_channel;
    const TensorDataType bias_term = TensorDataType(m_bias_scaling_factor) * local_bias(channel, 0);
    for (El::Int col = 0; col < local_width; ++col) {
      for (El::Int row = row_start; row < row_end; ++row) {
        local_output(row, col) += bias_term;
      }
    }
  }
}

template <typename TensorDataType, El::Device Device>
void base_convolution_layer<TensorDataType,Device>
::compute_gradients_im2col(bool using_transposed_convolution) {

  // Local matrices
  const DMatDT<Device>& local_input = this->get_local_prev_activations();
  const DMatDT<Device>& local_gradient_wrt_output = this->get_local_prev_error_signals();
  const bool has_local_data = (!local_input.IsEmpty()
                               && !local_gradient_wrt_output.IsEmpty());

  // Get convolution parameters
  const El::Int local_width = local_input.Width();
  const auto& input_dims = this->get_input_dims();
  const auto& output_dims = this->get_output_dims();
  const int num_input_channels = input_dims[0];
  const int num_output_channels = output_dims[0];
  const int num_per_output_channel = this->get_output_size() / num_output_channels;
  const auto& kernel_dims = this->get_kernel_dims();
  const auto& kernel_size = std::accumulate(kernel_dims.begin(),
                                            kernel_dims.end(),
                                            1, std::multiplies<int>());

  // Compute bias gradient
  // Note: Sum is computed with Kahan summation
  if (m_bias_scaling_factor != El::TypeTraits<ScalingType>::Zero()
      && this->get_weights(1).get_optimizer() != nullptr) {
    auto* bias_optimizer = this->get_weights(1).get_optimizer();
    TensorDataType dst_scale = El::TypeTraits<TensorDataType>::Zero(), gradient_scale = El::TypeTraits<TensorDataType>::Zero();
    auto& bias_gradient = bias_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, true);
    if (has_local_data) {
      auto& local_bias_gradient = bias_gradient.Matrix();
      LBANN_OMP_PARALLEL_FOR
      for (int channel = 0; channel < num_output_channels; ++channel) {
        const El::Int row_start = channel * num_per_output_channel;
        const El::Int row_end = (channel+1) * num_per_output_channel;
        auto sum = El::TypeTraits<TensorDataType>::Zero();
        auto correction = El::TypeTraits<TensorDataType>::Zero();
        for (El::Int col = 0; col < local_width; ++col) {
          for (El::Int row = row_start; row < row_end; ++row) {
            TensorDataType term = local_gradient_wrt_output(row, col);
            term += correction;
            const TensorDataType next_sum = sum + term;
            correction = term - (next_sum - sum);
            sum = next_sum;
          }
        }
        local_bias_gradient(channel, 0) = dst_scale*local_bias_gradient(channel, 0)
          + gradient_scale*sum;
      }
    } else {
      El::Scale(dst_scale, bias_gradient);
    }
  }

  // Stop early if kernel is not being optimized
  auto* kernel_optimizer = this->get_weights(0).get_optimizer();
  if (kernel_optimizer == nullptr) { return; }

  // Initialize matrices
  const int m = (using_transposed_convolution ?
                 kernel_size / num_input_channels :
                 kernel_size / num_output_channels);
  const int n = (using_transposed_convolution ?
                 num_input_channels :
                 num_output_channels);
  const int k = (using_transposed_convolution ?
                 this->get_input_size() / num_input_channels :
                 this->get_output_size() / num_output_channels);
  auto dst_scale = El::TypeTraits<TensorDataType>::Zero(), gradient_scale = El::TypeTraits<TensorDataType>::Zero();
  auto& kernel_gradient = kernel_optimizer->get_gradient_buffer(
    dst_scale, gradient_scale, true);
  El::Scale(dst_scale, kernel_gradient);
  DMatDT<Device> im2col_matrix(m, k);
  DMatDT<Device> kernel_gradient_matrix(m, n, kernel_gradient.Buffer(), m);

  // Compute kernel gradient contributions from each data sample
  for (El::Int col = 0; col < local_width; ++col) {
    if (using_transposed_convolution) {
      const DMatDT<Device> input_col(k, n, local_input.LockedBuffer(0,col), k);
      const DMatDT<Device> gradient_wrt_output_col =
        El::LockedView(local_gradient_wrt_output, El::ALL, El::IR(col));
      im2col<TensorDataType>(gradient_wrt_output_col,
                             im2col_matrix,
                             num_output_channels,
                             output_dims.size() - 1,
                             &output_dims[1],
                             m_pads.data(),
                             &kernel_dims[2],
                             m_strides.data());
      El::Gemm(El::NORMAL, El::NORMAL,
               gradient_scale, im2col_matrix, input_col,
               El::TypeTraits<TensorDataType>::One(), kernel_gradient_matrix);
    }
    else {
      const DMatDT<Device> input_col
        = El::LockedView(local_input, El::ALL, El::IR(col));
      const DMatDT<Device> gradient_wrt_output_col(k, n, local_gradient_wrt_output.LockedBuffer(0,col), k);
      im2col<TensorDataType>(input_col,
                             im2col_matrix,
                             num_input_channels,
                             input_dims.size() - 1,
                             &input_dims[1],
                             m_pads.data(),
                             &kernel_dims[2],
                             m_strides.data());
      El::Gemm(El::NORMAL, El::NORMAL,
               gradient_scale, im2col_matrix, gradient_wrt_output_col,
               El::TypeTraits<TensorDataType>::One(), kernel_gradient_matrix);
    }
  }
}

#ifdef LBANN_HAS_CUDNN
template <typename TensorDataType, El::Device Device>
cudnnConvolutionFwdAlgo_t
base_convolution_layer<TensorDataType,Device>::get_forward_algo_cudnn(
  const int local_mini_batch_size,
  const cudnn::TensorDescriptor& input_desc,
  const TensorDataType* input,
  const cudnn::FilterDescriptor& kernel_desc,
  const TensorDataType* kernel,
  const cudnn::ConvolutionDescriptor& conv_desc,
  const cudnn::TensorDescriptor& output_desc,
  TensorDataType* output,
  size_t ws_size,
  TensorDataType* ws) {
  if (m_fwd_cudnn_algos.count(local_mini_batch_size) == 0) {
#ifdef LBANN_DETERMINISTIC
    bool deterministic = true;
#else
    bool deterministic = false;
#endif
    m_fwd_cudnn_algos[local_mini_batch_size] =
      cudnn::get_fwd_algorithm(
        true, deterministic,
        input_desc, input,
        kernel_desc, kernel,
        conv_desc,
        output_desc, output,
        ws_size, ws);
  }
  return m_fwd_cudnn_algos[local_mini_batch_size];
}

template <typename TensorDataType, El::Device Device>
cudnnConvolutionBwdDataAlgo_t
base_convolution_layer<TensorDataType,Device>::get_backward_data_algo_cudnn(
  const int local_mini_batch_size,
  const cudnn::FilterDescriptor& kernel_desc,
  const TensorDataType* kernel,
  const cudnn::TensorDescriptor& prev_error_signal_desc,
  const TensorDataType* prev_error_signal,
  const cudnn::ConvolutionDescriptor& conv_desc,
  const cudnn::TensorDescriptor& error_signal_desc,
  TensorDataType* error_signal,
  size_t ws_size,
  TensorDataType* ws) {
  if (m_bwd_data_cudnn_algos.count(local_mini_batch_size) == 0) {
#ifdef LBANN_DETERMINISTIC
    bool deterministic = true;
#else
    bool deterministic = false;
#endif
    m_bwd_data_cudnn_algos[local_mini_batch_size] =
      cudnn::get_bwd_data_algorithm(
        true, deterministic,
        kernel_desc, kernel,
        prev_error_signal_desc, prev_error_signal,
        conv_desc,
        error_signal_desc, error_signal,
        ws_size, ws);
  }
  return m_bwd_data_cudnn_algos[local_mini_batch_size];
}

template <typename TensorDataType, El::Device Device>
cudnnConvolutionBwdFilterAlgo_t
base_convolution_layer<TensorDataType,Device>::get_backward_filter_algo_cudnn(
  const int local_mini_batch_size,
  const cudnn::TensorDescriptor& input_desc,
  const TensorDataType* input,
  const cudnn::TensorDescriptor& prev_error_signal_desc,
  const TensorDataType* prev_error_signal,
  const cudnn::ConvolutionDescriptor& conv_desc,
  const cudnn::FilterDescriptor& kernel_gradient_desc,
  size_t ws_size,
  TensorDataType* ws) {
  if (m_bwd_filter_cudnn_algos.count(local_mini_batch_size) == 0) {
#ifdef LBANN_DETERMINISTIC
    bool deterministic = true;
#else
    bool deterministic = false;
#endif
    // Temporary filter gradient buffer.
    El::Matrix<TensorDataType, El::Device::GPU> kernel_gradient;
#ifdef HYDROGEN_HAVE_CUB
    kernel_gradient.SetMemoryMode(1);
#endif
    kernel_gradient.Resize(this->get_weights(0).get_matrix_height(),
                           this->get_weights(0).get_matrix_width());
    m_bwd_filter_cudnn_algos[local_mini_batch_size] =
      cudnn::get_bwd_filter_algorithm(
        true, deterministic,
        input_desc, input,
        prev_error_signal_desc, prev_error_signal,
        conv_desc,
        kernel_gradient_desc, kernel_gradient.Buffer(),
        ws_size, ws);
  }
  return m_bwd_filter_cudnn_algos[local_mini_batch_size];
}
#endif // LBANN_HAS_CUDNN

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, El::Device Device>
void base_convolution_layer<TensorDataType,Device>::setup_distconv_adapter(
    const DataReaderMetaData& dr_metadata) {
  this->get_distconv_adapter_ptr() = make_unique<
    base_convolution_adapter<TensorDataType, Device>>(*this);
}

template <typename TensorDataType, El::Device Device>
const base_convolution_adapter<TensorDataType, Device>&
base_convolution_layer<TensorDataType, Device>::get_distconv_adapter() const {
  return dynamic_cast<const base_convolution_adapter<TensorDataType, Device>&>(
    data_type_layer<TensorDataType>::get_distconv_adapter());
}

template <typename TensorDataType, El::Device Device>
base_convolution_adapter<TensorDataType, Device>&
base_convolution_layer<TensorDataType, Device>::get_distconv_adapter() {
  return const_cast<base_convolution_adapter<TensorDataType, Device>&>(
    static_cast<const base_convolution_layer<TensorDataType, Device>&>(*this).get_distconv_adapter());
}

template <typename TensorDataType, El::Device Device>
void base_convolution_adapter<TensorDataType, Device>::setup_fp_tensors() {
  data_type_distconv_adapter<TensorDataType>::setup_fp_tensors();
  auto &layer = dynamic_cast<
    base_convolution_layer<TensorDataType, Device>&>(this->layer());
  const auto &input_dist = this->get_prev_activations_dist();

  const auto& kernel_dims = layer.get_kernel_dims();
  std::stringstream ss;
  dc::util::print_vector(ss, kernel_dims.begin(), kernel_dims.end());

  // assumes no partitioning on channel/filter dimensions
  assert_eq(input_dist.get_split_shape()[-2], 1);
  auto shared_dist = dc::Dist::make_shared_distribution(
    input_dist.get_locale_shape());

  dc::Shape kernel_shape(kernel_dims);
  std::reverse(kernel_shape.begin(), kernel_shape.end());
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
  m_kernel = make_unique<TensorDevType>(kernel_shape, loc, shared_dist);

  if (layer.m_bias_scaling_factor != TensorDataType(0)) {
    dc::Shape bias_shape(dc::get_num_dims(layer), 1);
    bias_shape[dc::get_channel_dim()] = layer.get_output_dims()[0];
    m_bias = make_unique<TensorDevType>(bias_shape, loc, shared_dist);
  }
}

template <typename TensorDataType, El::Device Device>
void base_convolution_adapter<TensorDataType, Device>::setup_bp_tensors() {
  data_type_distconv_adapter<TensorDataType>::setup_bp_tensors();
  auto &l = dynamic_cast<
    base_convolution_layer<TensorDataType, Device>&>(this->layer());

  const auto shared_dist = dc::Dist::make_shared_distribution(
    this->get_prev_error_signals_dist().get_locale_shape());
  dc::Shape kernel_shape(l.get_kernel_dims());
  std::reverse(kernel_shape.begin(), kernel_shape.end());
  const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

  m_kernel_gradient = make_unique<TensorDevType>(kernel_shape, loc, shared_dist);
  // Gradient buffer is needed for auto-tuning the bp filter algorithm
  auto* kernel_optimizer = static_cast<data_type_optimizer<TensorDataType>*>(l.get_weights(0).get_optimizer());
  assert0(dc::tensor::View(
            *m_kernel_gradient,
            kernel_optimizer->get_gradient().Buffer()));

  // Bias tensor. Shared by all procs
  if (l.m_bias_scaling_factor != TensorDataType(0)) {
    auto* bias_optimizer = static_cast<data_type_optimizer<TensorDataType>*>(l.get_weights(1).get_optimizer());
    if (bias_optimizer != nullptr) {
      dc::Shape bias_shape(dc::get_num_dims(l), 1);
      bias_shape[dc::get_channel_dim()] = l.get_output_dims()[0];
      m_bias_gradient = make_unique<TensorDevType>(bias_shape, loc, shared_dist);
      // setup_bias_gradients needs strides of the bias tensor,
      // which is set when its view is set.
      assert0(dc::tensor::View(
                *m_bias_gradient,
                bias_optimizer->get_gradient().Buffer()));
    }
  }
}

template <typename TensorDataType, El::Device Device>
void base_convolution_adapter<TensorDataType, Device>::setup_layer(
  size_t workspace_capacity) {
  data_type_distconv_adapter<TensorDataType>::setup_layer(workspace_capacity);
  auto &layer = dynamic_cast<base_convolution_layer<TensorDataType, Device>&>(this->layer());
  m_conv = make_unique<dc::Convolution<TensorDataType>>(
    dc::get_backend(), dc::get_num_dims(layer),
    dc::get_halo_exchange_method());
  if (layer.m_bias_scaling_factor != TensorDataType(0)) {
    m_conv->setup_bias(*m_bias);
    m_conv->setup_bias_gradient(*m_bias_gradient);
  }
}

template <typename TensorDataType, El::Device Device>
void base_convolution_adapter<TensorDataType, Device>::fp_compute_convolution() {
  auto &l = dynamic_cast<base_convolution_layer<
    TensorDataType, Device>&>(this->layer());
  assert0(dc::tensor::View(
            *m_kernel, l.weights_values(0).LockedBuffer()));
  m_conv->forward(TensorDataType{1}, this->get_prev_activations(),
                  *m_kernel, TensorDataType{0}, this->get_activations());
}

template <typename TensorDataType, El::Device Device>
void base_convolution_adapter<TensorDataType, Device>::fp_apply_bias() {
  auto &l = dynamic_cast<base_convolution_layer<
    TensorDataType, Device>&>(this->layer());
  if (l.m_bias_scaling_factor == TensorDataType(0)) return;
  assert0(dc::tensor::View(
            *m_bias, l.weights_values(1).LockedBuffer()));
  m_conv->apply_bias(l.m_bias_scaling_factor, *m_bias,
                     TensorDataType{1}, this->get_activations());
}

template <typename TensorDataType, El::Device Device>
void base_convolution_adapter<TensorDataType, Device>::bp_compute_convolution_data() {
  auto &l = dynamic_cast<base_convolution_layer<
    TensorDataType, Device>&>(this->layer());
  assert0(dc::tensor::View(
            *m_kernel, l.weights_values(0).LockedBuffer()));
  m_conv->backward_data(TensorDataType{1}, *m_kernel,
                        this->get_prev_error_signals(),
                        TensorDataType{0}, this->get_error_signals());
}

template <typename TensorDataType, El::Device Device>
void base_convolution_adapter<TensorDataType, Device>::bp_compute_convolution_filter() {
  auto &l = dynamic_cast<base_convolution_layer<
    TensorDataType, Device>&>(this->layer());
  const bool has_local_data = this->get_prev_activations().get_local_size() > 0 &&
    this->get_prev_error_signals().get_local_size() > 0;
  if (l.m_bias_scaling_factor != TensorDataType(0)
      && l.get_weights(1).get_optimizer() != nullptr) {
    auto* bias_optimizer = l.get_weights(1).get_optimizer();
    TensorDataType dst_scale{0}, gradient_scale{0};
    auto& bias_gradient = bias_optimizer->get_gradient_buffer(
      dst_scale, gradient_scale, true);
    assert0(dc::tensor::View(*m_bias_gradient,
                             bias_gradient.Buffer()));
    if (has_local_data) {
      m_conv->backward_bias(gradient_scale,
                            this->get_prev_error_signals(),
                            dst_scale, *m_bias_gradient, false);
    } else {
      m_bias_gradient->scale(dst_scale, hydrogen::cuda::GetDefaultStream());
    }
  }

  auto* kernel_optimizer = l.get_weights(0).get_optimizer();
  if (kernel_optimizer == nullptr) return;
  TensorDataType dst_scale{0}, gradient_scale{0};
  auto& kernel_gradient = kernel_optimizer->get_gradient_buffer(
    dst_scale, gradient_scale, true);
  assert0(dc::tensor::View(
            *m_kernel_gradient, kernel_gradient.Buffer()));
  if (has_local_data) {
    m_conv->backward_filter(gradient_scale,
                            this->get_prev_activations(),
                            this->get_prev_error_signals(),
                            dst_scale,
                            *m_kernel_gradient, false);
  } else {
    m_kernel_gradient->scale(dst_scale, hydrogen::cuda::GetDefaultStream());
  }
}


#define PROTO_DEVICE(T, Device)                                            \
  template class base_convolution_adapter<T, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_HAS_DISTCONV

#define PROTO_DEVICE(T, Device)                                            \
  template class base_convolution_layer<T, Device>

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
