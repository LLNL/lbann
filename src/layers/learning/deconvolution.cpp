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

#define LBANN_CONVOLUTION_LAYER_INSTANTIATE
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/layers/learning/deconvolution.hpp"
#include "lbann/utils/exception.hpp"

#include <sstream>
#include <string>

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
deconvolution_layer<TensorDataType,Layout,Device>::deconvolution_layer(
  int num_data_dims,
  int num_output_channels,
  int conv_dim,
  int pad,
  int stride,
  int dilation,
  int groups,
  bool has_bias)
  : deconvolution_layer(num_data_dims,
                        num_output_channels,
                        std::vector<int>(num_data_dims, conv_dim),
                        std::vector<int>(num_data_dims, pad),
                        std::vector<int>(num_data_dims, stride),
                        std::vector<int>(num_data_dims, dilation),
                        groups,
                        has_bias)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
deconvolution_layer<TensorDataType,Layout,Device>::deconvolution_layer(
  int num_data_dims,
  int num_output_channels,
  std::vector<int> conv_dims,
  std::vector<int> pads,
  std::vector<int> strides,
  std::vector<int> dilations,
  int groups,
  bool has_bias)
  : base_convolution_layer<TensorDataType, Device>(
    num_data_dims,
    num_output_channels,
    std::move(conv_dims),
    std::move(pads),
    std::move(strides),
    std::move(dilations),
    groups,
    has_bias)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
deconvolution_layer<TensorDataType,Layout,Device>::deconvolution_layer()
  : deconvolution_layer(0, 0, {}, {}, {}, {}, 0, false)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void
deconvolution_layer<TensorDataType,Layout,Device>
::setup_dims(DataReaderMetaData& dr_metadata) {
  base_convolution_layer<TensorDataType, Device>::setup_dims(dr_metadata);
  std::ostringstream err;

  // Get tensor dimensions
  const auto& input_dims = this->get_input_dims();
  auto output_dims = input_dims;

  // Check for unsupported features
  /// @todo Implement dilated and grouped deconvolution
  if (std::any_of(this->m_dilations.begin(),
                  this->m_dilations.end(),
                  [] (int d) { return d != 1; })) {
    err << this->get_type() << " layer "
        << "\"" << this->get_name() << "\" "
        << "has non-unit dilations (";
    for (size_t i = 0; i < this->m_dilations.size(); ++i) {
      err << (i > 0 ? ", " : "") << this->m_dilations[i];
    }
    err << ")";
    LBANN_ERROR(err.str());
  }
  if (this->m_groups != 1) {
    err << this->get_type() << " layer "
        << "\"" << this->get_name() << "\" "
        << "has non-unit groups "
        << "(" << this->m_groups << ")";
    LBANN_ERROR(err.str());
  }

  // Initialize output tensor dimensions
  /// @todo Dilated deconvolution
  output_dims[0] = this->m_output_channels;
  for (size_t i = 0; i < output_dims.size() - 1; ++i) {
    const auto& input_dim = input_dims[i+1];
    const auto& kernel_dim = this->m_conv_dims[i];
    const auto& stride = this->m_strides[i];
    const auto& pad = this->m_pads[i];
    // const auto& dilation = this->m_dilations[i];
    output_dims[i+1] = (input_dim-1) * stride + kernel_dim - 2 * pad;
  }
  this->set_output_dims(output_dims);

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::vector<int>
deconvolution_layer<TensorDataType,Layout,Device>
::get_kernel_dims() const {
  std::vector<int> dims;
  dims.push_back(this->get_input_dims()[0]);
  dims.push_back(this->m_output_channels);
  dims.insert(dims.end(),
              this->m_conv_dims.begin(),
              this->m_conv_dims.end());
  return dims;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void deconvolution_layer<TensorDataType,Layout,Device>::fp_compute() {
  using BaseConvLayer = base_convolution_layer<TensorDataType, Device>;
  if(this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      this->get_distconv_adapter().fp_compute_convolution();
      this->get_distconv_adapter().fp_apply_bias();
      return;
    }
#endif // LBANN_HAS_DISTCONV
    BaseConvLayer::apply_transposed_convolution_dnn(true);
    BaseConvLayer::apply_bias_dnn();
  } else {
    BaseConvLayer::apply_transposed_convolution_im2col(true);
    BaseConvLayer::apply_bias_cpu();
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void deconvolution_layer<TensorDataType,Layout,Device>::bp_compute() {
  using BaseConvLayer = base_convolution_layer<TensorDataType, Device>;
  if(this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
    if (this->distconv_enabled()) {
      if (this->get_distconv_adapter().m_conv->is_overlap_bwd_halo_exchange_enabled()) {
        this->get_distconv_adapter().m_conv->backward_data_exchange_halo(
          this->get_distconv_adapter().get_prev_error_signals());
      }
      this->get_distconv_adapter().bp_compute_convolution_filter();
      this->get_distconv_adapter().bp_compute_convolution_data();
      return;
    }
#endif // LBANN_HAS_DISTCONV
    BaseConvLayer::compute_gradients_dnn(true);
    BaseConvLayer::apply_convolution_dnn(false);
  } else {
    BaseConvLayer::compute_gradients_im2col(true);
    BaseConvLayer::apply_convolution_im2col(false);
  }
}

#if defined LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
void deconvolution_layer<TensorDataType,Layout,Device>
::setup_distconv_adapter(const DataReaderMetaData& dr_metadata) {
  this->get_distconv_adapter_ptr() = make_unique<
    deconvolution_distconv_adapter<TensorDataType, Layout, Device>>(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
bool deconvolution_layer<TensorDataType,Layout,Device>
::is_distconv_supported() const {
  const auto& kernel_dims = get_kernel_dims();
  for(int i = 0; i < dc::get_num_spatial_dims(*this); i++) {
    auto pad = this->m_pads[i];
    if (pad != 0) {
      dc::MPIPrintStreamDebug()
        << this->get_name()
        << " unsupported as padding must be zero";
      return false;
    }
    auto stride_size = this->m_strides[i];
    auto filter_size = kernel_dims[2+i];
    if (!(filter_size % 2 == 0 && filter_size == stride_size)) {
      dc::MPIPrintStreamDebug()
        << this->get_name()
        << " unsupported due to filter and stride sizes";
      return false;
    }
  }
  return true;
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void deconvolution_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_distributions(tensor_overlap_constraints &constraints) {
  base_convolution_adapter<TensorDataType, Dev>::setup_distributions(
      constraints);

  // Assumes zero halo all tensor for now
  // prev activations
  for (auto &d: this->m_prev_activations_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
  for (auto &d: this->m_activations_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
  for (auto &d: this->m_prev_error_signals_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
  for (auto &d: this->m_error_signals_dists) {
    d.clear_overlap();
    constraints.mark_updated(d);
    constraints.mark_invariant(d);
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dc::Shape deconvolution_distconv_adapter<TensorDataType, Layout, Device>::
get_activations_local_shape(int index) const {
  assert_eq(index, 0);
  const auto &layer = dynamic_cast<const deconvolution_layer<
    TensorDataType, Layout, Device>&>(this->layer());
  auto filter_dims = layer.get_kernel_dims();
  std::reverse(std::begin(filter_dims), std::end(filter_dims));
  auto strides = layer.m_strides;
  std::reverse(std::begin(strides), std::end(strides));
  auto dilations = layer.m_dilations;
  std::reverse(std::begin(dilations), std::end(dilations));
  const auto output_spatial_local_shape =
      ::distconv::get_deconvolution_output_local_tensor_shape(
          this->get_prev_activations(),
          filter_dims, strides, false, dilations,
          layer.m_groups);
  return output_spatial_local_shape;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void deconvolution_distconv_adapter<TensorDataType, Layout, Device>::setup_layer(
    size_t workspace_capacity) {
  base_convolution_adapter<TensorDataType, Device>::setup_layer(
      workspace_capacity);
  auto &layer = dynamic_cast<deconvolution_layer<
    TensorDataType, Layout, Device>&>(this->layer());

  if (dc::is_deterministic()) {
    dc::MPIRootPrintStreamDebug()
      << "Using deterministic convolution algorithms";
    this->m_fwd_algo = "DETERMINISTIC";
    this->m_bwd_data_algo = "DETERMINISTIC";
    this->m_bwd_filter_algo = "DETERMINISTIC";
  } else {
    this->m_fwd_algo = dc::get_convolution_bwd_data_algorithm();
    this->m_bwd_data_algo = dc::get_convolution_fwd_algorithm();
    this->m_bwd_filter_algo = dc::get_convolution_bwd_filter_algorithm();
  }

  std::vector<int> pads = layer.m_pads;
  std::reverse(pads.begin(), pads.end());
  std::vector<int> strides = layer.m_strides;
  std::reverse(strides.begin(), strides.end());
  std::vector<int> dilations = layer.m_dilations;
  std::reverse(dilations.begin(), dilations.end());

  this->m_conv->setup(this->get_prev_activations(),
                      *(this->m_kernel), this->get_activations(),
                      this->get_error_signals(),
                      *this->m_kernel_gradient,
                      this->get_prev_error_signals(),
                      pads, strides, dilations, layer.m_groups,
                      this->m_fwd_algo, this->m_bwd_data_algo,
                      this->m_bwd_filter_algo,
                      workspace_capacity, false, true);
}
#endif // defined LBANN_HAS_DISTCONV

#define PROTO_DEVICE(T, Device)                                             \
  template class deconvolution_layer<T, data_layout::DATA_PARALLEL, Device>;

#include "lbann/macros/instantiate_device.hpp"

}// namespace lbann

#define LBANN_LAYER_NAME deconvolution_layer
#include <lbann/macros/register_layer_with_cereal_data_parallel_only.hpp>
