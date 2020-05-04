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

#ifndef LBANN_LAYERS_LEARNING_CONVOLUTION_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_CONVOLUTION_HPP_INCLUDED

#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

// Forward declaration.
namespace callback {
class imcomm;
}

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout Layout, El::Device Device>
class convolution_distconv_adapter: public base_convolution_adapter<TensorDataType, Device> {
 public:
  using TensorDevType = typename base_convolution_adapter<TensorDataType, Device>::TensorDevType;

  convolution_distconv_adapter(Layer& layer): base_convolution_adapter<TensorDataType, Device>(layer) {}
  virtual ~convolution_distconv_adapter() = default;

  void setup_distributions(tensor_overlap_constraints &constraints) override;
  void setup_layer(size_t workspace_capacity) override;
  dc::Shape get_activations_local_shape(int index=0) const override;
};
#endif // LBANN_HAS_DISTCONV

/** @brief Standard deep learning convolution.
 *
 *  Applies convolution (more precisely, cross-correlation) to input
 *  tensors. This is primarily optimized for image data in NCHW
 *  format.
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class convolution_layer : public base_convolution_layer<TensorDataType, Device> {
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "convolution layer only supports DATA_PARALLEL");
private:

  friend class callback::imcomm;

public:

  convolution_layer(lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    int conv_dim,
                    int pad,
                    int stride,
                    int dilation,
                    int groups,
                    bool has_bias = true)
    : convolution_layer(comm,
                        num_data_dims,
                        num_output_channels,
                        std::vector<int>(num_data_dims, conv_dim),
                        std::vector<int>(num_data_dims, pad),
                        std::vector<int>(num_data_dims, stride),
                        std::vector<int>(num_data_dims, dilation),
                        groups,
                        has_bias) {}

  convolution_layer(lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    std::vector<int> conv_dims,
                    std::vector<int> pads,
                    std::vector<int> strides,
                    std::vector<int> dilations,
                    int groups,
                    bool has_bias = true)
    : base_convolution_layer<TensorDataType, Device>(
        comm,
        num_data_dims,
        num_output_channels,
        std::move(conv_dims),
        std::move(pads),
        std::move(strides),
        std::move(dilations),
        groups,
        has_bias) {
  }

  convolution_layer* copy() const override { return new convolution_layer(*this); }

  std::string get_type() const override { return "convolution"; }

  data_layout get_data_layout() const override { return Layout; }

  El::Device get_device_allocation() const override { return Device; }

protected:

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    base_convolution_layer<TensorDataType, Device>::setup_dims(dr_metadata);

    // Get tensor dimensions
    const auto& input_dims = this->get_input_dims();
    auto output_dims = input_dims;

    // Initialize output tensor dimensions
    output_dims[0] = this->m_output_channels;
    for (size_t i = 0; i < output_dims.size() - 1; ++i) {
      const auto& input_dim = input_dims[i+1];
      const auto& kernel_dim = this->m_conv_dims[i];
      const auto& stride = this->m_strides[i];
      const auto& pad = this->m_pads[i];
      const auto& dilation = this->m_dilations[i];
      const auto& effective_dim = (input_dim
                                   + 2 * pad
                                   - dilation * (kernel_dim-1));
      output_dims[i+1] = (effective_dim + stride - 1) / stride;
    }
    this->set_output_dims(output_dims);

  }

  std::vector<int> get_kernel_dims() const override {
    std::vector<int> dims;
    dims.push_back(this->m_output_channels);
    dims.push_back(this->get_input_dims()[0] / this->m_groups);
    dims.insert(dims.end(),
                this->m_conv_dims.begin(),
                this->m_conv_dims.end());
    return dims;
  }

  void fp_compute() override {
    if(this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
      if (this->distconv_enabled()) {
        this->get_distconv_adapter().fp_compute_convolution();
        this->get_distconv_adapter().fp_apply_bias();
        return;
      }
#endif // LBANN_HAS_DISTCONV
      base_convolution_layer<TensorDataType, Device>::apply_convolution_cudnn(true);
      base_convolution_layer<TensorDataType, Device>::apply_bias_cudnn();
    } else {
      base_convolution_layer<TensorDataType, Device>::apply_convolution_im2col(true);
      base_convolution_layer<TensorDataType, Device>::apply_bias_cpu();
    }
  }

  void bp_compute() override {
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
      base_convolution_layer<TensorDataType, Device>::compute_gradients_cudnn(false);
      base_convolution_layer<TensorDataType, Device>::apply_transposed_convolution_cudnn(false);
    } else {
      base_convolution_layer<TensorDataType, Device>::compute_gradients_im2col(false);
      base_convolution_layer<TensorDataType, Device>::apply_transposed_convolution_im2col(false);
    }
  }

#ifdef LBANN_HAS_DISTCONV
  friend class convolution_distconv_adapter<TensorDataType, Layout, Device>;
 protected:
  void setup_distconv_adapter() override {
    this->get_distconv_adapter_ptr() = make_unique<
      convolution_distconv_adapter<TensorDataType, Layout, Device>>(*this);
  }

  bool is_distconv_supported() const override {
    const auto& kernel_dims = get_kernel_dims();
    for(int i = 0; i < dc::get_num_spatial_dims(*this); i++) {
      if (kernel_dims[2 + i] != kernel_dims[2]) {
        dc::MPIRootPrintStreamDebug()
            << "Nonsymmetric kernel not supported";
        return false;
      }
      if (kernel_dims[2 + i] !=
          this->m_pads[i] / this->m_dilations[i] * 2 + 1) {
        dc::MPIRootPrintStreamDebug()
            << "Unsupported as padding does not match the kernel size";
        return false;
      }
    }
    return true;
  }
#endif // LBANN_HAS_DISTCONV
};

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void convolution_distconv_adapter<TensorDataType, T_layout, Dev>::
setup_distributions(tensor_overlap_constraints &constraints) {
  base_convolution_adapter<TensorDataType, Dev>::setup_distributions(
      constraints);
  auto &l = dynamic_cast<convolution_layer<
    TensorDataType, T_layout, Dev>&>(this->layer());
  auto kernel_dims = l.get_kernel_dims();
  std::reverse(kernel_dims.begin(), kernel_dims.end());
  auto dilations = l.m_dilations;
  std::reverse(dilations.begin(), dilations.end());
  dc::IntVector overlap(dc::get_num_dims(l), 0);
  const auto &ps = l.get_parallel_strategy();
  // i=0 -> width; i=1 -> height; i=2: -> depth;
  for(int i = 0; i < dc::get_num_spatial_dims(l); i++) {
    int splits = 0;
    switch (i) {
      case 0: splits = ps.width_splits; break;
      case 1: splits = ps.height_splits; break;
      case 2: splits = ps.depth_splits; break;
    }
    if (splits > 1) {
      overlap[i] = (kernel_dims[i] - 1) / 2 * dilations[i];
    }
  }
  auto &prev_activations_dist = this->get_prev_activations_dist();
  prev_activations_dist.set_overlap(overlap);
  constraints.mark_updated(prev_activations_dist);
  constraints.mark_invariant(prev_activations_dist);
  auto &prev_error_signals_dist = this->get_prev_error_signals_dist();
  prev_error_signals_dist.set_overlap(overlap);
  constraints.mark_updated(prev_error_signals_dist);
  constraints.mark_invariant(prev_error_signals_dist);
  // To deal with strides, error signals must have the same size
  // of overlap
  auto &error_signals_dist = this->get_error_signals_dist();
  error_signals_dist.set_overlap(overlap);
  constraints.mark_updated(error_signals_dist);
  constraints.mark_invariant(error_signals_dist);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
dc::Shape convolution_distconv_adapter<TensorDataType, Layout, Device>::
get_activations_local_shape(int index) const {
  assert_eq(index, 0);
  const auto &layer = dynamic_cast<const convolution_layer<
    TensorDataType, Layout, Device>&>(this->layer());
  auto filter_dims = layer.get_kernel_dims();
  std::reverse(std::begin(filter_dims), std::end(filter_dims));
  auto strides = layer.m_strides;
  std::reverse(std::begin(strides), std::end(strides));
  auto dilations = layer.m_dilations;
  std::reverse(std::begin(dilations), std::end(dilations));
  const auto output_spatial_local_shape =
      ::distconv::get_convolution_output_local_tensor_shape(
          this->get_prev_activations(),
          filter_dims, strides, true, dilations,
          layer.m_groups);
  return output_spatial_local_shape;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void convolution_distconv_adapter<TensorDataType, Layout, Device>::setup_layer(
    size_t workspace_capacity) {
  base_convolution_adapter<TensorDataType, Device>::setup_layer(
      workspace_capacity);
  auto &layer = dynamic_cast<convolution_layer<
    TensorDataType, Layout, Device>&>(this->layer());

  if (dc::is_deterministic()) {
    dc::MPIRootPrintStreamDebug() << "Using deterministic convolution algorithms";
    this->m_fwd_algo = "DETERMINISTIC";
    this->m_bwd_data_algo = "DETERMINISTIC";
    this->m_bwd_filter_algo = "DETERMINISTIC";
  } else {
    this->m_fwd_algo = dc::get_convolution_fwd_algorithm();
    this->m_bwd_data_algo = dc::get_convolution_bwd_data_algorithm();
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
                      workspace_capacity);
}
#endif // LBANN_HAS_DISTCONV

// Builder function
LBANN_DEFINE_LAYER_BUILDER(convolution);

#ifndef LBANN_CONVOLUTION_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device) \
  extern template class convolution_layer<T, data_layout::DATA_PARALLEL, Device>;

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_CONVOLUTION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_CONVOLUTION_HPP_INCLUDED
