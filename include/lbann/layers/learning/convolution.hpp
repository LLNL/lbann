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

  void setup_dims() override {
    base_convolution_layer<TensorDataType, Device>::setup_dims();

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
        this->distconv_forward();
        this->apply_bias_distconv();
        this->copy_out_activations();
        if (this->early_terminate_last_iteration() &&
            this->keep_original()) {
          base_convolution_layer<TensorDataType, Device>::apply_convolution_cudnn(true);
          base_convolution_layer<TensorDataType, Device>::apply_bias_cudnn();
          this->dump_reference_activations();
        }
      }
#else
      base_convolution_layer<TensorDataType, Device>::apply_convolution_cudnn(true);
      base_convolution_layer<TensorDataType, Device>::apply_bias_cudnn();
#endif // LBANN_HAS_DISTCONV
    } else {
      base_convolution_layer<TensorDataType, Device>::apply_convolution_im2col(true);
      base_convolution_layer<TensorDataType, Device>::apply_bias_cpu();
    }
  }

  void bp_compute() override {
    if(this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
      if (this->distconv_enabled()) {
        // Only weight gradients need to be computed
        if (this->skip_first_layer_bp()) {
          dc::MPIRootPrintStreamDebug() << "Skipping bp data for "
                                        << this->get_name();
          this->distconv_backward_filter();
          return;
        }
        if (this->m_conv->is_overlap_bwd_halo_exchange_enabled()) {
          this->m_conv->backward_data_exchange_halo(this->get_prev_error_signals_t());
        }
        this->distconv_backward_filter();
        this->distconv_backward_data();
        if (this->early_terminate_last_iteration() &&
            this->keep_original()) {
          base_convolution_layer<TensorDataType, Device>::compute_gradients_cudnn(false);
          base_convolution_layer<TensorDataType, Device>::apply_transposed_convolution_cudnn(false);
          this->dump_reference_error_signals();
        }
      }
#else
      base_convolution_layer<TensorDataType, Device>::compute_gradients_cudnn(false);
      base_convolution_layer<TensorDataType, Device>::apply_transposed_convolution_cudnn(false);
#endif // LBANN_HAS_DISTCONV
    } else {
      base_convolution_layer<TensorDataType, Device>::compute_gradients_im2col(false);
      base_convolution_layer<TensorDataType, Device>::apply_transposed_convolution_im2col(false);
    }
  }

#ifdef LBANN_HAS_DISTCONV
 public:
  void init_distribution(
      std::map<const Layer*, std::array<dc::Dist, dc::num_dists>> &dists,
      std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
      std::set<dc::Dist*> &updated,
      std::set<dc::Dist*> &fixed) override {
    base_convolution_layer<TensorDataType, Device>::init_distribution(
        dists, invariants, updated, fixed);
    if (!this->distconv_enabled()) return;

    auto kernel_dims = get_kernel_dims();
    std::reverse(kernel_dims.begin(), kernel_dims.end());
    auto dilations = this->m_dilations;
    std::reverse(dilations.begin(), dilations.end());
    dc::IntVector overlap(this->get_num_dims(), 0);
    const auto &ps = this->get_parallel_strategy();
    // i=0 -> width; i=1 -> height; i=2: -> depth;
    for(int i = 0; i < this->get_num_spatial_dims(); i++) {
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
    auto &prev_activations_dist = dists[this][0];
    prev_activations_dist.set_overlap(overlap);
    updated.insert(&prev_activations_dist);
    fixed.insert(&prev_activations_dist);
    auto &prev_error_signals_dist = dists[this][3];
    prev_error_signals_dist.set_overlap(overlap);
    updated.insert(&prev_error_signals_dist);
    fixed.insert(&prev_error_signals_dist);
    // To deal with strides, error signals must have the same size
    // of overlap
    auto &error_signals_dist = dists[this][2];
    error_signals_dist.set_overlap(overlap);
    updated.insert(&error_signals_dist);
    fixed.insert(&error_signals_dist);
  }

  dc::Shape get_activations_tensor_local_shape() const override {
    std::vector<int> filter_dims = get_kernel_dims();
    std::reverse(filter_dims.begin(), filter_dims.end());
    std::vector<int> strides = this->m_strides;
    std::reverse(strides.begin(), strides.end());
    std::vector<int> dilations = this->m_dilations;
    std::reverse(dilations.begin(), dilations.end());
    const auto output_spatial_local_shape =
        ::distconv::get_convolution_output_local_tensor_shape(
            this->get_prev_activations_t(),
            filter_dims, strides, true, dilations,
            this->m_groups);
    return output_spatial_local_shape;
  }

  void setup_distconv_post(size_t ws_size) override {
    base_convolution_layer<TensorDataType, Device>::setup_distconv_post(ws_size);
    if (!this->distconv_enabled()) return;

    if (dc::is_deterministic()) {
      dc::MPIRootPrintStreamDebug() << "Using deterministic convolution algorithms";
      // Same algorithm as LBANN
      this->m_fwd_algo = "IMPLICIT_GEMM";
      // Deterministic algorithm
      this->m_bwd_data_algo = "ALGO_1";
      this->m_bwd_filter_algo = "ALGO_1";
    } else {
      this->m_fwd_algo = dc::get_convolution_fwd_algorithm();
      this->m_bwd_data_algo = dc::get_convolution_bwd_data_algorithm();
      this->m_bwd_filter_algo = dc::get_convolution_bwd_filter_algorithm();
    }

    std::vector<int> pads = this->m_pads;
    std::reverse(pads.begin(), pads.end());
    std::vector<int> strides = this->m_strides;
    std::reverse(strides.begin(), strides.end());
    std::vector<int> dilations = this->m_dilations;
    std::reverse(dilations.begin(), dilations.end());

    this->m_conv->setup(this->get_prev_activations_t(),
                        this->m_kernel_t, this->get_activations_t(),
                        this->get_error_signals_t(), this->m_kernel_gradient_e,
                        this->get_prev_error_signals_t(),
                        pads, strides, dilations, this->m_groups,
                        this->m_fwd_algo, this->m_bwd_data_algo,
                        this->m_bwd_filter_algo,
                        ws_size, this->skip_first_layer_bp());
  }

 protected:
  bool using_distconv() const override {
    if (!base_convolution_layer<TensorDataType, Device>::using_distconv()) return false;

    bool cond = true;
    const auto& kernel_dims = get_kernel_dims();
    for(int i = 0; i < this->get_num_spatial_dims(); i++) {
      cond &= kernel_dims[2 + i] == kernel_dims[2];
      cond &= kernel_dims[2 + i] == this->m_pads[i] / this->m_dilations[i] * 2 + 1;
    }
    if (!cond) {
      dc::MPIPrintStreamDebug()
          << "Unsupported as padding does not match the kernel size";
      return false;
    }
    return true;
  }

#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_CONVOLUTION_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device) \
  extern template class convolution_layer<T, data_layout::DATA_PARALLEL, Device>;

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF

#endif // LBANN_CONVOLUTION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_CONVOLUTION_HPP_INCLUDED
