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

#ifndef LBANN_LAYERS_LEARNING_DECONVOLUTION_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_DECONVOLUTION_HPP_INCLUDED

#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

// Forward declaration.
namespace callback {
class imcomm;
}

/** @brief Transpose of the convolution layer. */
template <data_layout Layout = data_layout::DATA_PARALLEL, El::Device Device = El::Device::CPU>
class deconvolution_layer : public base_convolution_layer<Device> {
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "deconvolution layer only supports DATA_PARALLEL");
private:

  friend class callback::imcomm;

public:

  deconvolution_layer(lbann_comm *comm,
                      int num_data_dims,
                      int num_output_channels,
                      int conv_dim,
                      int pad,
                      int stride,
                      int dilation,
                      int groups,
                      bool has_bias = true)
    : deconvolution_layer(comm,
                          num_data_dims,
                          num_output_channels,
                          std::vector<int>(num_data_dims, conv_dim),
                          std::vector<int>(num_data_dims, pad),
                          std::vector<int>(num_data_dims, stride),
                          std::vector<int>(num_data_dims, dilation),
                          groups,
                          has_bias) {}

  deconvolution_layer(lbann_comm *comm,
                      int num_data_dims,
                      int num_output_channels,
                      std::vector<int> conv_dims,
                      std::vector<int> pads,
                      std::vector<int> strides,
                      std::vector<int> dilations,
                      int groups,
                      bool has_bias = true)
    : base_convolution_layer<Device>(
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

  deconvolution_layer* copy() const override { return new deconvolution_layer(*this); }

  std::string get_type() const override { return "deconvolution"; }

  data_layout get_data_layout() const override { return Layout; }

  El::Device get_device_allocation() const override { return Device; }

  void setup_dims() override {
    base_convolution_layer<Device>::setup_dims();
    std::stringstream err;

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

protected:

  std::vector<int> get_kernel_dims() const override {
    std::vector<int> dims;
    dims.push_back(this->get_input_dims()[0]);
    dims.push_back(this->m_output_channels);
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
          base_convolution_layer<Device>::apply_transposed_convolution_cudnn(true);
          base_convolution_layer<Device>::apply_bias_cudnn();
          this->dump_reference_activations();
        }
        return;
      }
#endif
      base_convolution_layer<Device>::apply_transposed_convolution_cudnn(true);
      base_convolution_layer<Device>::apply_bias_cudnn();
    } else {
      base_convolution_layer<Device>::apply_transposed_convolution_im2col(true);
      base_convolution_layer<Device>::apply_bias_cpu();
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
          this->m_conv->backward_data_exchange_halo(this->m_prev_error_signals_t);
        }
        this->distconv_backward_filter();
        this->distconv_backward_data();
        if (this->early_terminate_last_iteration() &&
            this->keep_original()) {
          base_convolution_layer<Device>::compute_gradients_cudnn(true);
          base_convolution_layer<Device>::apply_convolution_cudnn(false);
          this->dump_reference_error_signals();
        }
        return;
      }
#endif
      base_convolution_layer<Device>::compute_gradients_cudnn(true);
      base_convolution_layer<Device>::apply_convolution_cudnn(false);
    } else {
      base_convolution_layer<Device>::compute_gradients_im2col(true);
      base_convolution_layer<Device>::apply_convolution_im2col(false);
    }
  }

#ifdef LBANN_HAS_DISTCONV
 public:
  void setup_tensor_distribution_init(
      std::map<const Layer*, std::array<dc::Dist, dc::num_dists>> &dists,
      std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
      std::set<dc::Dist*> &updated,
      std::set<dc::Dist*> &fixed) override {
    Layer::setup_tensor_distribution_init(
        dists, invariants, updated, fixed);
    if (!this->distconv_enabled()) return;
    auto &prev_activations_dist = dists[this][0];
    auto &activations_dist = dists[this][1];
    auto &error_signals_dist = dists[this][2];
    auto &prev_error_signals_dist = dists[this][3];
    // Assumes zero halo all tensor for now
    const dc::IntVector overlap(this->get_num_dims(), 0);
    // prev activations
    prev_activations_dist.set_overlap(overlap);
    updated.insert(&prev_activations_dist);
    fixed.insert(&prev_activations_dist);
    // activations
    activations_dist.set_overlap(overlap);
    updated.insert(&activations_dist);
    fixed.insert(&activations_dist);
    // prev error signals
    prev_error_signals_dist.set_overlap(overlap);
    updated.insert(&prev_error_signals_dist);
    fixed.insert(&prev_error_signals_dist);
    // error signals
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
        ::distconv::get_deconvolution_output_local_tensor_shape(
            this->m_prev_activations_t,
            filter_dims, strides, false, dilations,
            this->m_groups);
    return output_spatial_local_shape;
  }

  void setup_distconv_post(size_t ws_size) override {
    Layer::setup_distconv_post(ws_size);
    if (!this->distconv_enabled()) return;

    if (dc::is_deterministic()) {
      dc::MPIRootPrintStreamInfo() << "Using deterministic convolution algorithms";
      this->m_fwd_algo = "DETERMINISTIC";
      this->m_bwd_data_algo = "DETERMINISTIC";
      this->m_bwd_filter_algo = "DETERMINISTIC";
    } else {
      this->m_fwd_algo = dc::get_convolution_bwd_data_algorithm();
      this->m_bwd_data_algo = dc::get_convolution_fwd_algorithm();
      this->m_bwd_filter_algo = dc::get_convolution_bwd_filter_algorithm();
    }

    std::vector<int> pads = this->m_pads;
    std::reverse(pads.begin(), pads.end());
    std::vector<int> strides = this->m_strides;
    std::reverse(strides.begin(), strides.end());
    std::vector<int> dilations = this->m_dilations;
    std::reverse(dilations.begin(), dilations.end());

    this->m_conv->setup(this->m_prev_activations_t,
                        this->m_kernel_t, this->m_activations_t,
                        this->m_error_signals_t, this->m_kernel_gradient_e,
                        this->m_prev_error_signals_t,
                        pads, strides, dilations, this->m_groups,
                        this->m_fwd_algo, this->m_bwd_data_algo,
                        this->m_bwd_filter_algo,
                        ws_size, this->skip_first_layer_bp(), true);
  }

 protected:
  bool using_distconv() const override {
    if (!Layer::using_distconv()) return false;

    const auto& kernel_dims = get_kernel_dims();
    for(int i = 0; i < this->get_num_spatial_dims(); i++) {
      auto pad = this->m_pads[i];
      if (pad != 0) {
        dc::MPIPrintStreamDebug() << this->get_name()
                                  << " unsupported as padding must be zero";
        return false;
      }
      auto stride_size = this->m_strides[i];
      auto filter_size = kernel_dims[2+i];
      if (!(filter_size % 2 == 0 && filter_size == stride_size)) {
        dc::MPIPrintStreamDebug() << this->get_name()
                                  << " unsupported due to filter and stride sizes";
        return false;
      }
    }
    return true;
  }
#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_DECONVOLUTION_LAYER_INSTANTIATE
extern template class deconvolution_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class deconvolution_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_DECONVOLUTION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_DECONVOLUTION_HPP_INCLUDED
