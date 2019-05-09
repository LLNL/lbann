////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

/** @brief Standard deep learning convolution.
 *
 *  Applies convolution (more precisely, cross-correlation) to input
 *  tensors. This is primarily optimized for image data in NCHW
 *  format.
 */
template <data_layout Layout = data_layout::DATA_PARALLEL, El::Device Device = El::Device::CPU>
class convolution_layer : public base_convolution_layer<Device> {
private:

  friend class lbann_callback_imcomm;


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
    static_assert(Layout == data_layout::DATA_PARALLEL,
                  "convolution layer only supports DATA_PARALLEL");

  }

  convolution_layer* copy() const override { return new convolution_layer(*this); }

  std::string get_type() const override { return "convolution"; }

  data_layout get_data_layout() const override { return Layout; }

  El::Device get_device_allocation() const override { return Device; }

protected:

  void setup_dims() override {
    base_convolution_layer<Device>::setup_dims();

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

  std::vector<int> get_kernel_dims() const {
    std::vector<int> dims;
    dims.push_back(this->m_output_channels);
    dims.push_back(this->get_input_dims()[0]);
    dims.insert(dims.end(),
                this->m_conv_dims.begin(),
                this->m_conv_dims.end());
    return dims;
  }


  void fp_compute() override {
    if(this->using_gpus()) {
#ifdef LBANN_HAS_DISTCONV
      if (this->distconv_enabled()) {
        apply_convolution_distconv();
        apply_bias_distconv();
        this->copy_out_activations();
        if (this->early_terminate_last_iteration() &&
            this->keep_original()) {
          base_convolution_layer<Device>::apply_convolution_cudnn(true);
          base_convolution_layer<Device>::apply_bias_cudnn();
          this->dump_reference_activations();
        }
      } else {
        base_convolution_layer<Device>::apply_convolution_cudnn(true);
        base_convolution_layer<Device>::apply_bias_cudnn();
      }
#else
      base_convolution_layer<Device>::apply_convolution_cudnn(true);
      base_convolution_layer<Device>::apply_bias_cudnn();
#endif
    } else {
      base_convolution_layer<Device>::apply_convolution_im2col(true);
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
          compute_gradients_distconv();
          return;
        }
        if (m_conv->is_overlap_bwd_halo_exchange_enabled()) {
          m_conv->backward_data_exchange_halo(this->m_prev_error_signals_t);
        }
        compute_gradients_distconv();
        apply_transposed_convolution_distconv();
        if (this->early_terminate_last_iteration() &&
            this->keep_original()) {
          base_convolution_layer<Device>::compute_gradients_cudnn(false);
          base_convolution_layer<Device>::apply_transposed_convolution_cudnn(false);
          this->dump_reference_error_signals();
        }
      } else {
        base_convolution_layer<Device>::compute_gradients_cudnn(false);
        base_convolution_layer<Device>::apply_transposed_convolution_cudnn(false);
      }
#else
      base_convolution_layer<Device>::compute_gradients_cudnn(false);
      base_convolution_layer<Device>::apply_transposed_convolution_cudnn(false);
#endif
    } else {
      base_convolution_layer<Device>::compute_gradients_im2col(false);
      base_convolution_layer<Device>::apply_transposed_convolution_im2col(false);
    }
  }

  void apply_convolution_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
        "Layer: DISTCONV not detected");
#else
    dc::MPIPrintStreamDebug() << this->get_name() << ": Forward convolution";

    assert0(dc::tensor::View(
        m_kernel_t, this->get_weights()[0]->get_values().LockedBuffer()));

    m_conv->forward(DataType(1.0), this->m_prev_activations_t, m_kernel_t,
                    DataType(0.0), this->m_activations_t);
#endif
  }

  void apply_bias_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__)
        + " :: " + "Layer: DISTCONV not detected");
#else
    if (this->m_bias_scaling_factor == DataType(0)) return;

    dc::MPIPrintStreamDebug() << "Applying bias";

    assert0(dc::tensor::View(
        m_bias_t, this->get_weights()[1]->get_values().LockedBuffer()));
    m_conv->apply_bias(this->m_bias_scaling_factor, m_bias_t,
                       DataType(1), this->m_activations_t);
#endif
  }

  void apply_transposed_convolution_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__)
        + " :: " + "Layer: DISTCONV not detected");
#else
    dc::MPIPrintStreamDebug() << this->get_name() << ": Backward convolution";

    // input: m_prev_error_signals_d[0]
    // kernel: m_weights[0]->get_values_gpu()
    // output: m_error_signals_d[0]

    assert0(dc::tensor::View(
        m_kernel_t, this->get_weights()[0]->get_values().LockedBuffer()));

    this->m_error_signals_t.zero(dc::get_backend().get_stream());
    dc::MPIPrintStreamDebug() << "Calling backward_data\n";
    m_conv->backward_data(DataType(1.0), m_kernel_t, this->m_prev_error_signals_t,
                          DataType(1.0), this->m_error_signals_t);

    this->copy_out_error_signals();
#endif
  }

  void compute_gradients_distconv() {
#ifndef LBANN_HAS_DISTCONV
    throw lbann_exception(
        std::string {} + __FILE__ + " " + std::to_string(__LINE__)
        + " :: " + "Layer: DISTCONV not detected");
#else
    dc::MPIPrintStreamDebug() << this->get_name() << ": Compute gradients";

    const int effective_mini_batch_size =
        this->m_model->get_effective_mini_batch_size();
    const bool has_local_data = this->m_prev_activations_t.get_local_size() > 0 &&
        this->m_prev_error_signals_t.get_local_size() > 0;

    if (this->m_bias_scaling_factor != DataType(0)
        && this->get_weights()[1]->get_optimizer() != nullptr) {
      optimizer* bias_optimizer = this->get_weights()[1]->get_optimizer();
      dc::MPIPrintStreamDebug() << "Compute bias gradients";
      DataType dst_scale = DataType(0), gradient_scale = DataType(0);
      auto& bias_gradient = bias_optimizer->get_gradient_buffer(
          dst_scale, gradient_scale, true);
      gradient_scale /= effective_mini_batch_size;
      // For comparison with the original LBANN, bias gradients will
      // be calculated again with the original LBANN. Do not accumulate the
      // gradients here as it would be otherwise accumulated twice.
      if (this->early_terminate_last_iteration()) {
        gradient_scale = 0;
      }
      assert0(dc::tensor::View(m_bias_gradient_t,
                               bias_gradient.Buffer()));
      if (has_local_data) {
        m_conv->backward_bias(gradient_scale, this->m_prev_error_signals_t,
                              dst_scale, m_bias_gradient_t, false);
      } else {
        m_bias_gradient_t.scale(dst_scale, dc::get_stream());
      }
    }

    optimizer* kernel_optimizer = this->get_weights()[0]->get_optimizer();
    if (kernel_optimizer == nullptr) return;

    dc::MPIPrintStreamDebug() << "Compute kernel gradients";

    DataType dst_scale = DataType(0), gradient_scale = DataType(0);
    auto& kernel_gradient = kernel_optimizer->get_gradient_buffer(
        dst_scale, gradient_scale, true);
    gradient_scale /= effective_mini_batch_size;

    assert0(dc::tensor::View(
        m_kernel_gradient_e, kernel_gradient.Buffer()));
    if (has_local_data) {
      m_conv->backward_filter(gradient_scale, this->m_prev_activations_t,
                              this->m_prev_error_signals_t, dst_scale,
                              m_kernel_gradient_e, false);
    } else {
      m_kernel_gradient_e.scale(dst_scale, dc::get_stream());
    }
#endif
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
    if (this->distconv_enabled()) {
      dc::IntVector overlap(dc::num_dims, 0);
      for(int i = 0; i < dc::num_spatial_dims; i++) {
#ifdef LBANN_DISTCONV_HAS_DEPTH
        const int splits = std::vector<int>(
            {this->get_parallel_strategy().depth_splits,
             this->get_parallel_strategy().height_splits,
             this->get_parallel_strategy().width_splits})[i];
#else
        const int splits = std::vector<int>(
            {this->get_parallel_strategy().height_splits,
             this->get_parallel_strategy().width_splits})[i];
#endif // LBANN_DISTCONV_HAS_DEPTH
        if(splits > 1)
          overlap[dc::num_spatial_dims - 1 - i] = (get_kernel_dims()[2 + i] - 1) / 2 * this->m_dilations[i];
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
            this->m_prev_activations_t,
            filter_dims, strides, true, dilations,
            this->m_groups);
    return output_spatial_local_shape;
  }

  void setup_tensors_fwd(const std::array<dc::Dist, dc::num_dists> &dists) override {
    using namespace dc;
    Layer::setup_tensors_fwd(dists);
    if (!this->distconv_enabled()) return;

    const auto& kernel_dims = get_kernel_dims();
    std::stringstream ss;
    util::print_vector(ss, kernel_dims.begin(), kernel_dims.end());
    MPIPrintStreamDebug()
        << "m_kernel_dims: " << ss.str();

    this->setup_prev_activations_tensor(dists);
    this->setup_activations_tensor(dists);
    this->setup_activations_copyout_tensor(dists);

    // assumes no partitioning on channel/filter dimensions
    assert_eq(dists[0].get_split_shape()[-2], 1);
    auto shared_dist = dc::Dist::make_shared_distribution(
        dists[0].get_locale_shape());

    Shape kernel_shape(kernel_dims);
    std::reverse(kernel_shape.begin(), kernel_shape.end());
    const LocaleMPI loc(dc::get_mpi_comm(), false);
    m_kernel_t = TensorDev(kernel_shape, loc, shared_dist);
    assert0(tensor::View(
        m_kernel_t, this->get_weights()[0]->get_values().LockedBuffer()));
    m_kernel_gradient_e = TensorDev(kernel_shape, loc, shared_dist);
    // Gradient buffer is needed for auto-tuning the bp filter algorithm
    assert0(tensor::View(
        m_kernel_gradient_e,
        this->get_weights()[0]->get_optimizer()->get_gradient().Buffer()));

    m_conv = new Convolution(dc::get_backend(),
                             dc::get_halo_exchange_method());

    // Bias tensor. Shared by all procs
    if (this->m_bias_scaling_factor != DataType(0)) {
      MPIPrintStreamDebug()
          << "Bias desc: "
          << dc::util::tostring(this->m_bias_cudnn_desc)
          << ", bias factor: " << this->m_bias_scaling_factor;
      std::vector<int> bias_shape_v(dc::num_dims, 1);
      bias_shape_v[dc::num_spatial_dims] = this->get_output_dims()[0];
      Shape bias_shape(bias_shape_v);
      m_bias_t = TensorDev(bias_shape, loc, shared_dist);
      assert0(tensor::View(m_bias_t,
                           this->get_weights()[1]->get_values().LockedBuffer()));
      MPIPrintStreamDebug() << "Bias tensor: " << m_bias_t;
      m_conv->setup_bias(m_bias_t);

      // Bias backprop
      optimizer* bias_optimizer = this->get_weights()[1]->get_optimizer();
      if (bias_optimizer != nullptr) {
        m_bias_gradient_t = TensorDev(bias_shape, loc, shared_dist);
        // setup_bias_gradients needs strides of the bias tensor,
        // which is set when its view is set.
        assert0(tensor::View(
            m_bias_gradient_t,
            this->get_weights()[1]->get_optimizer()->get_gradient().Buffer()));
        m_conv->setup_bias_gradient(m_bias_gradient_t);
      }
    }
  }

  void setup_tensors_bwd(const std::array<dc::Dist, dc::num_dists> &dists) override {
    Layer::setup_tensors_bwd(dists);
    if (!this->distconv_enabled()) return;

    this->setup_prev_error_signals_tensor(dists);
    this->setup_error_signals_tensor(dists);
    this->setup_error_signals_copyout_tensor(dists);
  }

  void setup_distconv_post(size_t ws_size) override {
    Layer::setup_distconv_post(ws_size);
    if (!this->distconv_enabled()) return;

    if (getenv("DISTCONV_DETERMINISTIC")) {
      // Same algorithm as LBANN
      m_fwd_algo = "IMPLICIT_GEMM";
      // Deterministic algorithm
      m_bwd_data_algo = "ALGO_1";
      m_bwd_filter_algo = "ALGO_1";
    } else {
      m_fwd_algo = dc::get_convolution_fwd_algorithm();
      m_bwd_data_algo = dc::get_convolution_bwd_data_algorithm();
      m_bwd_filter_algo = dc::get_convolution_bwd_filter_algorithm();
    }

    std::vector<int> pads = this->m_pads;
    std::reverse(pads.begin(), pads.end());
    std::vector<int> strides = this->m_strides;
    std::reverse(strides.begin(), strides.end());
    std::vector<int> dilations = this->m_dilations;
    std::reverse(dilations.begin(), dilations.end());

    m_conv->setup(this->m_prev_activations_t,
                  m_kernel_t, this->m_activations_t,
                  this->m_error_signals_t, m_kernel_gradient_e,
                  this->m_prev_error_signals_t,
                  pads, strides, dilations, this->m_groups,
                  m_fwd_algo, m_bwd_data_algo,
                  m_bwd_filter_algo,
                  ws_size, this->skip_first_layer_bp());
  }

 protected:
  dc::Convolution *m_conv;
  dc::TensorDev m_kernel_t;
  dc::TensorDev m_kernel_gradient_e;
  // Bias
  dc::TensorDev m_bias_t;
  dc::TensorDev m_bias_gradient_t;
  // Algorithms
  std::string m_fwd_algo;
  std::string m_bwd_data_algo;
  std::string m_bwd_filter_algo;

  bool using_distconv() const override {
    if (!Layer::using_distconv()) return false;

    bool cond = true;
    const auto& kernel_dims = get_kernel_dims();
    for(int i = 0; i < dc::num_spatial_dims; i++) {
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

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_CONVOLUTION_HPP_INCLUDED
