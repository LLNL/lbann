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

#ifndef LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED

#include "lbann_config.hpp"
#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/models/model.hpp"
#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/cuda.hpp"
#include "lbann/utils/distconv.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

enum class batch_normalization_stats_aggregation {
  /** Statistics are aggregated only within a single rank. */
  local,
  /** Statistics are aggregated among every rank in a single node. */
  node_local,
  /** Statistics are aggregated among every rank in the model. */
  global
};

/** @brief
 *
 *  Each input channel is normalized across the mini-batch to have
 *  zero mean and unit standard deviation. Learned scaling factors and
 *  biases are then applied. This uses the standard approach of
 *  maintaining the running mean and standard deviation (with
 *  exponential decay) for use at test time. See:
 *
 *  Sergey Ioffe and Christian Szegedy. "Batch Normalization:
 *  Accelerating Deep Network Training by Reducing Internal Covariate
 *  Shift." In International Conference on Machine Learning,
 *  pp. 448-456. 2015.
 */
template <data_layout T_layout, El::Device Dev>
class batch_normalization_layer : public regularizer_layer {

private:

  /** Decay rate for the running statistics. */
  DataType m_decay;
  /** Small number to avoid division by zero. */
  DataType m_epsilon;
  /** Type of statistics aggregation to use. */
  batch_normalization_stats_aggregation m_stats_aggregation;
  /**
   * Cache of node-local num_per_sum results for node-local stats.
   * Indexed by effective mini-batch size.
   */
  std::unordered_map<El::Int, El::Int> m_num_per_sum_cache;

  /** Current minibatch means. */
  std::unique_ptr<AbsDistMat> m_mean;
  /** Current minibatch standard deviations. */
  std::unique_ptr<AbsDistMat> m_var;
  /** Gradient w.r.t. means. */
  std::unique_ptr <AbsDistMat> m_mean_gradient;
  /** Gradient w.r.t. standard deviations. */
  std::unique_ptr<AbsDistMat> m_var_gradient;
  /** Gradient w.r.t. scaling terms. */
  std::unique_ptr<AbsDistMat> m_scale_gradient;
  /** Gradient w.r.t. bias terms. */
  std::unique_ptr<AbsDistMat> m_bias_gradient;

public:
  /**
   * Set up batch normalization.
   * @param decay Controls the momentum of the running mean/standard
   * deviation averages.
   * @param epsilon A small number to avoid division by zero.
   * @param use_global_stats Whether to use global statistics when
   * training.
   */
  batch_normalization_layer(lbann_comm *comm,
                            DataType decay=0.9,
                            DataType epsilon=1e-5,
                            batch_normalization_stats_aggregation stats_aggregation =
                            batch_normalization_stats_aggregation::local)
    : regularizer_layer(comm),
      m_decay(decay),
      m_epsilon(epsilon),
      m_stats_aggregation(stats_aggregation) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "batch normalization only supports DATA_PARALLEL");
#ifdef LBANN_DETERMINISTIC
    // Force global computation.
    m_stats_aggregation = batch_normalization_stats_aggregation::global;
#endif
  }

  batch_normalization_layer(const batch_normalization_layer& other)
    : regularizer_layer(other),
      m_decay(other.m_decay),
      m_epsilon(other.m_epsilon),
      m_stats_aggregation(other.m_stats_aggregation),
      m_num_per_sum_cache(other.m_num_per_sum_cache),
      m_mean(other.m_mean ? other.m_mean->Copy() : nullptr),
      m_var(other.m_var ? other.m_var->Copy() : nullptr),
      m_mean_gradient(other.m_mean_gradient ?
                      other.m_mean_gradient->Copy() : nullptr),
      m_var_gradient(other.m_var_gradient ?
                     other.m_var_gradient->Copy() : nullptr),
      m_scale_gradient(other.m_scale_gradient ?
                       other.m_scale_gradient->Copy() : nullptr),
      m_bias_gradient(other.m_bias_gradient ?
                      other.m_bias_gradient->Copy() : nullptr) {}

  batch_normalization_layer& operator=(const batch_normalization_layer& other) {
    regularizer_layer::operator=(other);
    m_decay = other.m_decay;
    m_epsilon = other.m_epsilon;
    m_stats_aggregation = other.m_stats_aggregation;
    m_num_per_sum_cache = other.m_num_per_sum_cache;

    // Deep copy matrices
    m_mean.reset(other.m_mean ? other.m_mean->Copy() : nullptr);
    m_var.reset(other.m_var ? other.m_var->Copy() : nullptr);
    m_mean_gradient.reset(other.m_mean_gradient ?
                          other.m_mean_gradient->Copy() : nullptr);
    m_var_gradient.reset(other.m_var_gradient ?
                         other.m_var_gradient->Copy() : nullptr);
    m_scale_gradient.reset(other.m_scale_gradient ?
                           other.m_scale_gradient->Copy() : nullptr);
    m_bias_gradient.reset(other.m_bias_gradient ?
                          other.m_bias_gradient->Copy() : nullptr);

    return *this;
  }

  batch_normalization_layer* copy() const override { return new batch_normalization_layer(*this); }
  std::string get_type() const override { return "batch normalization"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto&& desc = regularizer_layer::get_description();
    desc.add("Decay", m_decay);
    desc.add("Epsilon", m_epsilon);
    switch (m_stats_aggregation) {
    case batch_normalization_stats_aggregation::local:
      desc.add("Statistics aggregation", "local");
      break;
    case batch_normalization_stats_aggregation::node_local:
      desc.add("Statistics aggregation", "node-local");
      break;
    case batch_normalization_stats_aggregation::global:
      desc.add("Statistics aggregation", "global");
      break;
    }
    return desc;
  }

protected:

  void setup_matrices(const El::Grid& grid) override {
    regularizer_layer::setup_matrices(grid);
    m_mean.reset(new StarMat<Dev>(grid));
    m_var.reset(new StarMat<Dev>(grid));
    m_mean_gradient.reset(new StarMat<Dev>(grid));
    m_var_gradient.reset(new StarMat<Dev>(grid));
    m_scale_gradient.reset(new StarMat<Dev>(grid));
    m_bias_gradient.reset(new StarMat<Dev>(grid));
  }

  void setup_dims() override {
    regularizer_layer::setup_dims();
    set_output_dims(get_input_dims());
  }

  void setup_data() override {
    regularizer_layer::setup_data();
    const auto& output_dims = get_output_dims();
    const auto& num_channels = output_dims[0];

    // Display warning if mini-batch size is small
    const auto& output = get_activations();
    const auto& mini_batch_size = output.Width();
    const auto& local_mini_batch_size = mini_batch_size / output.DistSize();
    if (m_stats_aggregation == batch_normalization_stats_aggregation::global
        && mini_batch_size <= 4) {
      std::stringstream err;
      err << "LBANN warning: "
          << get_type() << " layer \"" << get_name() << "\" "
          << "is using global statistics and "
          << "the mini-batch size (" << mini_batch_size << ") "
          << "may be too small to get good statistics";
      if (output.DistRank() == 0) {
        std::cerr << err.str() << std::endl;
      }
    } else if (m_stats_aggregation == batch_normalization_stats_aggregation::node_local
               && local_mini_batch_size*m_comm->get_procs_per_node() <= 4) {
      std::stringstream err;
      err << "LBANN warning: "
          << get_type() << " layer \"" << get_name() << "\" "
          << "is using node-local statistics and "
          << "the node-local mini-batch size ("
          << (local_mini_batch_size*m_comm->get_procs_per_node()) << ") "
          << "may be too small to get good statistics";
      if (output.DistRank() == 0) {
        std::cerr << err.str() << std::endl;
      }
    } else if (m_stats_aggregation == batch_normalization_stats_aggregation::local
               && local_mini_batch_size <= 4) {
      std::stringstream err;
      err << "LBANN warning: "
          << get_type() << " layer \"" << get_name() << "\" "
          << "is using local statistics and "
          << "the local mini-batch size (" << local_mini_batch_size << ") "
          << "may be too small to get good statistics";
      if (output.DistRank() == 0) {
        std::cerr << err.str() << std::endl;
      }
    }

    // Initialize default weights if none are provided
    if (this->m_weights.size() > 4) {
      std::stringstream err;
      err << "attempted to setup layer \"" << m_name << "\" "
          << "with an invalid number of weights";
      LBANN_ERROR(err.str());
    }
    this->m_weights.resize(4, nullptr);
    if (this->m_weights[0] == nullptr) {
      this->m_weights[0] = new weights(get_comm());
      std::unique_ptr<weights_initializer> init(new constant_initializer(DataType(1)));
      std::unique_ptr<optimizer> opt(m_model->create_optimizer());
      this->m_weights[0]->set_name(get_name() + "_scale");
      this->m_weights[0]->set_initializer(init);
      this->m_weights[0]->set_optimizer(opt);
      this->m_model->add_weights(this->m_weights[0]);
    }
    if (this->m_weights[1] == nullptr) {
      this->m_weights[1] = new weights(get_comm());
      std::unique_ptr<weights_initializer> init(new constant_initializer(DataType(0)));
      std::unique_ptr<optimizer> opt(m_model->create_optimizer());
      this->m_weights[1]->set_name(get_name() + "_bias");
      this->m_weights[1]->set_initializer(init);
      this->m_weights[1]->set_optimizer(opt);
      this->m_model->add_weights(this->m_weights[1]);
    }
    if (this->m_weights[2] == nullptr) {
      this->m_weights[2] = new weights(get_comm());
      this->m_weights[2]->set_name(get_name() + "_running_mean");
      std::unique_ptr<weights_initializer> init(new constant_initializer(DataType(0)));
      this->m_weights[2]->set_initializer(init);
      this->m_model->add_weights(this->m_weights[2]);
    }
    if (this->m_weights[3] == nullptr) {
      this->m_weights[3] = new weights(get_comm());
      this->m_weights[3]->set_name(get_name() + "_running_variance");
      std::unique_ptr<weights_initializer> init(new constant_initializer(DataType(1)));
      this->m_weights[3]->set_initializer(init);
      this->m_model->add_weights(this->m_weights[3]);
    }

    // Setup weights
    auto dist = get_prev_activations().DistData();
    dist.colDist = El::STAR;
    dist.rowDist = El::STAR;
    for (auto* w : this->m_weights) {
      w->set_dims(num_channels);
      w->set_matrix_distribution(dist);
    }

    // Initialize matrices
    El::Zeros(*m_mean,           num_channels, 1);
    El::Zeros(*m_var,            num_channels, 1);
    El::Zeros(*m_mean_gradient,  num_channels, 1);
    El::Zeros(*m_var_gradient,   num_channels, 1);
    El::Zeros(*m_scale_gradient, num_channels, 1);
    El::Zeros(*m_bias_gradient,  num_channels, 1);

    // Initialize freeze state
    for (auto&& w : this->m_weights) {
      if (m_frozen) {
        w->freeze();
      } else {
        w->unfreeze();
      }
    }
    for (auto&& w : this->m_weights) {
      if (w->is_frozen() != m_frozen) {
        std::stringstream err;
        err << (m_frozen ? "" : "un") << "frozen "
            << "layer \"" << get_name() << "\" has "
            << (w->is_frozen() ? "" : "un") << "frozen "
            << "weights \"" << w->get_name() << "\"";
        LBANN_ERROR(err.str());
      }
    }

  }

  void fp_compute() override;
  void bp_compute() override;

#ifdef LBANN_HAS_DISTCONV
<<<<<<< bd7f96c0a16a9d632d30ffc2e915b5ac619de31c
 protected:
  void fp_compute_distconv();
  void bp_compute_distconv();

=======
      if (distconv_enabled()) {
        bp_compute_distconv();
        if (early_terminate_last_iteration()) {
          assert0(dc::tensor::View(
              m_error_signals_copyout,
              get_error_signals().Buffer()));
          m_error_signals_copyout.zero();
          bp_compute_gpu();
          dump_reference_error_signals();
        }
        return;
      }
#endif
      bp_compute_gpu();
    } else {
      bp_compute_cpu();
    }
  }

  void fp_compute_gpu() {
#ifndef LBANN_HAS_GPU
    LBANN_ERROR("CUDA not detected");
#else

    // Matrices
    const auto& input = get_prev_activations();
    const auto& local_input = input.LockedMatrix();
    auto& local_output = get_local_activations();

    // Compute statistics during training
    const bool is_training = this->m_model->get_execution_mode() == execution_mode::training;
    if (is_training) {
      const auto& output_dims = get_output_dims();
      const int num_channels = output_dims[0];
      const int channel_size = get_output_size() / num_channels;
      batch_normalization_cuda::channel_sums(num_channels,
                                             local_input,
                                             m_mean->Matrix(),
                                             m_var->Matrix());
      int num_per_sum = channel_size * input.LocalWidth();
      if (m_use_global_stats) {
        m_comm->allreduce(*m_mean, m_mean->RedundantComm(), El::mpi::SUM);
        m_comm->allreduce(*m_var, m_var->RedundantComm(), El::mpi::SUM);
        num_per_sum = channel_size * input.Width();
      }
      batch_normalization_cuda::compute_statistics(
        num_per_sum,
        m_epsilon,
        m_decay,
        m_mean->Matrix(),
        m_var->Matrix(),
        m_weights[2]->get_values().Matrix(),
        m_weights[3]->get_values().Matrix());
    }

    // Perform batch normalization
    const auto& mean = (is_training ?
                        m_mean->LockedMatrix() :
                        m_weights[2]->get_values().Matrix());
    const auto& var = (is_training ?
                       m_var->LockedMatrix() :
                       m_weights[3]->get_values().Matrix());
    batch_normalization_cuda::batch_normalization(
      local_input,
      mean, var, m_epsilon,
      m_weights[0]->get_values().Matrix(),
      m_weights[1]->get_values().Matrix(),
      local_output);

#endif // LBANN_HAS_GPU
  }

  void bp_compute_gpu() {
#ifndef LBANN_HAS_GPU
    LBANN_ERROR("CUDA not detected");
#else
    const bool is_training = this->m_model->get_execution_mode() == execution_mode::training;
    const int effective_mini_batch_size = this->m_model->get_effective_mini_batch_size();

    // GPU matrices
    const auto& input = get_prev_activations();
    const auto& local_input = input.LockedMatrix();
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_gradient_wrt_input = get_local_error_signals();
    const auto& mean = (is_training ?
                        m_mean->LockedMatrix() :
                        m_weights[2]->get_values().Matrix());
    const auto& var = (is_training ?
                       m_var->LockedMatrix() :
                       m_weights[3]->get_values().Matrix());

    // Compute gradients w.r.t. batch norm parameters
    batch_normalization_cuda::backprop1(local_input,
                                        local_gradient_wrt_output,
                                        mean, var, m_epsilon,
                                        m_weights[0]->get_values().Matrix(),
                                        m_scale_gradient->Matrix(),
                                        m_bias_gradient->Matrix(),
                                        m_mean_gradient->Matrix(),
                                        m_var_gradient->Matrix());

    // Accumulate gradients
    if (is_training) {
      if (m_use_global_stats) {
        m_comm->allreduce(*m_mean_gradient,
                          m_mean_gradient->RedundantComm(),
                          El::mpi::SUM);
        m_comm->allreduce(*m_var_gradient,
                          m_var_gradient->RedundantComm(),
                          El::mpi::SUM);
      }
    } else {
      El::Zero(*m_mean_gradient);
      El::Zero(*m_var_gradient);
    }
    auto* scale_optimizer = m_weights[0]->get_optimizer();
    if (scale_optimizer != nullptr) {
      scale_optimizer->add_to_gradient_staging(
        *m_scale_gradient,
        DataType(1) / effective_mini_batch_size);
    }
    auto* bias_optimizer = m_weights[1]->get_optimizer();
    if (bias_optimizer != nullptr) {
      bias_optimizer->add_to_gradient_staging(
        *m_bias_gradient,
        DataType(1) / effective_mini_batch_size);
    }

    // Compute gradient w.r.t. input
    batch_normalization_cuda::backprop2(m_use_global_stats ? input.Width() : input.LocalWidth(),
                                        local_input,
                                        local_gradient_wrt_output,
                                        mean, var, m_epsilon,
                                        m_weights[0]->get_values().Matrix(),
                                        m_mean_gradient->LockedMatrix(),
                                        m_var_gradient->LockedMatrix(),
                                        local_gradient_wrt_input);

#endif // LBANN_HAS_GPU
  }

  void fp_compute_cpu() {
    const DataType zero = DataType(0);
    const DataType one = DataType(1);

    // Check execution mode
    const bool is_training = this->m_model->get_execution_mode() == execution_mode::training;

    // Matrices
    const auto& input = get_prev_activations();
    const auto& local_input = input.LockedMatrix();
    auto& local_output = get_local_activations();

    // Matrix parameters
    const int width = input.Width();
    const El::Int local_width = local_input.Width();
    const auto& output_dims = get_output_dims();
    const int num_channels = output_dims[0];
    const int channel_size = get_output_size() / num_channels;

    // Compute statistics
    if (is_training) {

      // Local matrices
      // Note: local_new_running_mean and local_new_running_var are
      // stored in m_mean_gradient and m_var_gradient.
      auto& local_mean = m_mean->Matrix();
      auto& local_var = m_var->Matrix();
      const auto& local_running_mean = this->m_weights[2]->get_values().LockedMatrix();
      const auto& local_running_var = this->m_weights[3]->get_values().LockedMatrix();
      auto& local_new_running_mean = m_mean_gradient->Matrix();
      auto& local_new_running_var = m_var_gradient->Matrix();

      // Compute sums and sums of squares
      #pragma omp parallel for
      for (int channel = 0; channel < num_channels; ++channel) {
        DataType sum = zero;
        DataType sqsum = zero;
        const El::Int row_start = channel * channel_size;
        const El::Int row_end = (channel+1) * channel_size;
        for (El::Int col = 0; col < local_width; ++col) {
          for (El::Int row = row_start; row < row_end; ++row) {
            const DataType x = local_input(row, col);
            sum += x;
            sqsum += x * x;
          }
        }
        local_mean(channel, 0) = sum;
        local_var(channel, 0) = sqsum;
      }
      DataType num_per_sum;
      if (m_use_global_stats) {
        m_comm->allreduce(*m_mean, m_mean->RedundantComm(), El::mpi::SUM);
        m_comm->allreduce(*m_var, m_var->RedundantComm(), El::mpi::SUM);
        num_per_sum = channel_size * width;
      } else {
        num_per_sum = channel_size * local_width;
      }

      // Compute minibatch statistics
      // Note: local_new_running_mean and local_new_running_var are
      // stored in m_mean_gradient and m_var_gradient.
      if (num_per_sum <= 1) {
        El::Fill(local_var, one);
      } else {
        #pragma omp parallel for
        for (int channel = 0; channel < num_channels; ++channel) {
          const DataType mean = local_mean(channel, 0) / num_per_sum;
          const DataType sqmean = local_var(channel, 0) / num_per_sum;
          const DataType var = num_per_sum / (num_per_sum - one) * std::max(sqmean - mean * mean, m_epsilon);
          const DataType old_running_mean = local_running_mean(channel, 0);
          const DataType old_running_var = local_running_var(channel, 0);
          const DataType new_running_mean = m_decay * old_running_mean + (one - m_decay) * mean;
          const DataType new_running_var = m_decay * old_running_var + (one - m_decay) * var;
          local_mean(channel, 0) = mean;
          local_var(channel, 0) = var;
          local_new_running_mean(channel, 0) = new_running_mean;
          local_new_running_var(channel, 0) = new_running_var;
        }
        m_weights[2]->set_values(*m_mean_gradient);
        m_weights[3]->set_values(*m_var_gradient);
      }

    }

    // Get matrices
    const auto& local_scale = this->m_weights[0]->get_values().LockedMatrix();
    const auto& local_bias = this->m_weights[1]->get_values().LockedMatrix();
    const auto& local_mean = (is_training ?
                              m_mean->LockedMatrix() :
                              this->m_weights[2]->get_values().LockedMatrix());
    const auto& local_var = (is_training ?
                             m_var->LockedMatrix() :
                             this->m_weights[3]->get_values().LockedMatrix());

    // Iterate through channels
    #pragma omp parallel for
    for (int channel = 0; channel < num_channels; ++channel) {

      // Get channel parameters
      const DataType mean = local_mean(channel, 0);
      const DataType var = local_var(channel, 0);
      const DataType inv_stdev = 1 / std::sqrt(var + m_epsilon);
      const DataType scale = local_scale(channel, 0);
      const DataType bias = local_bias(channel, 0);

      // Apply batch normalization to inputs in channel
      const El::Int row_start = channel * channel_size;
      const El::Int row_end = (channel+1) * channel_size;
      for (El::Int col = 0; col < local_width; ++col) {
        for (El::Int row = row_start; row < row_end; ++row) {
          const DataType x = local_input(row, col);
          const DataType xhat = (x - mean) * inv_stdev;
          const DataType y = scale * xhat + bias;
          local_output(row, col) = y;
        }
      }

    }

  }

  void bp_compute_cpu() {

    // Check execution mode
    const bool is_training = this->m_model->get_execution_mode() == execution_mode::training;

    // Matrices
    const auto& local_scale = this->m_weights[0]->get_values().LockedMatrix();
    const auto& local_mean = (is_training ?
                              m_mean->LockedMatrix() :
                              this->m_weights[2]->get_values().LockedMatrix());
    const auto& local_var = (is_training ?
                             m_var->LockedMatrix() :
                             this->m_weights[3]->get_values().LockedMatrix());
    const auto& input = get_prev_activations();
    const auto& local_input = input.LockedMatrix();
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_gradient_wrt_input = get_local_error_signals();
    auto& local_mean_gradient = m_mean_gradient->Matrix();
    auto& local_var_gradient = m_var_gradient->Matrix();
    auto& local_scale_gradient = m_scale_gradient->Matrix();
    auto& local_bias_gradient = m_bias_gradient->Matrix();

    // Matrix parameters
    const int effective_mini_batch_size = this->m_model->get_effective_mini_batch_size();
    const int width = input.Width();
    const El::Int local_width = local_input.Width();
    const auto& output_dims = get_output_dims();
    const int num_channels = output_dims[0];
    const int channel_size = get_output_size() / num_channels;

    // Compute local gradients
    #pragma omp parallel for
    for (int channel = 0; channel < num_channels; ++channel) {

      // Initialize channel parameters and gradients
      const DataType mean = local_mean(channel, 0);
      const DataType var = local_var(channel, 0);
      const DataType scale = local_scale(channel, 0);
      const DataType inv_stdev = 1 / std::sqrt(var + m_epsilon);
      const DataType dvar_factor = inv_stdev * inv_stdev * inv_stdev / 2;
      DataType dmean = DataType(0);
      DataType dvar = DataType(0);
      DataType dscale = DataType(0);
      DataType dbias = DataType(0);

      // Compute gradient contributions from local entries
      const El::Int row_start = channel * channel_size;
      const El::Int row_end = (channel+1) * channel_size;
      for (El::Int col = 0; col < local_width; ++col) {
        for (El::Int row = row_start; row < row_end; ++row) {
          const DataType x = local_input(row, col);
          const DataType xhat = (x - mean) * inv_stdev;
          const DataType dy = local_gradient_wrt_output(row, col);
          dscale += dy * xhat;
          dbias += dy;
          const DataType dxhat = dy * scale;
          dmean += - dxhat * inv_stdev;
          dvar += - dxhat * (x - mean) * dvar_factor;
        }
      }
      local_mean_gradient(channel, 0) = dmean;
      local_var_gradient(channel, 0) = dvar;
      local_scale_gradient(channel, 0) = dscale;
      local_bias_gradient(channel, 0) = dbias;

    }

    // Accumulate gradients
    if (is_training) {
      if (m_use_global_stats) {
        m_comm->allreduce(*m_mean_gradient,
                          m_mean_gradient->RedundantComm(),
                          El::mpi::SUM);
        m_comm->allreduce(*m_var_gradient,
                          m_var_gradient->RedundantComm(),
                          El::mpi::SUM);
      }
    } else {
      El::Zero(*m_mean_gradient);
      El::Zero(*m_var_gradient);
    }
    optimizer* scale_optimizer = m_weights[0]->get_optimizer();
    if (scale_optimizer != nullptr) {
      scale_optimizer->add_to_gradient_staging(
        *m_scale_gradient,
        DataType(1) / effective_mini_batch_size);
    }
    optimizer* bias_optimizer = m_weights[1]->get_optimizer();
    if (bias_optimizer != nullptr) {
      bias_optimizer->add_to_gradient_staging(
        *m_bias_gradient,
        DataType(1) / effective_mini_batch_size);
    }

    // Compute error signal
    const int num_per_sum = (m_use_global_stats ?
                             width * channel_size :
                             local_width * channel_size);
    if (num_per_sum <= 1) {
      El::Zero(local_gradient_wrt_input);
    } else {
      #pragma omp parallel for
      for (int channel = 0; channel < num_channels; ++channel) {

        // Initialize channel parameters and gradients
        const auto& mean = local_mean(channel, 0);
        const auto& var = local_var(channel, 0);
        const auto& scale = local_scale(channel, 0);
        const auto& dmean = local_mean_gradient(channel, 0);
        const auto& dvar = local_var_gradient(channel, 0);

        // Compute useful constants
        const DataType inv_stdev = 1 / std::sqrt(var + m_epsilon);
        const auto& dmean_term = dmean / num_per_sum;
        const auto& dvar_term = dvar * 2 / (num_per_sum - 1);

        // Compute error signal for current channel
        const El::Int row_start = channel * channel_size;
        const El::Int row_end = (channel+1) * channel_size;
        for (El::Int col = 0; col < local_width; ++col) {
          for (El::Int row = row_start; row < row_end; ++row) {
            const auto& x = local_input(row, col);
            const auto& dy = local_gradient_wrt_output(row, col);
            const auto& dxhat = dy * scale;
            auto dx = dxhat * inv_stdev;
            dx += dmean_term;
            dx += dvar_term * (x - mean);
            local_gradient_wrt_input(row, col) = dx;
          }
        }

      }
    }

  }

 private:

  void deallocate_matrices() {
    if (m_mean != nullptr)           delete m_mean;
    if (m_var != nullptr)            delete m_var;
    if (m_mean_gradient != nullptr)  delete m_mean_gradient;
    if (m_var_gradient != nullptr)   delete m_var_gradient;
    if (m_scale_gradient != nullptr) delete m_scale_gradient;
    if (m_bias_gradient != nullptr)  delete m_bias_gradient;
    m_mean = nullptr;
    m_var = nullptr;
    m_mean_gradient = nullptr;
    m_var_gradient = nullptr;
    m_scale_gradient = nullptr;
    m_bias_gradient = nullptr;
  }

#ifdef LBANN_HAS_DISTCONV
 public:

  void fp_compute_distconv() {
    dc::MPIPrintStreamDebug() << get_name() << ": " << __FUNCTION__ << "\n";
    assert_always(distconv_enabled());

    const bool is_training =
        this->m_model->get_execution_mode() == execution_mode::training;
    
    assert_always(this->m_model->get_current_mini_batch_size() ==
                  get_prev_activations().Width());

    assert0(dc::tensor::View(
        m_scale_t, get_weights()[0]->get_values().LockedBuffer()));
    assert0(dc::tensor::View(
        m_bias_t, get_weights()[1]->get_values().LockedBuffer()));
    assert0(dc::tensor::View(
        m_running_mean_t, get_weights()[2]->get_values().Buffer()));
    assert0(dc::tensor::View(
        m_running_var_t, get_weights()[3]->get_values().Buffer()));

    m_bn->forward(m_prev_activations_t,
                  m_mean_t,
                  m_var_t,
                  m_running_mean_t,
                  m_running_var_t,
                  m_scale_t,
                  m_bias_t,
                  m_activations_t,
                  is_training);

    copy_out_activations();
  }
  
  void bp_compute_distconv() {
    dc::MPIPrintStreamDebug() << get_name() << ": " << __FUNCTION__ << "\n";
    assert_always(distconv_enabled());

    // Check execution mode
    const bool is_training = this->m_model->get_execution_mode() == execution_mode::training;
    
    //assert_always(is_training && m_use_global_stats);
    assert_always(is_training);
    
    assert0(dc::tensor::View(
        m_scale_t, get_weights()[0]->get_values().LockedBuffer()));
    
    m_bn->backward_stage1(m_prev_activations_t,
                          m_prev_error_signals_t,
                          m_mean_t, m_var_t, m_scale_t,
                          m_scale_gradient_t, m_bias_gradient_t,
                          m_mean_gradient_t, m_var_gradient_t,
                          false);
        
    // Verbatim copy from bp_compute_gpu
    // Accumulate gradients
    if (is_training) {
      if (m_use_global_stats) {
        m_comm->allreduce(*m_mean_gradient,
                          m_mean_gradient->RedundantComm(),
                          El::mpi::SUM);
        m_comm->allreduce(*m_var_gradient,
                          m_var_gradient->RedundantComm(),
                          El::mpi::SUM);
      }
    } else {
      Zero(*m_mean_gradient);
      Zero(*m_var_gradient);
    }

    const int effective_mini_batch_size = this->m_model->get_effective_mini_batch_size();
    
    optimizer* scale_optimizer = m_weights[0]->get_optimizer();
    if (scale_optimizer != nullptr) {
      scale_optimizer->add_to_gradient_staging(
          *m_scale_gradient,
          DataType(1) / effective_mini_batch_size);
    }
    optimizer* bias_optimizer = m_weights[1]->get_optimizer();
    if (bias_optimizer != nullptr) {
      bias_optimizer->add_to_gradient_staging(
          *m_bias_gradient,
          DataType(1) / effective_mini_batch_size);
    }

    m_bn->backward_stage2(m_prev_activations_t,
                          m_prev_error_signals_t,
                          m_mean_t, m_var_t, m_scale_t,
                          m_mean_gradient_t, m_var_gradient_t,
                          m_error_signals_t);

    copy_out_error_signals();
  }
    
 protected:
>>>>>>> Support local statistics in batch norm
  dc::BatchNormalization *m_bn;
  dc::TensorDev m_mean_t;
  dc::TensorDev m_var_t;
  dc::TensorDev m_scale_t;
  dc::TensorDev m_bias_t;
  dc::TensorDev m_running_mean_t;
  dc::TensorDev m_running_var_t;
  dc::TensorDev m_mean_gradient_t;
  dc::TensorDev m_var_gradient_t;
  dc::TensorDev m_scale_gradient_t;
  dc::TensorDev m_bias_gradient_t;

  bool using_distconv() const override {
    char *env = getenv("DISTCONV_DISABLE");
    if (env) {
      std::string s(env);
      if (s.find(get_name()) != std::string::npos) {
        return false;
      }
    }
    return true;
  }

  void setup_tensors_fwd(const std::array<dc::Dist, 4> &dists) override {
    Layer::setup_tensors_fwd(dists);
    if (!distconv_enabled()) return;

    setup_prev_activations_tensor(dists);
    setup_activations_tensor(dists);
    setup_activations_copyout_tensor(dists);

    dc::MPIPrintStreamDebug()
        << "BN prev_activations: " << m_prev_activations_t
        << ", activations: " << m_activations_t << "\n";

    const int num_channels = this->get_output_dims()[0];
    dc::Array4 per_channel_stat_shape = {1, 1, num_channels, 1};
    const auto shared_dist = dc::Dist();
    const dc::LocaleMPI loc(m_comm->get_model_comm().comm, false);
    // mean
    m_mean_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    assert0(dc::tensor::View(m_mean_t, this->m_mean->Buffer()));
    // var
    m_var_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    assert0(dc::tensor::View(m_var_t, this->m_var->Buffer()));
    // scale: view to weights[0]
    m_scale_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    // bias: view to weights[1]
    m_bias_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    // running_mean: view to weights[2]
    m_running_mean_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    // running_var: view to weights[3]
    m_running_var_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    // scale_gradient
    m_scale_gradient_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    assert0(dc::tensor::View(
        m_scale_gradient_t, this->m_scale_gradient->Buffer()));
    // bias_gradient
    m_bias_gradient_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    assert0(dc::tensor::View(
        m_bias_gradient_t, this->m_bias_gradient->Buffer()));
    // mean_gradient
    m_mean_gradient_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    assert0(dc::tensor::View(
        m_mean_gradient_t, this->m_mean_gradient->Buffer()));
    // var_gradient
    m_var_gradient_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    assert0(dc::tensor::View(
        m_var_gradient_t, this->m_var_gradient->Buffer()));

    // spatial decomposition requires global communication
    // m_use_global_stats = true;
  }

  void setup_tensors_bwd(const std::array<dc::Dist, 4> &dists) override {
    Layer::setup_tensors_bwd(dists);
    if (!distconv_enabled()) return;

    setup_prev_error_signals_tensor(dists);
    setup_error_signals_tensor(dists);
    setup_error_signals_copyout_tensor(dists);

    m_bn = new dc::BatchNormalization(
        dc::get_backend(this->get_comm()->get_model_comm().comm), m_decay, m_epsilon,
        m_use_global_stats);

    dc::MPIPrintStreamDebug()
        << "BN prev_error_signals: " << m_prev_error_signals_t
        << ", error_signals: " << m_error_signals_t << "\n";
  }
#endif

};

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
