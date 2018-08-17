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

#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/utils/cuda.hpp"

namespace lbann {

#ifdef LBANN_HAS_GPU
namespace batch_normalization_cuda {
/** Compute channel sums.
 *  Sums and squares of sums are used to compute mean and variance.
 */
void channel_sums(int num_channels,
                  const AbsMat& data,
                  AbsMat& sums,
                  AbsMat& sqsums);
/** Compute statistics from sums.
 *  On input, mean and var are assumed to contain sums and squares
 *  of sums, respectively.
 */
void compute_statistics(int num_per_sum,
                        DataType epsilon,
                        DataType decay,
                        AbsMat& mean,
                        AbsMat& var,
                        AbsMat& running_mean,
                        AbsMat& running_var);  
/** Apply batch normalization. */
void batch_normalization(const AbsMat& input,
                         const AbsMat& mean,
                         const AbsMat& var,
                         DataType epsilon,
                         const AbsMat& scale,
                         const AbsMat& bias,
                         AbsMat& output);
/** Compute gradients w.r.t. batch norm parameters. */
void backprop1(const AbsMat& input,
               const AbsMat& gradient_wrt_output,
               const AbsMat& mean,
               const AbsMat& var,
               DataType epsilon,
               const AbsMat& scale,
               AbsMat& dscale,
               AbsMat& dbias,
               AbsMat& dmean,
               AbsMat& dvar);
/** Compute gradients w.r.t. inputs. */
void backprop2(int global_width,
               const AbsMat& input,
               const AbsMat& gradient_wrt_output,
               const AbsMat& mean,
               const AbsMat& var,
               DataType epsilon,
               const AbsMat& scale,
               const AbsMat& dmean,
               const AbsMat& dvar,
               AbsMat& gradient_wrt_input);
}
#endif // LBANN_HAS_GPU

/** Batch normalization layer.
 *  Each input channel is normalized across the mini-batch to have
 *  zero mean and unit standard deviation. Learned scaling factors and
 *  biases are then applied. See:
 *    Sergey Ioffe and Christian Szegedy. "Batch Normalization:
 *    Accelerating Deep Network Training by Reducing Internal
 *    Covariate Shift." ICML 2015.
 *  This uses the standard approach of maintaining the running mean
 *  and standard deviation (with exponential decay) for use at test
 *  time. See:
 *    https://cthorey.github.io/backpropagation/
 */
template <data_layout T_layout, El::Device Dev>
class batch_normalization : public regularizer_layer {

 private:

  /** Decay rate for the running statistics. */
  DataType m_decay;
  /** Small number to avoid division by zero. */
  DataType m_epsilon;
  /** Whether to use global statistics when training. */
  bool m_use_global_stats;

  /** Current minibatch means. */
  AbsDistMat *m_mean;
  /** Current minibatch standard deviations. */
  AbsDistMat *m_var;
  /** Gradient w.r.t. means. */
  AbsDistMat *m_mean_gradient;
  /** Gradient w.r.t. standard deviations. */
  AbsDistMat *m_var_gradient;
  /** Gradient w.r.t. scaling terms. */
  AbsDistMat *m_scale_gradient;
  /** Gradient w.r.t. bias terms. */
  AbsDistMat *m_bias_gradient;

 public:
  /**
   * Set up batch normalization.
   * @param decay Controls the momentum of the running mean/standard
   * deviation averages.
   * @param epsilon A small number to avoid division by zero.
   * @param use_global_stats Whether to use global statistics when
   * training.
   */
  batch_normalization(lbann_comm *comm,
                      DataType decay=0.9,
                      DataType epsilon=1e-5,
                      bool use_global_stats = false)
    : regularizer_layer(comm),
      m_decay(decay),
      m_epsilon(epsilon),
      m_use_global_stats(use_global_stats),
      m_mean(nullptr),
      m_var(nullptr),
      m_mean_gradient(nullptr),
      m_var_gradient(nullptr),
      m_scale_gradient(nullptr),
      m_bias_gradient(nullptr) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "batch normalization only supports DATA_PARALLEL");
#ifdef LBANN_DETERMINISTIC
    // Force global computation.
    m_use_global_stats = true;
#endif
  }

  batch_normalization(const batch_normalization& other) :
    regularizer_layer(other),
    m_decay(other.m_decay),
    m_epsilon(other.m_epsilon),
    m_use_global_stats(other.m_use_global_stats),
    m_mean(other.m_mean),
    m_var(other.m_var),
    m_mean_gradient(other.m_mean_gradient),
    m_var_gradient(other.m_var_gradient),
    m_scale_gradient(other.m_scale_gradient),
    m_bias_gradient(other.m_bias_gradient) {

    // Deep copy matrices
    if (m_mean != nullptr)           { m_mean = m_mean->Copy(); }
    if (m_var != nullptr)            { m_var = m_var->Copy(); }
    if (m_mean_gradient != nullptr)  { m_mean_gradient = m_mean_gradient->Copy(); }
    if (m_var_gradient != nullptr)   { m_var_gradient = m_var_gradient->Copy(); }
    if (m_scale_gradient != nullptr) { m_scale_gradient = m_scale_gradient->Copy(); }
    if (m_bias_gradient != nullptr)  { m_bias_gradient = m_bias_gradient->Copy(); }
  }

  batch_normalization& operator=(const batch_normalization& other) {
    regularizer_layer::operator=(other);
    m_decay = other.m_decay;
    m_epsilon = other.m_epsilon;
    m_use_global_stats = other.m_use_global_stats;

    // Deallocate matrices
    deallocate_matrices();

    // Deep copy matrices
    m_mean = other.m_mean;
    m_var = other.m_var;
    m_mean_gradient = other.m_mean_gradient;
    m_var_gradient = other.m_var_gradient;
    m_scale_gradient = other.m_scale_gradient;
    m_bias_gradient = other.m_bias_gradient;
    if (m_mean != nullptr)           { m_mean = m_mean->Copy(); }
    if (m_var != nullptr)            { m_var = m_var->Copy(); }
    if (m_mean_gradient != nullptr)  { m_mean_gradient = m_mean_gradient->Copy(); }
    if (m_var_gradient != nullptr)   { m_var_gradient = m_var_gradient->Copy(); }
    if (m_scale_gradient != nullptr) { m_scale_gradient = m_scale_gradient->Copy(); }
    if (m_bias_gradient != nullptr)  { m_bias_gradient = m_bias_gradient->Copy(); }

    return *this;
  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream ss;
    ss << " batch_normalization; "
       << "decay: " << m_decay
       << " epsilon : " << m_epsilon
       << " data_layout: " << get_data_layout_string(get_data_layout());
    return ss.str();
  }

  virtual ~batch_normalization() override {
    deallocate_matrices();
  }

  batch_normalization* copy() const override { return new batch_normalization(*this); }

  std::string get_type() const override { return "batch normalization"; }

  void setup_matrices(const El::Grid& grid) override {
    regularizer_layer::setup_matrices(grid);
    deallocate_matrices();
    m_mean = new StarMat<Dev>(grid);
    m_var = new StarMat<Dev>(grid);
    m_mean_gradient = new StarMat<Dev>(grid);
    m_var_gradient = new StarMat<Dev>(grid);
    m_scale_gradient = new StarMat<Dev>(grid);
    m_bias_gradient = new StarMat<Dev>(grid);
  }

  data_layout get_data_layout() const override { return T_layout; }

  El::Device get_device_allocation() const override { return Dev; }

  void setup_data() override {
    regularizer_layer::setup_data();
    const auto& output_dims = get_output_dims();
    const auto& num_channels = output_dims[0];

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

  void fp_compute() override {
    if (this->using_gpus()) {
      fp_compute_gpu();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() override {
    if (this->using_gpus()) {
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

};

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
