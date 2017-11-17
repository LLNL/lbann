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
//
// batch_normalization.hpp - Batch normalization layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED

#include "lbann/layers/regularizers/regularizer.hpp"
#ifdef __LIB_CUDNN
#include "lbann/layers/regularizers/batch_normalization_cuda.hpp"
#endif // __LIB_CUDNN
namespace lbann {

/**
 * Batch normalization: normalize layers to zero mean/unit standard deviation.
 * See paper:
 * Sergey Ioffe and Christian Szegedy. "Batch Normalization:
 * Accelerating Deep Network Training by Reducing Internal Covariate
 * Shift." ICML 2015.  This keeps a running mean and standard
 * deviation (with exponential decay) instead of computing it over the
 * data at test time. This approach seems to have become standard.
 * See also:
 * https://cthorey.github.io/backpropagation/
 */
template <data_layout T_layout>
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
  /** View into running means. */
  AbsDistMat *m_running_mean_v;
  /** View into running variance. */
  AbsDistMat *m_running_var_v;

  /** View into scaling terms. */
  AbsDistMat *m_scale_v;
  /** View into bias terms. */
  AbsDistMat *m_bias_v;
  /** Gradient w.r.t. scaling terms. */
  AbsDistMat *m_scale_gradient;
  /** Gradient w.r.t. bias terms. */
  AbsDistMat *m_bias_gradient;

#ifdef __LIB_CUDNN
  /** GPU memory for current minibatch means. */
  std::vector<DataType *> m_mean_d;
  /** GPU memory for current minibatch variances. */
  std::vector<DataType *> m_var_d;
  /** GPU memory for mean gradient. */
  std::vector<DataType *> m_mean_gradient_d;
  /** GPU memory for variance gradient. */
  std::vector<DataType *> m_var_gradient_d;
  /** GPU memory for scaling term gradient. */
  std::vector<DataType *> m_scale_gradient_d;
  /** GPU memory for bias term gradient. */
  std::vector<DataType *> m_bias_gradient_d;
#endif // __LIB_CUDNN

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
                      cudnn::cudnn_manager *cudnn = nullptr,
                      bool use_global_stats = true
                      )
    : regularizer_layer(comm),
      m_decay(decay),
      m_epsilon(epsilon),
      m_use_global_stats(use_global_stats),
      m_mean(nullptr),
      m_var(nullptr),
      m_mean_gradient(nullptr),
      m_var_gradient(nullptr),
      m_running_mean_v(nullptr),
      m_running_var_v(nullptr),
      m_scale_v(nullptr),
      m_bias_v(nullptr),
      m_scale_gradient(nullptr),
      m_bias_gradient(nullptr) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "batch normalization only supports DATA_PARALLEL");
  #ifdef LBANN_SEQUENTIAL_CONSISTENCY
    // Force global computation.
    m_use_global_stats = true;
  #endif
    // Setup the data distribution
    initialize_distributed_matrices();

  #ifdef __LIB_CUDNN
    // Initialize GPU memory if using GPU
    if(cudnn != nullptr) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN

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
    m_running_mean_v(other.m_running_mean_v),
    m_running_var_v(other.m_running_var_v),
    m_scale_v(other.m_scale_v),
    m_bias_v(other.m_bias_v),
    m_scale_gradient(other.m_scale_gradient),
    m_bias_gradient(other.m_bias_gradient) {
    
    // Deep copy matrices
    if (m_mean != nullptr)           { m_mean = m_mean->Copy(); }
    if (m_var != nullptr)            { m_var = m_var->Copy(); }
    if (m_mean_gradient != nullptr)  { m_mean_gradient = m_mean_gradient->Copy(); }
    if (m_var_gradient != nullptr)   { m_var_gradient = m_var_gradient->Copy(); }
    if (m_running_mean_v != nullptr) { m_running_mean_v = m_running_mean_v->Copy(); }
    if (m_running_var_v != nullptr)  { m_running_var_v = m_running_var_v->Copy(); }
    if (m_scale_v != nullptr)        { m_scale_v = m_scale_v->Copy(); }
    if (m_bias_v != nullptr)         { m_bias_v = m_bias_v->Copy(); }
    if (m_scale_gradient != nullptr) { m_scale_gradient = m_scale_gradient->Copy(); }
    if (m_bias_gradient != nullptr)  { m_bias_gradient = m_bias_gradient->Copy(); }

  #ifdef __LIB_CUDNN
    // Copy GPU data
    if (m_cudnn != nullptr) {
      m_mean_d = m_cudnn->copy(other.m_mean_d,
                               m_mean->Height(),
                               m_mean->Width());
      m_var_d = m_cudnn->copy(other.m_var_d,
                              m_var->Height(),
                              m_var->Width());
      m_mean_gradient_d = m_cudnn->copy(other.m_mean_gradient_d,
                                        m_mean_gradient->Height(),
                                        m_mean_gradient->Width());
      m_var_gradient_d = m_cudnn->copy(other.m_var_gradient_d,
                                       m_var_gradient->Height(),
                                       m_var_gradient->Width());
      m_scale_gradient_d = m_cudnn->copy(other.m_scale_gradient_d,
                                         m_scale_gradient->Height(),
                                         m_scale_gradient->Width());
      m_bias_gradient_d = m_cudnn->copy(other.m_bias_gradient_d,
                                        m_bias_gradient->Height(),
                                        m_bias_gradient->Width());
    }
  #endif // __LIB_CUDNN
  }

  batch_normalization& operator=(const batch_normalization& other) {
    regularizer_layer::operator=(other);
    m_decay = other.m_decay;
    m_epsilon = other.m_epsilon;
    m_use_global_stats = other.m_use_global_stats;

    // Copy matrices
  #define COPY_MATRIX(src, dst)                 \
    do {                                        \
      if(src != nullptr && dst != nullptr) {    \
        El::Copy(*src, *dst);                   \
      }                                         \
      if(src != nullptr && dst == nullptr) {    \
        dst = src->Copy();                      \
      }                                         \
      if(src == nullptr && dst != nullptr) {    \
        delete dst;                             \
        dst = nullptr;                          \
      }                                         \
    } while(false)
    COPY_MATRIX(other.m_mean, m_mean);
    COPY_MATRIX(other.m_var, m_var);
    COPY_MATRIX(other.m_mean_gradient, m_mean_gradient);
    COPY_MATRIX(other.m_var_gradient, m_var_gradient);
    COPY_MATRIX(other.m_running_mean_v, m_running_mean_v);
    COPY_MATRIX(other.m_running_var_v, m_running_var_v);
    COPY_MATRIX(other.m_scale_v, m_scale_v);
    COPY_MATRIX(other.m_bias_v, m_bias_v);
    COPY_MATRIX(other.m_scale_gradient, m_scale_gradient);
    COPY_MATRIX(other.m_bias_gradient, m_bias_gradient);
  #undef COPY_MATRIX

  #ifdef __LIB_CUDNN
    // Copy GPU data
    if (m_cudnn != nullptr) {
      m_cudnn->deallocate_on_gpus(m_mean_d);
      m_cudnn->deallocate_on_gpus(m_var_d);
      m_cudnn->deallocate_on_gpus(m_mean_gradient_d);
      m_cudnn->deallocate_on_gpus(m_var_gradient_d);
      m_cudnn->deallocate_on_gpus(m_scale_gradient_d);
      m_cudnn->deallocate_on_gpus(m_bias_gradient_d);
      m_mean_d = m_cudnn->copy(other.m_mean_d,
                               m_mean->Height(),
                               m_mean->Width());
      m_var_d = m_cudnn->copy(other.m_var_d,
                              m_var->Height(),
                              m_var->Width());
      m_mean_gradient_d = m_cudnn->copy(other.m_mean_gradient_d,
                                        m_mean_gradient->Height(),
                                        m_mean_gradient->Width());
      m_var_gradient_d = m_cudnn->copy(other.m_var_gradient_d,
                                       m_var_gradient->Height(),
                                       m_var_gradient->Width());
      m_scale_gradient_d = m_cudnn->copy(other.m_scale_gradient_d,
                                         m_scale_gradient->Height(),
                                         m_scale_gradient->Width());
      m_bias_gradient_d = m_cudnn->copy(other.m_bias_gradient_d,
                                        m_bias_gradient->Height(),
                                        m_bias_gradient->Width());
    }
  #endif // __LIB_CUDNN

    return *this;
  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream ss;
    ss << " batch_normalization; "
       << "decay: " << m_decay
       << "epsilon : " << m_epsilon
       << "data_layout: " << get_data_layout_string(get_data_layout());
    return ss.str();
  }

  ~batch_normalization() {
  #ifdef __LIB_CUDNN
    // Deallocate GPU memory
    if (m_cudnn != nullptr) {
      this->m_cudnn->deallocate_on_gpus(m_mean_d);
      this->m_cudnn->deallocate_on_gpus(m_var_d);
      this->m_cudnn->deallocate_on_gpus(m_mean_gradient_d);
      this->m_cudnn->deallocate_on_gpus(m_var_gradient_d);
      this->m_cudnn->deallocate_on_gpus(m_scale_gradient_d);
      this->m_cudnn->deallocate_on_gpus(m_bias_gradient_d);
    }
  #endif // #ifdef __LIB_CUDNN

    // Deallocate matrices
    if (m_mean != nullptr)           { delete m_mean; }
    if (m_var != nullptr)            { delete m_var; }
    if (m_mean_gradient != nullptr)  { delete m_mean_gradient; }
    if (m_var_gradient != nullptr)   { delete m_var_gradient; }
    if (m_running_mean_v != nullptr) { delete m_running_mean_v; }
    if (m_running_var_v != nullptr)  { delete m_running_var_v; }
    if (m_scale_v != nullptr)        { delete m_scale_v; }
    if (m_bias_v != nullptr)         { delete m_bias_v; }
    if (m_scale_gradient != nullptr) { delete m_scale_gradient; }
    if (m_bias_gradient != nullptr)  { delete m_bias_gradient; }
  }

  batch_normalization* copy() const override { return new batch_normalization(*this); }

  std::string get_type() const override { return "batch normalization"; }

  void initialize_distributed_matrices() {
    regularizer_layer::initialize_distributed_matrices<T_layout>();
    m_mean = new StarMat(this->m_comm->get_model_grid());
    m_var = new StarMat(this->m_comm->get_model_grid());
    m_mean_gradient = new StarMat(this->m_comm->get_model_grid());
    m_var_gradient = new StarMat(this->m_comm->get_model_grid());
    m_running_mean_v = new StarMat(this->m_comm->get_model_grid());
    m_running_var_v = new StarMat(this->m_comm->get_model_grid());
    m_scale_v = new StarMat(this->m_comm->get_model_grid());
    m_bias_v = new StarMat(this->m_comm->get_model_grid());
    m_scale_gradient = new StarMat(this->m_comm->get_model_grid());
    m_bias_gradient = new StarMat(this->m_comm->get_model_grid());
  }

  data_layout get_data_layout() const override { return T_layout; }

  void setup_data() override {
    regularizer_layer::setup_data();

    // Initialize default weights if none are provided
    if (this->m_weights.size() > 4) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "attempted to setup " << m_name << " with an invalid number of weights";
      throw lbann_exception(err.str());
    }
    this->m_weights.resize(4, nullptr);
    if (this->m_weights[0] == nullptr) {
      this->m_weights[0] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[0]->set_name(this->m_name + "_scale");
      this->m_weights[0]->set_initializer(new constant_initializer(this->m_comm, DataType(1)));
      this->m_weights[0]->set_optimizer(m_neural_network_model->create_optimizer());
      this->m_neural_network_model->add_weights(this->m_weights[0]);
    }
    if (this->m_weights[1] == nullptr) {
      this->m_weights[1] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[1]->set_name(this->m_name + "_bias");
      this->m_weights[1]->set_initializer(new constant_initializer(this->m_comm, DataType(0)));
      this->m_weights[1]->set_optimizer(m_neural_network_model->create_optimizer());
      this->m_neural_network_model->add_weights(this->m_weights[1]);
    }
    if (this->m_weights[2] == nullptr) {
      this->m_weights[2] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[2]->set_name(this->m_name + "_running_mean");
      this->m_weights[2]->set_initializer(new constant_initializer(this->m_comm, DataType(0)));
      this->m_neural_network_model->add_weights(this->m_weights[2]);
    }
    if (this->m_weights[3] == nullptr) {
      this->m_weights[3] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[3]->set_name(this->m_name + "_running_variance");
      this->m_weights[3]->set_initializer(new constant_initializer(this->m_comm, DataType(1)));
      this->m_neural_network_model->add_weights(this->m_weights[3]);
    }

    // Setup weights
    this->m_weights[0]->setup(this->m_neuron_dims[0], 1, El::STAR, El::STAR);
    this->m_weights[1]->setup(this->m_neuron_dims[0], 1, El::STAR, El::STAR);
    this->m_weights[2]->setup(this->m_neuron_dims[0], 1, El::STAR, El::STAR);
    this->m_weights[3]->setup(this->m_neuron_dims[0], 1, El::STAR, El::STAR);
    
    // Initialize matrices
    El::Zeros(*m_mean, this->m_neuron_dims[0], 1);
    El::Zeros(*m_var, this->m_neuron_dims[0], 1);
    El::Zeros(*m_mean_gradient, this->m_neuron_dims[0], 1);
    El::Zeros(*m_var_gradient, this->m_neuron_dims[0], 1);
    El::Zeros(*m_scale_gradient, this->m_neuron_dims[0], 1);
    El::Zeros(*m_bias_gradient, this->m_neuron_dims[0], 1);

  }

  void setup_views() override {
    regularizer_layer::setup_views();
    this->m_weights[0]->get_values_view(*m_scale_v);
    this->m_weights[1]->get_values_view(*m_bias_v);
    this->m_weights[2]->get_values_view(*m_running_mean_v);
    this->m_weights[3]->get_values_view(*m_running_var_v);
  }

  void setup_gpu() override {
    regularizer_layer::setup_gpu();
  #ifndef __LIB_CUDNN
    throw lbann_exception("convolution_layer: cuDNN not detected");
  #else

    // Allocate GPU memory
    this->m_cudnn->allocate_on_gpus(m_mean_d,
                                    m_mean->Height(),
                                    m_mean->Width());
    this->m_cudnn->allocate_on_gpus(m_var_d,
                                    m_var->Height(),
                                    m_var->Width());
    this->m_cudnn->allocate_on_gpus(m_scale_gradient_d,
                                    m_scale_gradient->Height(),
                                    m_scale_gradient->Width());
    this->m_cudnn->allocate_on_gpus(m_bias_gradient_d,
                                    m_bias_gradient->Height(),
                                    m_bias_gradient->Width());
    this->m_cudnn->allocate_on_gpus(m_mean_gradient_d,
                                    m_mean_gradient->Height(),
                                    m_mean_gradient->Width());
    this->m_cudnn->allocate_on_gpus(m_var_gradient_d,
                                    m_var_gradient->Height(),
                                    m_var_gradient->Width());

  #endif // __LIB_CUDNN

  }

  void fp_set_std_matrix_view() override {
    regularizer_layer::fp_set_std_matrix_view();
    this->m_weights[0]->get_values_view(*m_scale_v);
    this->m_weights[1]->get_values_view(*m_bias_v);
    this->m_weights[2]->get_values_view(*m_running_mean_v);
    this->m_weights[3]->get_values_view(*m_running_var_v);
  }

  void bp_set_std_matrix_view() override {
    regularizer_layer::bp_set_std_matrix_view();
    this->m_weights[0]->get_values_view(*m_scale_v);
    this->m_weights[1]->get_values_view(*m_bias_v);
    this->m_weights[2]->get_values_view(*m_running_mean_v);
    this->m_weights[3]->get_values_view(*m_running_var_v);
  }

  void fp_compute() override {
    if(this->m_using_gpus) {
      fp_compute_gpu();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
      bp_compute_gpu();
    } else {
      bp_compute_cpu();
    }
  }

  void fp_compute_gpu() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("batch_normalization_layer: cuDNN not detected");
  #else

    // Check execution mode
    const bool is_training = this->get_execution_mode() == execution_mode::training;

    // Matrix parameters
    const int num_gpus = this->m_cudnn->get_num_gpus();
    const int height = this->m_prev_activations->Height();
    const int width = this->m_prev_activations->Width();
    const int local_width = this->m_prev_activations->LocalWidth();
    const int num_channels = this->m_neuron_dims[0];
    const int channel_size = this->m_num_neurons / num_channels;

    // Compute statistics
    if(is_training) {

      // Get GPU objects
      std::vector<DataType*> running_mean_d = m_weights[2]->get_values_gpu();
      std::vector<DataType*> running_var_d = m_weights[3]->get_values_gpu();

      // Compute sums and sums of squares on GPUs
      for(int i=0; i<num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        const int col_start = std::min(i * this->m_mini_batch_size_per_gpu, local_width);
        const int col_end = std::min((i+1) * this->m_mini_batch_size_per_gpu, local_width);
        const int current_width = col_end - col_start;
        batch_normalization_cuda
          ::channel_sums_and_sqsums<DataType>(height,
                                              current_width,
                                              num_channels,
                                              this->m_prev_activations_d[i],
                                              m_mean_d[i],
                                              m_var_d[i],
                                              this->m_cudnn->get_stream(i));
      }
      this->m_cudnn->allreduce(m_mean_d, num_channels, 1);
      this->m_cudnn->allreduce(m_var_d, num_channels, 1);

      // Accumulate sums and sums of squares across nodes if needed
      int samples_per_sum;
      if (m_use_global_stats) {
        this->m_cudnn->copy_from_gpu(0, m_mean->Matrix(), m_mean_d[0]);
        this->m_cudnn->copy_from_gpu(0, m_var->Matrix(), m_var_d[0]);
        this->m_cudnn->synchronize();
        El::AllReduce(*m_mean, m_mean->RedundantComm(), El::mpi::SUM);
        El::AllReduce(*m_var, m_var->RedundantComm(), El::mpi::SUM);
        this->m_cudnn->broadcast_to_gpus(m_mean_d, m_mean->LockedMatrix());
        this->m_cudnn->broadcast_to_gpus(m_var_d, m_var->LockedMatrix());
        samples_per_sum = channel_size * width;
      } else {
        samples_per_sum = channel_size * local_width;
      }

      // Compute minibatch statistics and running statistics
      for(int i=0; i<num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        batch_normalization_cuda
          ::sums_to_statistics<DataType>(num_channels,
                                         samples_per_sum,
                                         m_decay,
                                         m_mean_d[i],
                                         m_var_d[i],
                                         running_mean_d[i],
                                         running_var_d[i],
                                         this->m_cudnn->get_stream(i));
      }
      
    }

    // Get GPU objects
    std::vector<DataType*> scale_d = m_weights[0]->get_values_gpu();
    std::vector<DataType*> bias_d = m_weights[1]->get_values_gpu();
    std::vector<DataType*> mean_d = is_training ? m_mean_d : m_weights[2]->get_values_gpu();
    std::vector<DataType*> var_d = is_training ? m_var_d : m_weights[3]->get_values_gpu();

    // Perform batch normalization with each GPU
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      const int col_start = std::min(i * this->m_mini_batch_size_per_gpu, local_width);
      const int col_end = std::min((i+1) * this->m_mini_batch_size_per_gpu, local_width);
      const int current_width = col_end - col_start;
      batch_normalization_cuda
        ::batch_normalization<DataType>(height,
                                        current_width,
                                        num_channels,
                                        this->m_prev_activations_d[i],
                                        mean_d[i],
                                        var_d[i],
                                        m_epsilon,
                                        scale_d[i],
                                        bias_d[i],
                                        this->m_activations_d[i],
                                        this->m_cudnn->get_stream(i));
    }

  #endif // __LIB_CUDNN
  }

  void bp_compute_gpu() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("batch_normalization_layer: cuDNN not detected");
  #else

    // Check execution mode
    const bool is_training = this->get_execution_mode() == execution_mode::training;

    // GPU objects
    const int num_gpus = this->m_cudnn->get_num_gpus();
    std::vector<DataType*> scale_d = m_weights[0]->get_values_gpu();
    std::vector<DataType*> mean_d = is_training ? m_mean_d : m_weights[2]->get_values_gpu();
    std::vector<DataType*> var_d = is_training ? m_var_d : m_weights[3]->get_values_gpu();

    // Matrix parameters
    const int height = this->m_prev_activations->Height();
    const int width = this->m_prev_activations->Width();
    const int local_width = this->m_prev_activations->LocalWidth();
    const int num_channels = this->m_neuron_dims[0];

    // Compute local gradient contributions
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      const int col_start = std::min(i * this->m_mini_batch_size_per_gpu, local_width);
      const int col_end = std::min((i+1) * this->m_mini_batch_size_per_gpu, local_width);
      const int current_width = col_end - col_start;
      batch_normalization_cuda
        ::batch_normalization_backprop1<DataType>(height,
                                                  current_width,
                                                  num_channels,
                                                  this->m_prev_activations_d[i],
                                                  this->m_prev_error_signal_d[i],
                                                  mean_d[i],
                                                  var_d[i],
                                                  m_epsilon,
                                                  scale_d[i],
                                                  m_scale_gradient_d[i],
                                                  m_bias_gradient_d[i],
                                                  m_mean_gradient_d[i],
                                                  m_var_gradient_d[i],
                                                  this->m_cudnn->get_stream(i));
    }
    this->m_cudnn->allreduce(m_mean_gradient_d, num_channels, 1);
    this->m_cudnn->allreduce(m_var_gradient_d, num_channels, 1);

    // Accumulate gradients
    if(is_training) {
      if(m_use_global_stats) {
        this->m_cudnn->copy_from_gpu(0, m_mean_gradient->Matrix(), m_mean_gradient_d[0]);
        this->m_cudnn->copy_from_gpu(0, m_var_gradient->Matrix(), m_var_gradient_d[0]);
        this->m_cudnn->synchronize();
        El::AllReduce(*m_mean_gradient, m_mean_gradient->RedundantComm(), El::mpi::SUM);
        El::AllReduce(*m_var_gradient, m_var_gradient->RedundantComm(), El::mpi::SUM);
        this->m_cudnn->broadcast_to_gpus(m_mean_gradient_d, m_mean_gradient->LockedMatrix());
        this->m_cudnn->broadcast_to_gpus(m_var_gradient_d, m_var_gradient->LockedMatrix());
      }
    } else {
      m_cudnn->clear_on_gpus(m_mean_gradient_d, num_channels, 1);
      m_cudnn->clear_on_gpus(m_var_gradient_d, num_channels, 1);
    }
    optimizer* scale_optimizer = m_weights[0]->get_optimizer();
    if (scale_optimizer != nullptr) {
      for(int i=0; i<num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        CHECK_CUBLAS(cublas::scal(this->m_cudnn->get_cublas_handle(i),
                                  num_channels,
                                  DataType(1) / this->m_neural_network_model->get_effective_mini_batch_size(),
                                  m_scale_gradient_d[i], 1));
      }
      scale_optimizer->gpu_allreduce_and_add_to_gradient(m_scale_gradient_d);
    }
    optimizer* bias_optimizer = m_weights[1]->get_optimizer();
    if (bias_optimizer != nullptr) {
      for(int i=0; i<num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        CHECK_CUBLAS(cublas::scal(this->m_cudnn->get_cublas_handle(i),
                                  num_channels,
                                  DataType(1) / this->m_neural_network_model->get_effective_mini_batch_size(),
                                  m_bias_gradient_d[i], 1));
      }
      bias_optimizer->gpu_allreduce_and_add_to_gradient(m_bias_gradient_d);
    }

    // Compute error signal
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      const int col_start = std::min(i * this->m_mini_batch_size_per_gpu, local_width);
      const int col_end = std::min((i+1) * this->m_mini_batch_size_per_gpu, local_width);
      const int current_width = col_end - col_start;
      batch_normalization_cuda
        ::batch_normalization_backprop2<DataType>(height,
                                                  current_width,
                                                  width,
                                                  num_channels,
                                                  this->m_prev_activations_d[i],
                                                  this->m_prev_error_signal_d[i],
                                                  m_mean_d[i],
                                                  m_var_d[i],
                                                  m_epsilon,
                                                  scale_d[i],
                                                  m_mean_gradient_d[i],
                                                  m_var_gradient_d[i],
                                                  this->m_error_signal_d[i],
                                                  this->m_cudnn->get_stream(i));
    }

  #endif // __LIB_CUDNN
  }

  void fp_compute_cpu() {
    
    // Check execution mode
    const bool is_training = this->get_execution_mode() == execution_mode::training;

    // Local matrices
    const Mat& prev_activations_local = this->m_prev_activations->LockedMatrix();
    Mat& activations_local = this->m_activations_v->Matrix();

    // Matrix parameters
    const int width = this->m_prev_activations->Width();
    const El::Int local_width = this->m_prev_activations->LocalWidth();
    const int num_channels = this->m_neuron_dims[0];
    const int channel_size = this->m_num_neurons / num_channels;

    // Compute statistics
    if(is_training) {

      // Local matrices
      Mat& mean_local = m_mean->Matrix();
      Mat& var_local = m_var->Matrix();
      const Mat& running_mean_local = m_running_mean_v->LockedMatrix();
      const Mat& running_var_local = m_running_var_v->LockedMatrix();
      Mat& new_running_mean_local = m_mean_gradient->Matrix();
      Mat& new_running_var_local = m_var_gradient->Matrix();

      // Compute sums and sums of squares
      #pragma omp parallel for
      for(int channel = 0; channel < num_channels; ++channel) {
        DataType sum = DataType(0);
        DataType sqsum = DataType(0);
        const El::Int row_start = channel * channel_size;
        const El::Int row_end = (channel+1) * channel_size;
        for(El::Int col = 0; col < local_width; ++col) {
          for(El::Int row = row_start; row < row_end; ++row) {
            const DataType x = prev_activations_local(row, col);
            sum += x;
            sqsum += x * x;
          }
        }
        mean_local(channel, 0) = sum;
        var_local(channel, 0) = sqsum;
      }
      DataType num_samples;
      if (m_use_global_stats) {
        El::AllReduce(*m_mean, m_mean->RedundantComm(), El::mpi::SUM);
        El::AllReduce(*m_var, m_var->RedundantComm(), El::mpi::SUM);
        num_samples = channel_size * width;
      } else {
        num_samples = channel_size * local_width;
      }

      // Compute minibatch statistics
      // Note: new_running_mean and new_running_var are stored in
      // m_mean_gradient and m_var_gradient
      #pragma omp parallel for
      for(int channel = 0; channel < num_channels; ++channel) {
        const DataType mean = mean_local(channel, 0) / num_samples;
        const DataType sqmean = var_local(channel, 0) / num_samples;
        const DataType var = num_samples / (num_samples - DataType(1)) * std::max(sqmean - mean * mean, DataType(0));
        const DataType old_running_mean = running_mean_local(channel, 0);
        const DataType old_running_var = running_var_local(channel, 0);
        const DataType new_running_mean = m_decay * old_running_mean + (DataType(1) - m_decay) * mean;
        const DataType new_running_var = m_decay * old_running_var + (DataType(1) - m_decay) * var;
        mean_local(channel, 0) = mean;
        var_local(channel, 0) = var;
        new_running_mean_local(channel, 0) = new_running_mean;
        new_running_var_local(channel, 0) = new_running_var;
      }
      m_weights[2]->set_values(*m_mean_gradient);
      m_weights[3]->set_values(*m_var_gradient);
      
    }

    // Local matrices
    const Mat& mean_local = is_training ? m_mean->LockedMatrix() : m_running_mean_v->LockedMatrix();
    const Mat& var_local = is_training ? m_var->LockedMatrix() : m_running_var_v->LockedMatrix();
    const Mat& scale_local = m_scale_v->LockedMatrix();
    const Mat& bias_local = m_bias_v->LockedMatrix();

    // Iterate through channels
    #pragma omp parallel for
    for(int channel = 0; channel < num_channels; ++channel) {

      // Get channel parameters
      const DataType mean = mean_local(channel, 0);
      const DataType var = var_local(channel, 0);
      const DataType inv_stdev = 1 / std::sqrt(var + m_epsilon);
      const DataType scale = scale_local(channel, 0);
      const DataType bias = bias_local(channel, 0);

      // Apply batch normalization to inputs in channel
      const El::Int row_start = channel * channel_size;
      const El::Int row_end = (channel+1) * channel_size;
      for(El::Int col = 0; col < local_width; ++col) {
        for(El::Int row = row_start; row < row_end; ++row) {
          const DataType x = prev_activations_local(row, col);
          const DataType xhat = (x - mean) * inv_stdev;
          const DataType y = scale * xhat + bias;
          activations_local(row, col) = y;
        }
      }

    }

  }  

  void bp_compute_cpu() {
    
    // Check execution mode
    const bool is_training = this->get_execution_mode() == execution_mode::training;

    // Local matrices
    const Mat& prev_activations_local = this->m_prev_activations->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal->LockedMatrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();
    const Mat& mean_local = is_training ? m_mean->LockedMatrix() : m_running_mean_v->LockedMatrix();
    const Mat& var_local = is_training ? m_var->LockedMatrix() : m_running_var_v->LockedMatrix();
    const Mat& scale_local = m_scale_v->LockedMatrix();
    Mat& mean_gradient_local = m_mean_gradient->Matrix();
    Mat& var_gradient_local = m_var_gradient->Matrix();
    Mat& scale_gradient_local = m_scale_gradient->Matrix();
    Mat& bias_gradient_local = m_bias_gradient->Matrix();
    
    // Matrix parameters
    const int width = this->m_prev_activations->Width();
    const El::Int local_width = this->m_prev_activations->LocalWidth();
    const int num_channels = this->m_neuron_dims[0];
    const int channel_size = this->m_num_neurons / num_channels;

    // Compute local gradients
    #pragma omp parallel for
    for(int channel = 0; channel < num_channels; ++channel) {

      // Initialize channel parameters and gradients
      const DataType mean = mean_local(channel, 0);
      const DataType var = var_local(channel, 0);
      const DataType scale = scale_local(channel, 0);
      const DataType inv_stdev = 1 / std::sqrt(var + m_epsilon);
      const DataType dvar_factor = inv_stdev * inv_stdev * inv_stdev / 2;
      DataType dmean = DataType(0);
      DataType dvar = DataType(0);
      DataType dscale = DataType(0);
      DataType dbias = DataType(0);

      // Compute gradient contributions from local entries
      const El::Int row_start = channel * channel_size;
      const El::Int row_end = (channel+1) * channel_size;
      for(El::Int col = 0; col < local_width; ++col) {
        for(El::Int row = row_start; row < row_end; ++row) {
          const DataType x = prev_activations_local(row, col);
          const DataType xhat = (x - mean) * inv_stdev;
          const DataType dy = prev_error_signal_local(row, col);
          dscale += dy * xhat;
          dbias += dy;
          const DataType dxhat = dy * scale;
          dmean += - dxhat * inv_stdev;
          dvar += - dxhat * (x - mean) * dvar_factor;
        }
      }
      mean_gradient_local(channel, 0) = dmean;
      var_gradient_local(channel, 0) = dvar;
      scale_gradient_local(channel, 0) = dscale;
      bias_gradient_local(channel, 0) = dbias;

    }

    // Accumulate gradients
    if (is_training) {
      if (m_use_global_stats) {
        El::AllReduce(*m_mean_gradient,
                      m_mean_gradient->RedundantComm(),
                      El::mpi::SUM);
        El::AllReduce(*m_var_gradient,
                      m_var_gradient->RedundantComm(),
                      El::mpi::SUM);
      }
    } else {
      El::Zero(*m_mean_gradient);
      El::Zero(*m_var_gradient);
    }
    optimizer* scale_optimizer = m_weights[0]->get_optimizer();
    if (scale_optimizer != nullptr) {
      El::Scale(DataType(1) / this->m_neural_network_model->get_effective_mini_batch_size(),
                *m_scale_gradient);
      scale_optimizer->allreduce_and_add_to_gradient(*m_scale_gradient);
    }
    optimizer* bias_optimizer = m_weights[1]->get_optimizer();
    if (bias_optimizer != nullptr) {
      El::Scale(DataType(1) / this->m_neural_network_model->get_effective_mini_batch_size(),
                *m_bias_gradient);
      bias_optimizer->allreduce_and_add_to_gradient(*m_bias_gradient);
    }
    
    // Compute error signal
    #pragma omp parallel for
    for(int channel = 0; channel < num_channels; ++channel) {

      // Initialize channel parameters and gradients
      const DataType mean = mean_local(channel, 0);
      const DataType var = var_local(channel, 0);
      const DataType scale = scale_local(channel, 0);
      const DataType dmean = mean_gradient_local(channel, 0);
      const DataType dvar = var_gradient_local(channel, 0);

      // Compute useful constants
      const DataType num_samples = width * channel_size;
      const DataType inv_stdev = 1 / std::sqrt(var + m_epsilon);
      const DataType dmean_term = dmean / num_samples;
      const DataType dvar_term = dvar * 2 / (num_samples - DataType(1));

      // Compute error signal for current channel
      const El::Int row_start = channel * channel_size;
      const El::Int row_end = (channel+1) * channel_size;
      for(El::Int col = 0; col < local_width; ++col) {
        for(El::Int row = row_start; row < row_end; ++row) {
          const DataType x = prev_activations_local(row, col);
          const DataType dy = prev_error_signal_local(row, col);
          const DataType dxhat = dy * scale;
          DataType dx = dxhat * inv_stdev;
          dx += dmean_term;
          dx += dvar_term * (x - mean);
          error_signal_local(row, col) = dx;
        }
      }

    }

  }

};

} // namespace lbann

#endif  // LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
