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
// lbann_batch_normalization .cpp .hpp - Batch normalization implementation
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED

#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/utils/statistics.hpp"

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
  /** Batch normalization parameters. */
  AbsDistMat *m_parameters;
  /** Batch normalization parameter gradients. */
  AbsDistMat *m_parameters_gradient;

  /** Decay rate for the running statistics. */
  DataType m_decay;
  /** View into statistics for current minibatch. */
  AbsDistMat *m_statistics_v;
  /** View into current minibatch means. */
  AbsDistMat *m_mean_v;
  /** View into current minibatch standard deviations. */
  AbsDistMat *m_stdev_v;
  /** View into running means. */
  AbsDistMat *m_running_mean_v;
  /** View into running standard deviations. */
  AbsDistMat *m_running_stdev_v;
  /** View into gradient w.r.t. means. */
  AbsDistMat *m_mean_gradient_v;
  /** View into gradient w.r.t. standard deviations. */
  AbsDistMat *m_stdev_gradient_v;

  /** Initial value for scaling term. */
  DataType m_scale_init;
  /** Initial value for bias term. */
  DataType m_bias_init;
  /** View into scaling term. */
  AbsDistMat *m_scale_v;
  /** View into bias term. */
  AbsDistMat *m_bias_v;
  /** View into gradient w.r.t. scaling term. */
  AbsDistMat *m_scale_gradient_v;
  /** View into gradient w.r.t. bias term. */
  AbsDistMat *m_bias_gradient_v;
  /** Optimizer for learning scaling term. */
  optimizer *m_scale_optimizer;
  /** Optimizer for learning bias term. */
  optimizer *m_bias_optimizer;

  /** Small number to avoid division by zero. */
  DataType m_epsilon;
  /** Whether to use running statistics when training. */
  bool m_use_global_stats;

 public:
  /**
   * Set up batch normalization.
   * @param decay Controls the momentum of the running mean/standard
   * deviation averages.
   * @param scale The initial value for scaling parameter
   * \f$\gamma$\f$. The paper recommends 1.0 as a starting point, but
   * other papers have had better results with a smaller value
   * (e.g. 0.1).
   * @param bias The initial value for bias parameter
   * \f$\beta\f$. This should almost always stay at zero.
   */
  batch_normalization(int index,
                      lbann_comm *comm,
                      int mini_batch_size,
                      DataType decay=0.9,
                      DataType scale_init=1.0,
                      DataType bias_init=0.0,
                      DataType epsilon=1e-5,
                      bool use_global_stats=false)
    : regularizer_layer(index, comm, mini_batch_size),
      m_decay(decay),
      m_scale_init(scale_init),
      m_bias_init(bias_init),
      m_epsilon(epsilon),
      m_use_global_stats(use_global_stats) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "batch normalization only supports DATA_PARALLEL");
    // Setup the data distribution
    initialize_distributed_matrices();
  }

  batch_normalization(const batch_normalization& other) :
    regularizer_layer(other),
    m_decay(other.m_decay),
    m_scale_init(other.m_scale_init),
    m_bias_init(other.m_bias_init),
    m_epsilon(other.m_epsilon),
    m_use_global_stats(other.m_use_global_stats) {

    // Copy matrices
    m_parameters = other.m_parameters->Copy();
    m_parameters_gradient = other.m_parameters_gradient->Copy();
    m_mean_v = other.m_mean_v->Copy();
    m_stdev_v = other.m_stdev_v->Copy();
    m_running_mean_v = other.m_running_mean_v->Copy();
    m_running_stdev_v = other.m_running_stdev_v->Copy();
    m_mean_gradient_v = other.m_mean_gradient_v->Copy();
    m_stdev_gradient_v = other.m_stdev_gradient_v->Copy();
    m_scale_v = other.m_scale_v->Copy();
    m_bias_v = other.m_bias_v->Copy();
    m_scale_gradient_v = other.m_scale_gradient_v->Copy();
    m_bias_gradient_v = other.m_bias_gradient_v->Copy();

    // Setup matrix views
    setup_views();

    // Copy optimizers
    if(other.m_scale_optimizer) {
      m_scale_optimizer = other.m_scale_optimizer->copy();
      if(m_scale_optimizer->get_parameters()) {
        m_scale_optimizer->set_parameters(m_scale_v);
      }
    } else {
      m_scale_optimizer = NULL;
    }
    if(other.m_bias_optimizer) {
      m_bias_optimizer = other.m_bias_optimizer->copy();
      if(m_bias_optimizer->get_parameters()) {
        m_bias_optimizer->set_parameters(m_bias_v);
      }
    } else {
      m_scale_optimizer = NULL;
    }

  }

  batch_normalization& operator=(const batch_normalization& other) {
    regularizer_layer::operator=(other);

    // Deallocate matrices
    if(m_parameters)          delete m_parameters;
    if(m_parameters_gradient) delete m_parameters_gradient;
    if(m_mean_v)              delete m_mean_v;
    if(m_stdev_v)             delete m_stdev_v;
    if(m_running_mean_v)      delete m_running_mean_v;
    if(m_running_stdev_v)     delete m_running_stdev_v;
    if(m_mean_gradient_v)     delete m_mean_gradient_v;
    if(m_stdev_gradient_v)    delete m_stdev_gradient_v;
    if(m_scale_v)             delete m_scale_v;
    if(m_bias_v)              delete m_bias_v;
    if(m_scale_gradient_v)    delete m_scale_gradient_v;
    if(m_bias_gradient_v)     delete m_bias_gradient_v;
    if(m_scale_optimizer)     delete m_scale_optimizer;
    if(m_bias_optimizer)      delete m_bias_optimizer;

    // Copy POD members
    m_decay = other.m_decay;
    m_scale_init = other.m_scale_init;
    m_bias_init = other.m_bias_init;
    m_epsilon = other.m_epsilon;
    m_use_global_stats = other.m_use_global_stats;

    // Copy matrices
    m_parameters = other.m_parameters->Copy();
    m_parameters_gradient = other.m_parameters_gradient->Copy();
    m_mean_v = other.m_mean_v->Copy();
    m_stdev_v = other.m_stdev_v->Copy();
    m_running_mean_v = other.m_running_mean_v->Copy();
    m_running_stdev_v = other.m_running_stdev_v->Copy();
    m_mean_gradient_v = other.m_mean_gradient_v->Copy();
    m_stdev_gradient_v = other.m_stdev_gradient_v->Copy();
    m_scale_v = other.m_scale_v->Copy();
    m_bias_v = other.m_bias_v->Copy();
    m_scale_gradient_v = other.m_scale_gradient_v->Copy();
    m_bias_gradient_v = other.m_bias_gradient_v->Copy();

    // Setup matrix view
    setup_views();

    // Copy optimizers
    if(other.m_scale_optimizer) {
      m_scale_optimizer = other.m_scale_optimizer->copy();
      if(m_scale_optimizer->get_parameters()) {
        m_scale_optimizer->set_parameters(m_scale_v);
      }
    } else {
      m_scale_optimizer = NULL;
    }
    if(other.m_bias_optimizer) {
      m_bias_optimizer = other.m_bias_optimizer->copy();
      if(m_bias_optimizer->get_parameters()) {
        m_bias_optimizer->set_parameters(m_bias_v);
      }
    } else {
      m_scale_optimizer = NULL;
    }

    // Return copy
    return *this;

  }

  ~batch_normalization() {
    if(m_parameters)          delete m_parameters;
    if(m_parameters_gradient) delete m_parameters_gradient;
    if(m_mean_v)              delete m_mean_v;
    if(m_stdev_v)             delete m_stdev_v;
    if(m_running_mean_v)      delete m_running_mean_v;
    if(m_running_stdev_v)     delete m_running_stdev_v;
    if(m_mean_gradient_v)     delete m_mean_gradient_v;
    if(m_stdev_gradient_v)    delete m_stdev_gradient_v;
    if(m_scale_v)             delete m_scale_v;
    if(m_bias_v)              delete m_bias_v;
    if(m_scale_gradient_v)    delete m_scale_gradient_v;
    if(m_bias_gradient_v)     delete m_bias_gradient_v;
    if(m_scale_optimizer)     delete m_scale_optimizer;
    if(m_bias_optimizer)      delete m_bias_optimizer;
  }

  batch_normalization* copy() const { return new batch_normalization(*this); }

  std::string get_name() const { return "batch normalization"; }

  void initialize_distributed_matrices() {
    regularizer_layer::initialize_distributed_matrices<T_layout>();
    m_parameters = new StarMat(this->m_comm->get_model_grid());
    m_parameters_gradient = new StarMat(this->m_comm->get_model_grid());
    m_statistics_v = new StarMat(this->m_comm->get_model_grid());
    m_mean_v = new StarMat(this->m_comm->get_model_grid());
    m_stdev_v = new StarMat(this->m_comm->get_model_grid());
    m_running_mean_v = new StarMat(this->m_comm->get_model_grid());
    m_running_stdev_v = new StarMat(this->m_comm->get_model_grid());
    m_mean_gradient_v = new StarMat(this->m_comm->get_model_grid());
    m_stdev_gradient_v = new StarMat(this->m_comm->get_model_grid());
    m_scale_v = new StarMat(this->m_comm->get_model_grid());
    m_bias_v = new StarMat(this->m_comm->get_model_grid());
    m_scale_gradient_v = new StarMat(this->m_comm->get_model_grid());
    m_bias_gradient_v = new StarMat(this->m_comm->get_model_grid());
  }

  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_data() {
    regularizer_layer::setup_data();

    // Allocate memory
    m_parameters->Resize(this->m_neuron_dims[0], 6);
    m_parameters_gradient->Resize(this->m_neuron_dims[0], 4);

    // Initialize statistics
    El::View(*m_running_mean_v, *m_parameters, El::ALL, El::IR(2));
    El::View(*m_running_mean_v, *m_parameters, El::ALL, El::IR(3));
    El::Zero(*m_running_mean_v);
    El::Zero(*m_running_stdev_v);

    // Initialize scaling and bias terms
    El::View(*m_scale_v, *m_parameters, El::ALL, El::IR(4));
    El::View(*m_bias_v, *m_parameters, El::ALL, El::IR(5));
    El::Fill(*m_scale_v, m_scale_init);
    El::Fill(*m_bias_v, m_bias_init);

    // Initialize optimizers
    m_scale_optimizer = this->get_neural_network_model()->create_optimizer();
    m_bias_optimizer = this->get_neural_network_model()->create_optimizer();
    m_scale_optimizer->setup(m_scale_v);
    m_bias_optimizer->setup(m_bias_v);

  }

  void setup_views() {
    regularizer_layer::setup_views();

    // Initialize views into parameters
    El::View(*m_statistics_v, *m_parameters, El::ALL, El::IR(0,2));
    El::View(*m_mean_v, *m_statistics_v, El::ALL, El::IR(0));
    El::View(*m_stdev_v, *m_statistics_v, El::ALL, El::IR(1));
    El::View(*m_running_mean_v, *m_statistics_v, El::ALL, El::IR(2));
    El::View(*m_running_stdev_v, *m_statistics_v, El::ALL, El::IR(3));
    El::View(*m_scale_v, *m_parameters, El::ALL, El::IR(4));
    El::View(*m_bias_v, *m_parameters, El::ALL, El::IR(5));

    // Initialize views into parameter gradients
    El::View(*m_mean_gradient_v, *m_parameters_gradient, El::ALL, El::IR(0));
    El::View(*m_stdev_gradient_v, *m_parameters_gradient, El::ALL, El::IR(1));
    El::View(*m_scale_gradient_v, *m_parameters_gradient, El::ALL, El::IR(2));
    El::View(*m_bias_gradient_v, *m_parameters_gradient, El::ALL, El::IR(3));

  }

  void fp_compute() {
    if(this->m_using_gpus) {
      fp_compute_cudnn();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() {
    if(this->m_using_gpus) {
      bp_compute_cudnn();
    } else {
      bp_compute_cpu();
    }
  }

  void fp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("batch_normalization_layer: cuDNN not detected");
  #else
    throw lbann_exception("batch_normalization_layer: no cuDNN implementation");
  #endif // __LIB_CUDNN
  }

  void bp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("batch_normalization_layer: cuDNN not detected");
  #else
    throw lbann_exception("batch_normalization_layer: no cuDNN implementation");
  #endif // __LIB_CUDNN
  }

  void fp_compute_cpu() {

    // Local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    Mat& mean_local = m_mean_v->Matrix();
    Mat& stdev_local = m_stdev_v->Matrix();
    Mat& running_mean_local = m_running_mean_v->Matrix();
    Mat& running_stdev_local = m_running_stdev_v->Matrix();
    Mat& scale_local = m_scale_v->Matrix();
    Mat& bias_local = m_bias_v->Matrix();
    Mat& activations_local = this->m_activations_v->Matrix();
    
    // Matrix parameters
    const El::Int width = this->m_prev_activations_v->Width();
    const El::Int local_width = this->m_prev_activations_v->LocalWidth();
    const int num_channels = this->m_neuron_dims[0];
    const int channel_size = this->m_num_neurons / this->m_neuron_dims[0];

    // Compute statistics
    if(this->get_execution_mode() == execution_mode::training) {

      // Compute sums and sums of squares
      #pragma omp parallel for
      for(int channel = 0; channel < num_channels; ++channel) {
        DataType sum = 0;
        DataType sqsum = 0;
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
        stdev_local(channel, 0) = sqsum;
      }
      
      El::AllReduce(*m_statistics_v,
                    m_statistics_v->RedundantComm(),
                    El::mpi::SUM);

      // Compute minibatch statistics and running statistics
      #pragma omp parallel for
      for(int channel = 0; channel < num_channels; ++channel) {
        const DataType mean = mean_local(channel, 0) / (width * channel_size);
        const DataType sqmean = stdev_local(channel, 0) / (width * channel_size);
        const DataType var = std::max(sqmean - mean * mean, DataType(0));
        const DataType stdev = std::sqrt(var);
        mean_local(channel, 0) = mean;
        stdev_local(channel, 0) = stdev;
        DataType& running_mean = running_mean_local(channel, 0);
        DataType& running_stdev = running_stdev_local(channel, 0);
        running_mean = m_decay * running_mean + (DataType(1) - m_decay) * mean;
        running_stdev = m_decay * running_stdev + (DataType(1) - m_decay) * stdev;
      }
      
    }

    // Iterate through channels
    #pragma omp parallel for
    for(int channel = 0; channel < num_channels; ++channel) {

      // Get channel parameters
      const DataType scale = scale_local(channel, 0);
      const DataType bias = bias_local(channel, 0);
      DataType mean, stdev;
      if(this->get_execution_mode() == execution_mode::training
         && !m_use_global_stats) { 
        mean = mean_local(channel, 0);
        stdev = stdev_local(channel, 0);
      } else {
        mean = running_mean_local(channel, 0);
        stdev = running_stdev_local(channel, 0);
      }

      // Apply batch normalization to inputs in channel
      const El::Int row_start = channel * channel_size;
      const El::Int row_end = (channel+1) * channel_size;
      for(El::Int col = 0; col < local_width; ++col) {
        for(El::Int row = row_start; row < row_end; ++row) {
          const DataType x = prev_activations_local(row, col);
          const DataType xhat = (x - mean) / (stdev + m_epsilon);
          const DataType y = scale * xhat + bias;
          activations_local(row, col) = y;
        }
      }

    }

  }  

  void bp_compute_cpu() {

    // Local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal_v->LockedMatrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();
    const Mat& mean_local = m_mean_v->LockedMatrix();
    const Mat& stdev_local = m_stdev_v->LockedMatrix();
    const Mat& running_mean_local = m_running_mean_v->LockedMatrix();
    const Mat& running_stdev_local = m_running_stdev_v->LockedMatrix();
    const Mat& scale_local = m_scale_v->LockedMatrix();
    Mat& mean_gradient_local = m_mean_gradient_v->Matrix();
    Mat& stdev_gradient_local = m_stdev_gradient_v->Matrix();
    Mat& scale_gradient_local = m_scale_gradient_v->Matrix();
    Mat& bias_gradient_local = m_bias_gradient_v->Matrix();
    
    // Matrix parameters
    const El::Int width = this->m_prev_activations_v->Width();
    const El::Int local_width = this->m_prev_activations_v->LocalWidth();
    const int num_channels = this->m_neuron_dims[0];
    const int channel_size = this->m_num_neurons / this->m_neuron_dims[0];

    // Compute local gradients
    #pragma omp parallel for
    for(int channel = 0; channel < num_channels; ++channel) {

      // Initialize channel parameters and gradients
      const DataType scale = scale_local(channel, 0);
      DataType mean, stdev;
      if(this->get_execution_mode() == execution_mode::training
         && !m_use_global_stats) { 
        mean = mean_local(channel, 0);
        stdev = stdev_local(channel, 0);
      } else {
        mean = running_mean_local(channel, 0);
        stdev = running_stdev_local(channel, 0);
      }
      DataType dscale = 0;
      DataType dbias = 0;
      DataType dmean = 0;
      DataType dstdev = 0;

      // Compute gradient contributions from local entries
      const El::Int row_start = channel * channel_size;
      const El::Int row_end = (channel+1) * channel_size;
      for(El::Int col = 0; col < local_width; ++col) {
        for(El::Int row = row_start; row < row_end; ++row) {
          const DataType x = prev_activations_local(row, col);
          const DataType xhat = (x - mean) / (stdev + m_epsilon);
          const DataType dy = prev_error_signal_local(row, col);
          dscale += dy * xhat;
          dbias += dy;
          const DataType dxhat = dy * scale;
          dmean += - dxhat / (stdev + m_epsilon);
          dstdev += - dxhat * (x - mean) / std::pow(stdev + m_epsilon, 2);
        }
      }
      scale_gradient_local(channel, 0) = dscale;
      bias_gradient_local(channel, 0) = dbias;
      mean_gradient_local(channel, 0) = dmean;
      stdev_gradient_local(channel, 0) = dstdev;

    }

    // Get global gradients by accumulating local gradients
    El::AllReduce(*m_parameters_gradient,
                  m_parameters_gradient->RedundantComm(),
                  El::mpi::SUM);
    
    // Compute error signal
    #pragma omp parallel for
    for(int channel = 0; channel < num_channels; ++channel) {

      // Initialize channel parameters and gradients
      const DataType scale = scale_local(channel, 0);
      DataType mean, stdev, dmean, dstdev;
      if(this->get_execution_mode() == execution_mode::training
         && !m_use_global_stats) { 
        mean = mean_local(channel, 0);
        stdev = stdev_local(channel, 0);
        dmean = mean_gradient_local(channel, 0);
        dstdev = stdev_gradient_local(channel, 0);
      } else {
        mean = running_mean_local(channel, 0);
        stdev = running_stdev_local(channel, 0);
        dmean = mean_gradient_local(channel, 0) * (DataType(1) - m_decay);
        dstdev = stdev_gradient_local(channel, 0) * (DataType(1) - m_decay);
      }

      // Compute error signal for current channel
      const El::Int row_start = channel * channel_size;
      const El::Int row_end = (channel+1) * channel_size;
      for(El::Int col = 0; col < local_width; ++col) {
        for(El::Int row = row_start; row < row_end; ++row) {
          const DataType x = prev_activations_local(row, col);
          const DataType dy = prev_error_signal_local(row, col);
          const DataType dxhat = dy * scale;
          DataType dx = dxhat / (stdev + m_epsilon);
          dx += dmean / (width * channel_size);
          if(stdev > 0) {
            dx += dstdev * (x - mean) / (stdev * width * channel_size);
          }
          error_signal_local(row, col) = dx;
        }
      }

    }

  }

  bool update_compute() {
    if (this->get_execution_mode() == execution_mode::training) {
      m_scale_optimizer->update(m_scale_gradient_v);
      m_bias_optimizer->update(m_bias_gradient_v);
    }
    return true;
  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
