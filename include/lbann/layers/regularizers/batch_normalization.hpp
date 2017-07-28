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
  AbsDistMat *m_var_v;
  /** View into running means. */
  AbsDistMat *m_running_mean_v;
  /** View into running variance. */
  AbsDistMat *m_running_var_v;
  /** View into gradient w.r.t. means. */
  AbsDistMat *m_mean_gradient_v;
  /** View into gradient w.r.t. standard deviations. */
  AbsDistMat *m_var_gradient_v;

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

#ifdef __LIB_CUDNN

  /** Channel tensor descriptor.
   *  This tensor has the same dimensions as the means, variances,
   *  scaling term and bias term. */
  cudnnTensorDescriptor_t m_channel_tensor_desc;

  /** GPU memory for current minibatch means. */
  std::vector<DataType *> m_mean_d;
  /** GPU memory for current minibatch variances. */
  std::vector<DataType *> m_var_d;
  /** GPU memory for running means. */
  std::vector<DataType *> m_running_mean_d;
  /** GPU memory for running variances. */
  std::vector<DataType *> m_running_var_d;
  /** GPU memory for scaling term. */
  std::vector<DataType *> m_scale_d;
  /** GPU memory for bias term. */
  std::vector<DataType *> m_bias_d;
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
   * @param scale_init The initial value for scaling parameter
   * \f$\gamma$\f$. The paper recommends 1.0 as a starting point, but
   * other papers have had better results with a smaller value
   * (e.g. 0.1).
   * @param bias_init The initial value for bias parameter
   * \f$\beta\f$. This should almost always stay at zero.
   * @param epsilon A small number to avoid division by zero.
   * @param use_global_stats Whether to use running statistics when
   * training.
   */
  batch_normalization(int index,
                      lbann_comm *comm,
                      DataType decay=0.9,
                      DataType scale_init=1.0,
                      DataType bias_init=0.0,
                      DataType epsilon=1e-5,
                      cudnn::cudnn_manager *cudnn = NULL
                      )
    : regularizer_layer(index, comm),
      m_decay(decay),
      m_scale_init(scale_init),
      m_bias_init(bias_init),
      m_epsilon(epsilon) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "batch normalization only supports DATA_PARALLEL");
    // Setup the data distribution
    initialize_distributed_matrices();

  #ifdef __LIB_CUDNN

    // Initialize cuDNN objects
    m_channel_tensor_desc = NULL;    

    // Initialize GPU memory if using GPU
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN

  }

  batch_normalization(const batch_normalization& other) :
    regularizer_layer(other),
    m_decay(other.m_decay),
    m_scale_init(other.m_scale_init),
    m_bias_init(other.m_bias_init),
    m_epsilon(other.m_epsilon) {

    // Copy matrices
    m_parameters = other.m_parameters->Copy();
    m_parameters_gradient = other.m_parameters_gradient->Copy();
    m_mean_v = other.m_mean_v->Copy();
    m_var_v = other.m_var_v->Copy();
    m_running_mean_v = other.m_running_mean_v->Copy();
    m_running_var_v = other.m_running_var_v->Copy();
    m_mean_gradient_v = other.m_mean_gradient_v->Copy();
    m_var_gradient_v = other.m_var_gradient_v->Copy();
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
    if(m_var_v)               delete m_var_v;
    if(m_running_mean_v)      delete m_running_mean_v;
    if(m_running_var_v)       delete m_running_var_v;
    if(m_mean_gradient_v)     delete m_mean_gradient_v;
    if(m_var_gradient_v)      delete m_var_gradient_v;
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

    // Copy matrices
    m_parameters = other.m_parameters->Copy();
    m_parameters_gradient = other.m_parameters_gradient->Copy();
    m_mean_v = other.m_mean_v->Copy();
    m_var_v = other.m_var_v->Copy();
    m_running_mean_v = other.m_running_mean_v->Copy();
    m_running_var_v = other.m_running_var_v->Copy();
    m_mean_gradient_v = other.m_mean_gradient_v->Copy();
    m_var_gradient_v = other.m_var_gradient_v->Copy();
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
  #ifdef __LIB_CUDNN

    // Destroy cuDNN objects
    if(m_channel_tensor_desc) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_channel_tensor_desc));
    }

    // Deallocate GPU memory
    this->m_cudnn->deallocate_on_gpus(m_mean_d);
    this->m_cudnn->deallocate_on_gpus(m_var_d);
    this->m_cudnn->deallocate_on_gpus(m_running_mean_d);
    this->m_cudnn->deallocate_on_gpus(m_running_var_d);
    this->m_cudnn->deallocate_on_gpus(m_scale_d);
    this->m_cudnn->deallocate_on_gpus(m_bias_d);
    this->m_cudnn->deallocate_on_gpus(m_scale_gradient_d);
    this->m_cudnn->deallocate_on_gpus(m_bias_gradient_d);

  #endif // #ifdef __LIB_CUDNN

    // Deallocate matrices
    if(m_parameters)          delete m_parameters;
    if(m_parameters_gradient) delete m_parameters_gradient;
    if(m_mean_v)              delete m_mean_v;
    if(m_var_v)               delete m_var_v;
    if(m_running_mean_v)      delete m_running_mean_v;
    if(m_running_var_v)       delete m_running_var_v;
    if(m_mean_gradient_v)     delete m_mean_gradient_v;
    if(m_var_gradient_v)      delete m_var_gradient_v;
    if(m_scale_v)             delete m_scale_v;
    if(m_bias_v)              delete m_bias_v;
    if(m_scale_gradient_v)    delete m_scale_gradient_v;
    if(m_bias_gradient_v)     delete m_bias_gradient_v;

    // Deallocate optimizers
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
    m_var_v = new StarMat(this->m_comm->get_model_grid());
    m_running_mean_v = new StarMat(this->m_comm->get_model_grid());
    m_running_var_v = new StarMat(this->m_comm->get_model_grid());
    m_mean_gradient_v = new StarMat(this->m_comm->get_model_grid());
    m_var_gradient_v = new StarMat(this->m_comm->get_model_grid());
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
    El::View(*m_running_var_v, *m_parameters, El::ALL, El::IR(3));
    El::Zero(*m_running_mean_v);
    El::Zero(*m_running_var_v);

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
    El::View(*m_var_v, *m_statistics_v, El::ALL, El::IR(1));
    El::View(*m_running_mean_v, *m_parameters, El::ALL, El::IR(2));
    El::View(*m_running_var_v, *m_parameters, El::ALL, El::IR(3));
    El::View(*m_scale_v, *m_parameters, El::ALL, El::IR(4));
    El::View(*m_bias_v, *m_parameters, El::ALL, El::IR(5));

    // Initialize views into parameter gradients
    El::View(*m_mean_gradient_v, *m_parameters_gradient, El::ALL, El::IR(0));
    El::View(*m_var_gradient_v, *m_parameters_gradient, El::ALL, El::IR(1));
    El::View(*m_scale_gradient_v, *m_parameters_gradient, El::ALL, El::IR(2));
    El::View(*m_bias_gradient_v, *m_parameters_gradient, El::ALL, El::IR(3));

  }

  void setup_gpu() {
    regularizer_layer::setup_gpu();
  #ifndef __LIB_CUDNN
    throw lbann_exception("convolution_layer: cuDNN not detected");
  #else

    // Set tensor descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_channel_tensor_desc));
    std::vector<int> tensor_dims(this->m_num_neuron_dims+1, 1);
    std::vector<int> tensor_strides(this->m_num_neuron_dims+1, 1);
    tensor_dims[1] = this->m_neuron_dims[0];
    tensor_strides[0] = tensor_dims[1];
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(m_channel_tensor_desc,
                                           this->m_cudnn->get_cudnn_data_type(),
                                           tensor_dims.size(),
                                           tensor_dims.data(),
                                           tensor_strides.data()));

    // Allocate GPU memory
    this->m_cudnn->allocate_on_gpus(m_mean_d,
                                    m_mean_v->Height(),
                                    m_mean_v->Width());
    this->m_cudnn->allocate_on_gpus(m_var_d,
                                    m_var_v->Height(),
                                    m_var_v->Width());
    this->m_cudnn->allocate_on_gpus(m_running_mean_d,
                                    m_running_mean_v->Height(),
                                    m_running_mean_v->Width());
    this->m_cudnn->allocate_on_gpus(m_running_var_d,
                                    m_running_var_v->Height(),
                                    m_running_var_v->Width());
    this->m_cudnn->allocate_on_gpus(m_scale_d,
                                    m_scale_v->Height(),
                                    m_scale_v->Width());
    this->m_cudnn->allocate_on_gpus(m_bias_d,
                                    m_bias_v->Height(),
                                    m_bias_v->Width());
    this->m_cudnn->allocate_on_gpus(m_scale_gradient_d,
                                    m_scale_gradient_v->Height(),
                                    m_scale_gradient_v->Width());
    this->m_cudnn->allocate_on_gpus(m_bias_gradient_d,
                                    m_bias_gradient_v->Height(),
                                    m_bias_gradient_v->Width());

  #endif // __LIB_CUDNN

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

  /** Batch normalization forward propagation using cuDNN.
   *  Note: Batch statistics are computed separately on each
   *  GPU. Running statistics for inference are global, but the
   *  running variance is approximate (it is the weighted average of
   *  variances computed on each GPU).
   */
  void fp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("batch_normalization_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Get local matrices
    Mat& mean_local = m_mean_v->Matrix();
    Mat& var_local = m_var_v->Matrix();
    Mat& running_mean_local = m_running_mean_v->Matrix();
    Mat& running_var_local = m_running_var_v->Matrix();
    const Mat& scale_local = m_scale_v->LockedMatrix();
    const Mat& bias_local = m_bias_v->LockedMatrix();

    // Clear unused GPU data
    this->m_cudnn->clear_unused_columns_on_gpus(this->m_prev_activations_d,
                                                this->m_num_prev_neurons,
                                                this->m_prev_activations_v->LocalWidth(),
                                                this->m_mini_batch_size_per_gpu);

    // Transfer parameters from CPU to GPUs
    this->m_cudnn->broadcast_to_gpus(m_scale_d, scale_local);
    this->m_cudnn->broadcast_to_gpus(m_bias_d, bias_local);
    if(this->get_execution_mode() != execution_mode::training) {
      this->m_cudnn->broadcast_to_gpus(m_running_mean_d, running_mean_local);
      this->m_cudnn->broadcast_to_gpus(m_running_var_d, running_var_local);
    }

    // Initialize tensor descriptor
    cudnnDataType_t cudnn_data_type;
    int cudnn_num_dims;
    std::vector<int> input_dims(std::max(this->m_num_prev_neuron_dims+1, 4));
    std::vector<int> input_strides(std::max(this->m_num_prev_neuron_dims+1, 4));
    std::vector<int> output_dims(std::max(this->m_num_neuron_dims+1, 4));
    std::vector<int> output_strides(std::max(this->m_num_neuron_dims+1, 4));
    CHECK_CUDNN(cudnnGetTensorNdDescriptor(this->m_prev_neurons_cudnn_desc,
                                           input_dims.size(),
                                           &cudnn_data_type,
                                           &cudnn_num_dims,
                                           input_dims.data(),
                                           input_strides.data()));
  #ifdef LBANN_DEBUG
    if(cudnn_data_type != m_cudnn->get_cudnn_data_type()) {
      throw lbann_exception("batch_normalization_layer: cuDNN descriptor for previous neuron tensor has unexpected data type");
    }
    if(cudnn_num_dims != (int) input_dims.size()) {
      throw lbann_exception("batch_normalization_layer: cuDNN descriptor for previous neuron tensor has unexpected dimensions");
    }
  #endif // LBANN_DEBUG
    CHECK_CUDNN(cudnnGetTensorNdDescriptor(this->m_neurons_cudnn_desc,
                                           output_dims.size(),
                                           &cudnn_data_type,
                                           &cudnn_num_dims,
                                           output_dims.data(),
                                           output_strides.data()));
  #ifdef LBANN_DEBUG
    if(cudnn_data_type != m_cudnn->get_cudnn_data_type()) {
      throw lbann_exception("batch_normalization_layer: cuDNN descriptor for previous neuron tensor has unexpected data type");
    }
    if(cudnn_num_dims != (int) output_dims.size()) {
      throw lbann_exception("batch_normalization_layer: cuDNN descriptor for neuron tensor has unexpected dimensions");
    }
  #endif // LBANN_DEBUG
    output_dims[0] = this->m_mini_batch_size_per_gpu;
    input_dims[0] = this->m_mini_batch_size_per_gpu;
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(this->m_prev_neurons_cudnn_desc,
                                           cudnn_data_type,
                                           input_dims.size(),
                                           input_dims.data(),
                                           input_strides.data()));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(this->m_neurons_cudnn_desc,
                                           cudnn_data_type,
                                           output_dims.size(),
                                           output_dims.data(),
                                           output_strides.data()));

    // Perform batch normalization with each GPU
    const int effective_mini_batch_size = this->m_neural_network_model->get_effective_mini_batch_size();
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));

      // Get number of columns assigned to current GPU
      const int col_start = std::min(i * this->m_mini_batch_size_per_gpu,
                                     effective_mini_batch_size);
      const int col_end = std::min((i+1) * this->m_mini_batch_size_per_gpu,
                                   effective_mini_batch_size);
      const int num_cols = col_end - col_start;
      if(num_cols == 0) {
        break;
      }
      if(num_cols < this->m_mini_batch_size_per_gpu) {
        output_dims[0] = num_cols;
        input_dims[0] = num_cols;
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(this->m_prev_neurons_cudnn_desc,
                                               cudnn_data_type,
                                               input_dims.size(),
                                               input_dims.data(),
                                               input_strides.data()));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(this->m_neurons_cudnn_desc,
                                               cudnn_data_type,
                                               output_dims.size(),
                                               output_dims.data(),
                                               output_strides.data()));
      }

      // Apply batch normalization
      if(this->get_execution_mode() == execution_mode::training) {
        CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(this->m_cudnn->get_handle(i),
                                                           CUDNN_BATCHNORM_SPATIAL,
                                                           &one,
                                                           &zero,
                                                           this->m_prev_neurons_cudnn_desc,
                                                           this->m_prev_activations_d[i],
                                                           this->m_neurons_cudnn_desc,
                                                           this->m_activations_d[i],
                                                           m_channel_tensor_desc,
                                                           m_scale_d[i],
                                                           m_bias_d[i],
                                                           0.0,
                                                           NULL,
                                                           NULL,
                                                           m_epsilon,
                                                           m_mean_d[i],
                                                           m_var_d[i]));
      } else {
        CHECK_CUDNN(cudnnBatchNormalizationForwardInference(this->m_cudnn->get_handle(i),
                                                            CUDNN_BATCHNORM_SPATIAL,
                                                            &one,
                                                            &zero,
                                                            this->m_prev_neurons_cudnn_desc,
                                                            this->m_prev_activations_d[i],
                                                            this->m_neurons_cudnn_desc,
                                                            this->m_activations_d[i],
                                                            m_channel_tensor_desc,
                                                            m_scale_d[i],
                                                            m_bias_d[i],
                                                            m_running_mean_d[i],
                                                            m_running_var_d[i],
                                                            m_epsilon));
      }

    }

    // Estimate mini-batch statistics and running statistics
    // Note: Compute weighted average of means and variances to
    //   estimate statistics over entire mini-batch. The mean is
    //   exact, but the variance is approximate.
    if(this->get_execution_mode() == execution_mode::training) {
      Mat mean_per_gpu(m_mean_v->Height(), num_gpus);
      Mat var_per_gpu(m_var_v->Height(), num_gpus);
      this->m_cudnn->gather_from_gpus(mean_per_gpu, m_mean_d, 1);
      this->m_cudnn->gather_from_gpus(var_per_gpu, m_var_d, 1);
      El::Zero(*this->m_statistics_v);
      this->m_cudnn->synchronize();
      for(int i=0; i<num_gpus; ++i) {
        const int col_start = std::min(i * this->m_mini_batch_size_per_gpu,
                                       effective_mini_batch_size);
        const int col_end = std::min((i+1) * this->m_mini_batch_size_per_gpu,
                                     effective_mini_batch_size);
        const int num_cols = col_end - col_start;
        El::Axpy(DataType(num_cols), mean_per_gpu(El::ALL, El::IR(i)), m_mean_v->Matrix());
        El::Axpy(DataType(num_cols), var_per_gpu(El::ALL, El::IR(i)), m_var_v->Matrix());
      }
      *m_statistics_v *= DataType(1) / effective_mini_batch_size;
      El::AllReduce(*m_statistics_v,
                    m_statistics_v->RedundantComm(),
                    El::mpi::SUM);
      #pragma omp parallel for
      for(int channel = 0; channel < this->m_neuron_dims[0]; ++channel) {
        const DataType mean = mean_local(channel, 0);
        const DataType var = var_local(channel, 0);
        DataType& running_mean = running_mean_local(channel, 0);
        DataType& running_var = running_var_local(channel, 0);
        running_mean = m_decay * running_mean + (DataType(1) - m_decay) * mean;
        running_var = m_decay * running_var + (DataType(1) - m_decay) * var;
      }
    
    }

  #endif // __LIB_CUDNN
  }

  void bp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("batch_normalization_layer: cuDNN not detected");
  #else

    // Useful constants
    const DataType one = 1;
    const DataType zero = 0;

    // Clear unused GPU data
    this->m_cudnn->clear_unused_columns_on_gpus(this->m_prev_error_signal_d,
                                                this->m_num_neurons,
                                                this->m_prev_error_signal_v->LocalWidth(),
                                                this->m_mini_batch_size_per_gpu);

    // Transfer parameters from CPU to GPUs
    this->m_cudnn->broadcast_to_gpus(m_scale_d, m_scale_v->LockedMatrix());

    // Initialize tensor descriptor
    cudnnDataType_t cudnn_data_type;
    int cudnn_num_dims;
    std::vector<int> input_dims(std::max(this->m_num_prev_neuron_dims+1, 4));
    std::vector<int> input_strides(std::max(this->m_num_prev_neuron_dims+1, 4));
    std::vector<int> output_dims(std::max(this->m_num_neuron_dims+1, 4));
    std::vector<int> output_strides(std::max(this->m_num_neuron_dims+1, 4));
    CHECK_CUDNN(cudnnGetTensorNdDescriptor(this->m_prev_neurons_cudnn_desc,
                                           input_dims.size(),
                                           &cudnn_data_type,
                                           &cudnn_num_dims,
                                           input_dims.data(),
                                           input_strides.data()));
  #ifdef LBANN_DEBUG
    if(cudnn_data_type != m_cudnn->get_cudnn_data_type()) {
      throw lbann_exception("batch_normalization_layer: cuDNN descriptor for previous neuron tensor has unexpected data type");
    }
    if(cudnn_num_dims != (int) input_dims.size()) {
      throw lbann_exception("batch_normalization_layer: cuDNN descriptor for previous neuron tensor has unexpected dimensions");
    }
  #endif // LBANN_DEBUG
    CHECK_CUDNN(cudnnGetTensorNdDescriptor(this->m_neurons_cudnn_desc,
                                           output_dims.size(),
                                           &cudnn_data_type,
                                           &cudnn_num_dims,
                                           output_dims.data(),
                                           output_strides.data()));
  #ifdef LBANN_DEBUG
    if(cudnn_data_type != m_cudnn->get_cudnn_data_type()) {
      throw lbann_exception("batch_normalization_layer: cuDNN descriptor for previous neuron tensor has unexpected data type");
    }
    if(cudnn_num_dims != (int) output_dims.size()) {
      throw lbann_exception("batch_normalization_layer: cuDNN descriptor for neuron tensor has unexpected dimensions");
    }
  #endif // LBANN_DEBUG
    output_dims[0] = this->m_mini_batch_size_per_gpu;
    input_dims[0] = this->m_mini_batch_size_per_gpu;
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(this->m_prev_neurons_cudnn_desc,
                                           cudnn_data_type,
                                           input_dims.size(),
                                           input_dims.data(),
                                           input_strides.data()));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(this->m_neurons_cudnn_desc,
                                           cudnn_data_type,
                                           output_dims.size(),
                                           output_dims.data(),
                                           output_strides.data()));

    // Perform batch normalization backward propagation with each GPU
    const int effective_mini_batch_size = this->m_neural_network_model->get_effective_mini_batch_size();
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));

      // Get number of columns assigned to current GPU
      const int col_start = std::min(i * this->m_mini_batch_size_per_gpu,
                                     effective_mini_batch_size);
      const int col_end = std::min((i+1) * this->m_mini_batch_size_per_gpu,
                                   effective_mini_batch_size);
      const int num_cols = col_end - col_start;
      if(num_cols == 0) {
        this->m_cudnn->clear_unused_columns_on_gpus(m_scale_gradient_d,
                                                    m_scale_gradient_v->Height(),
                                                    i,
                                                    1);
        this->m_cudnn->clear_unused_columns_on_gpus(m_bias_gradient_d,
                                                    m_bias_gradient_v->Height(),
                                                    i,
                                                    1);
        break;
      }
      if(num_cols < this->m_mini_batch_size_per_gpu) {
        output_dims[0] = num_cols;
        input_dims[0] = num_cols;
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(this->m_prev_neurons_cudnn_desc,
                                               cudnn_data_type,
                                               input_dims.size(),
                                               input_dims.data(),
                                               input_strides.data()));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(this->m_neurons_cudnn_desc,
                                               cudnn_data_type,
                                               output_dims.size(),
                                               output_dims.data(),
                                               output_strides.data()));
      }

      // Apply batch normalization backward propagation
      CHECK_CUDNN(cudnnBatchNormalizationBackward(this->m_cudnn->get_handle(i),
                                                  CUDNN_BATCHNORM_SPATIAL,
                                                  &one,
                                                  &zero,
                                                  &one,
                                                  &zero,
                                                  this->m_prev_neurons_cudnn_desc,
                                                  this->m_prev_activations_d[i],
                                                  this->m_neurons_cudnn_desc,
                                                  this->m_prev_error_signal_d[i],
                                                  this->m_prev_neurons_cudnn_desc,
                                                  this->m_error_signal_d[i],
                                                  m_channel_tensor_desc,
                                                  m_scale_d[i],
                                                  m_scale_gradient_d[i],
                                                  m_bias_gradient_d[i],
                                                  m_epsilon,
                                                  m_mean_d[i],
                                                  m_var_d[i]));

    }

    // Transfer outputs from GPUs to CPU and reduce
    this->m_cudnn->reduce_from_gpus(m_scale_gradient_v->Matrix(),
                                    m_scale_gradient_d);
    this->m_cudnn->reduce_from_gpus(m_bias_gradient_v->Matrix(),
                                    m_bias_gradient_d);
    *m_parameters_gradient *= DataType(1) / effective_mini_batch_size;
    El::AllReduce(*m_parameters_gradient,
                  m_parameters_gradient->RedundantComm(),
                  El::mpi::SUM);

  #endif // __LIB_CUDNN
  }

  void fp_compute_cpu() {

    // Local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    Mat& mean_local = m_mean_v->Matrix();
    Mat& var_local = m_var_v->Matrix();
    Mat& running_mean_local = m_running_mean_v->Matrix();
    Mat& running_var_local = m_running_var_v->Matrix();
    Mat& scale_local = m_scale_v->Matrix();
    Mat& bias_local = m_bias_v->Matrix();
    Mat& activations_local = this->m_activations_v->Matrix();
    
    // Matrix parameters
    const El::Int width = this->m_prev_activations_v->Width();
    const El::Int local_width = this->m_prev_activations_v->LocalWidth();
    const int num_channels = this->m_neuron_dims[0];
    const int channel_size = this->m_num_neurons / num_channels;

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
        var_local(channel, 0) = sqsum;
      }
      El::AllReduce(*m_statistics_v,
                    m_statistics_v->RedundantComm(),
                    El::mpi::SUM);

      // Compute minibatch statistics and running statistics
      #pragma omp parallel for
      for(int channel = 0; channel < num_channels; ++channel) {
        const DataType mean = mean_local(channel, 0) / (width * channel_size);
        const DataType sqmean = var_local(channel, 0) / (width * channel_size);
        const DataType var = std::max(sqmean - mean * mean, DataType(0));
        mean_local(channel, 0) = mean;
        var_local(channel, 0) = var;
        DataType& running_mean = running_mean_local(channel, 0);
        DataType& running_var = running_var_local(channel, 0);
        running_mean = m_decay * running_mean + (DataType(1) - m_decay) * mean;
        running_var = m_decay * running_var + (DataType(1) - m_decay) * var;
      }
      
    }

    // Iterate through channels
    #pragma omp parallel for
    for(int channel = 0; channel < num_channels; ++channel) {

      // Get channel parameters
      DataType mean, var;
      if(this->get_execution_mode() == execution_mode::training) { 
        mean = mean_local(channel, 0);
        var = var_local(channel, 0);
      } else {
        mean = running_mean_local(channel, 0);
        var = running_var_local(channel, 0);
      }
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

    // Local matrices
    const Mat& prev_activations_local = this->m_prev_activations_v->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal_v->LockedMatrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();
    const Mat& mean_local = m_mean_v->LockedMatrix();
    const Mat& var_local = m_var_v->LockedMatrix();
    const Mat& scale_local = m_scale_v->LockedMatrix();
    Mat& mean_gradient_local = m_mean_gradient_v->Matrix();
    Mat& var_gradient_local = m_var_gradient_v->Matrix();
    Mat& scale_gradient_local = m_scale_gradient_v->Matrix();
    Mat& bias_gradient_local = m_bias_gradient_v->Matrix();
    
    // Matrix parameters
    const El::Int width = this->m_prev_activations_v->Width();
    const El::Int local_width = this->m_prev_activations_v->LocalWidth();
    const int num_channels = this->m_neuron_dims[0];
    const int channel_size = this->m_num_neurons / num_channels;

    // Compute local gradients
    #pragma omp parallel for
    for(int channel = 0; channel < num_channels; ++channel) {

      // Initialize channel parameters and gradients
      const DataType mean = mean_local(channel, 0);
      const DataType var = var_local(channel, 0);
      const DataType inv_stdev = 1 / std::sqrt(var + m_epsilon);
      const DataType scale = scale_local(channel, 0);
      DataType dmean = 0;
      DataType dvar = 0;
      DataType dscale = 0;
      DataType dbias = 0;

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
          dvar += - dxhat * (x - mean) * std::pow(inv_stdev, 3) / 2;
        }
      }
      mean_gradient_local(channel, 0) = dmean;
      var_gradient_local(channel, 0) = dvar;
      scale_gradient_local(channel, 0) = dscale;
      bias_gradient_local(channel, 0) = dbias;

    }

    // Get global gradients by accumulating local gradients
    scale_gradient_local *= DataType(1) /
      this->m_neural_network_model->get_effective_mini_batch_size();
    bias_gradient_local *= DataType(1) /
      this->m_neural_network_model->get_effective_mini_batch_size();
    El::AllReduce(*m_parameters_gradient,
                  m_parameters_gradient->RedundantComm(),
                  El::mpi::SUM);
    
    // Compute error signal
    #pragma omp parallel for
    for(int channel = 0; channel < num_channels; ++channel) {

      // Initialize channel parameters and gradients
      const DataType mean = mean_local(channel, 0);
      const DataType var = var_local(channel, 0);
      const DataType dmean = mean_gradient_local(channel, 0);
      const DataType dvar = var_gradient_local(channel, 0);
      const DataType inv_stdev = 1 / std::sqrt(var + m_epsilon);
      const DataType scale = scale_local(channel, 0);

      // Compute error signal for current channel
      const El::Int row_start = channel * channel_size;
      const El::Int row_end = (channel+1) * channel_size;
      for(El::Int col = 0; col < local_width; ++col) {
        for(El::Int row = row_start; row < row_end; ++row) {
          const DataType x = prev_activations_local(row, col);
          const DataType dy = prev_error_signal_local(row, col);
          const DataType dxhat = dy * scale;
          DataType dx = dxhat * inv_stdev;
          dx += dmean / (width * channel_size);
          dx += dvar * 2 * (x - mean) / (width * channel_size);
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
