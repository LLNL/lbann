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

#include "lbann/layers/regularizers/learning_regularizer.hpp"

namespace lbann {

#ifdef __LIB_CUDNN
namespace batch_normalization_cuda {
/** Compute sums and squares of sums over channels on GPUs. */
template <typename T>
void channel_sums_and_sqsums(int height,
                             int width,
                             int num_channels,
                             const T *data_d,
                                   T *sums_d,
                                   T *sqsums_d,
                             cudaStream_t stream);
/** Apply batch normalization on GPUs. */
template <typename T>
void batch_normalization(int height,
                         int width,
                         int num_channels,
                         const T *prev_activations_d,
                         const T *mean_d,
                         const T *var_d,
                         T epsilon,
                         const T *scale_d,
                         const T *bias_d,
                               T *activations_d,
                         cudaStream_t stream);
/** Perform first phase of batch normalization backprop on GPUs.
 *  Compute gradient w.r.t. scaling factor, bias term, mean, and
 *  variance.
 */
template <typename T>
void batch_normalization_backprop1(int height,
                                   int width,
                                   int num_channels,
                                   const T *prev_activations_d,
                                   const T *prev_error_signal_d,
                                   const T *mean_d,
                                   const T *var_d,
                                   T epsilon,
                                   const T *scale_d,
                                         T *dscale_d,
                                         T *dbias_d,
                                         T *dmean_d,
                                         T *dvar_d,
                                   cudaStream_t stream);
/** Perform second phase of batch normalization backprop on GPUs.
 *  Compute error signal (i.e. gradient w.r.t. inputs).
 */
template <typename T>
void batch_normalization_backprop2(int height,
                                   int local_width,
                                   int global_width,
                                   int num_channels,
                                   const T *prev_activations_d,
                                   const T *prev_error_signal_d,
                                   const T *mean_d,
                                   const T *var_d,
                                   T epsilon,
                                   const T *scale_d,
                                   const T *dmean_d,
                                   const T *dvar_d,
                                         T *error_signal_d,
                                   cudaStream_t stream);
} // namespace batch_normalization_cuda
#endif // __LIB_CUDNN

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
class batch_normalization : public learning_regularizer {

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
  /** View into both the scale and bias terms. */
  AbsDistMat *m_scale_bias_v;
  /** View into gradient w.r.t. scaling term. */
  AbsDistMat *m_scale_gradient_v;
  /** View into gradient w.r.t. bias term. */
  AbsDistMat *m_bias_gradient_v;
  /** View into both the scale and bias gradients. */
  AbsDistMat *m_scale_bias_gradient_v;

  /** Small number to avoid division by zero. */
  DataType m_epsilon;
  /** Whether to use global running statistics when training. */
  bool m_use_global_stats;

#ifdef __LIB_CUDNN

  /** Channel tensor descriptor.
   *  This tensor has the same dimensions as the means, variances,
   *  scaling term and bias term. */
  cudnnTensorDescriptor_t m_channel_tensor_cudnn_desc;

  /** GPU memory for current minibatch means. */
  std::vector<DataType *> m_mean_d;
  /** GPU memory for current minibatch variances. */
  std::vector<DataType *> m_var_d;
  /** GPU memory for scaling term. */
  std::vector<DataType *> m_scale_d;
  /** GPU memory for bias term. */
  std::vector<DataType *> m_bias_d;
  /** GPU memory for scaling term gradient. */
  std::vector<DataType *> m_scale_gradient_d;
  /** GPU memory for bias term gradient. */
  std::vector<DataType *> m_bias_gradient_d;
  /** GPU memory for mean gradient. */
  std::vector<DataType *> m_mean_gradient_d;
  /** GPU memory for variance gradient. */
  std::vector<DataType *> m_var_gradient_d;

  /** Workspace with pinned memory. */
  AbsDistMat *m_pinned_workspace;

#endif // __LIB_CUDNN

 public:
  /**
   * Set up batch normalization.
   * @param decay Controls the momentum of the running mean/standard
   * deviation averages.
   * @param scale_init The initial value for scaling parameter
   * \f$\gamma\f$. The paper recommends 1.0 as a starting point, but
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
                      optimizer *opt,
                      DataType decay=0.9,
                      DataType scale_init=1.0,
                      DataType bias_init=0.0,
                      DataType epsilon=1e-5,
                      cudnn::cudnn_manager *cudnn = NULL,
                      bool use_global_stats = true
                      )
    : learning_regularizer(index, comm, opt),
      m_decay(decay),
      m_scale_init(scale_init),
      m_bias_init(bias_init),
      m_epsilon(epsilon),
      m_use_global_stats(use_global_stats) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "batch normalization only supports DATA_PARALLEL");
  #ifdef LBANN_SEQUENTIAL_CONSISTENCY
    // Force global computation.
    m_use_global_stats = true;
  #endif
    // Setup the data distribution
    initialize_distributed_matrices();

  #ifdef __LIB_CUDNN

    // Initialize cuDNN objects
    m_channel_tensor_cudnn_desc = NULL;    

    // Initialize GPU memory if using GPU
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN

  }

  batch_normalization(const batch_normalization& other) :
    learning_regularizer(other),
    m_decay(other.m_decay),
    m_scale_init(other.m_scale_init),
    m_bias_init(other.m_bias_init),
    m_epsilon(other.m_epsilon),
    m_use_global_stats(other.m_use_global_stats) {

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
    m_scale_bias_v = other.m_scale_bias_v->Copy();
    m_scale_gradient_v = other.m_scale_gradient_v->Copy();
    m_bias_gradient_v = other.m_bias_gradient_v->Copy();
    m_scale_bias_gradient_v = other.m_scale_bias_gradient_v->Copy();

  #ifdef __LIB_CUDNN

    // Copy cuDNN tensor descriptor
    m_channel_tensor_cudnn_desc = nullptr;
    cudnn::copy_tensor_cudnn_desc(other.m_channel_tensor_cudnn_desc,
                                  m_channel_tensor_cudnn_desc);
    
    // Copy GPU data
    m_mean_d = m_cudnn->copy(other.m_mean_d,
                             m_mean_v->Height(),
                             m_mean_v->Width());
    m_var_d = m_cudnn->copy(other.m_var_d,
                            m_var_v->Height(),
                            m_var_v->Width());
    m_scale_d = m_cudnn->copy(other.m_scale_d,
                              m_scale_v->Height(),
                              m_scale_v->Width());
    m_bias_d = m_cudnn->copy(other.m_bias_d,
                             m_bias_v->Height(),
                             m_bias_v->Width());
    m_scale_gradient_d = m_cudnn->copy(other.m_scale_gradient_d,
                                       m_scale_gradient_v->Height(),
                                       m_scale_gradient_v->Width());
    m_bias_gradient_d = m_cudnn->copy(other.m_bias_gradient_d,
                                      m_bias_gradient_v->Height(),
                                      m_bias_gradient_v->Width());
    m_mean_gradient_d = m_cudnn->copy(other.m_mean_gradient_d,
                                      m_mean_gradient_v->Height(),
                                      m_mean_gradient_v->Width());
    m_var_gradient_d = m_cudnn->copy(other.m_var_gradient_d,
                                     m_var_gradient_v->Height(),
                                     m_var_gradient_v->Width());

    // Copy pinned workspace
    m_pinned_workspace = other.m_pinned_workspace->Copy();
    if(m_pinned_workspace != nullptr) {
      m_cudnn->pin_matrix(*m_pinned_workspace);
    }

  #endif // __LIB_CUDNN

    // Setup matrix views
    setup_views();

    // Update optimizer.
    this->m_optimizer->set_parameters(m_scale_bias_v);
  }

  batch_normalization& operator=(const batch_normalization& other) {
    learning_regularizer::operator=(other);

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
    if(m_scale_bias_v)        delete m_scale_bias_v;
    if(m_scale_gradient_v)    delete m_scale_gradient_v;
    if(m_bias_gradient_v)     delete m_bias_gradient_v;
    if(m_scale_bias_gradient_v) delete m_scale_bias_gradient_v;

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
    m_var_v = other.m_var_v->Copy();
    m_running_mean_v = other.m_running_mean_v->Copy();
    m_running_var_v = other.m_running_var_v->Copy();
    m_mean_gradient_v = other.m_mean_gradient_v->Copy();
    m_var_gradient_v = other.m_var_gradient_v->Copy();
    m_scale_v = other.m_scale_v->Copy();
    m_bias_v = other.m_bias_v->Copy();
    m_scale_bias_v = other.m_scale_bias_v->Copy();
    m_scale_gradient_v = other.m_scale_gradient_v->Copy();
    m_bias_gradient_v = other.m_bias_gradient_v->Copy();
    m_scale_bias_gradient_v = other.m_scale_bias_gradient_v->Copy();

  #ifdef __LIB_CUDNN

    // Copy cuDNN tensor descriptor
    cudnn::copy_tensor_cudnn_desc(other.m_channel_tensor_cudnn_desc,
                                  m_channel_tensor_cudnn_desc);
    
    // Copy GPU data
    m_cudnn->deallocate_on_gpus(m_mean_d);
    m_cudnn->deallocate_on_gpus(m_var_d);
    m_cudnn->deallocate_on_gpus(m_scale_d);
    m_cudnn->deallocate_on_gpus(m_bias_d);
    m_cudnn->deallocate_on_gpus(m_scale_gradient_d);
    m_cudnn->deallocate_on_gpus(m_bias_gradient_d);
    m_cudnn->deallocate_on_gpus(m_mean_gradient_d);
    m_cudnn->deallocate_on_gpus(m_var_gradient_d);
    m_mean_d = m_cudnn->copy(other.m_mean_d,
                             m_mean_v->Height(),
                             m_mean_v->Width());
    m_var_d = m_cudnn->copy(other.m_var_d,
                            m_var_v->Height(),
                            m_var_v->Width());
    m_scale_d = m_cudnn->copy(other.m_scale_d,
                              m_scale_v->Height(),
                              m_scale_v->Width());
    m_bias_d = m_cudnn->copy(other.m_bias_d,
                             m_bias_v->Height(),
                             m_bias_v->Width());
    m_scale_gradient_d = m_cudnn->copy(other.m_scale_gradient_d,
                                       m_scale_gradient_v->Height(),
                                       m_scale_gradient_v->Width());
    m_bias_gradient_d = m_cudnn->copy(other.m_bias_gradient_d,
                                      m_bias_gradient_v->Height(),
                                      m_bias_gradient_v->Width());
    m_mean_gradient_d = m_cudnn->copy(other.m_mean_gradient_d,
                                      m_mean_gradient_v->Height(),
                                      m_mean_gradient_v->Width());
    m_var_gradient_d = m_cudnn->copy(other.m_var_gradient_d,
                                     m_var_gradient_v->Height(),
                                     m_var_gradient_v->Width());

    // Copy pinned workspace
    if(m_pinned_workspace != nullptr) {
      m_cudnn->unpin_matrix(*m_pinned_workspace);
    }
    m_pinned_workspace = other.m_pinned_workspace->Copy();
    if(m_pinned_workspace != nullptr) {
      m_cudnn->pin_matrix(*m_pinned_workspace);
    }

  #endif // __LIB_CUDNN

    // Setup matrix view
    setup_views();

    // Update optimizer.
    this->m_optimizer->set_parameters(m_scale_bias_v);

    // Return copy
    return *this;

  }

  /** Returns description of ctor params */
  std::string get_description() const {
    return std::string {} +
     std::to_string(this->m_index) + " batch_normalization; decay: " 
     + std::to_string(this->m_decay) + " scale_init: " 
     + std::to_string(this->m_scale_init)
     + " bias: " + std::to_string(this->m_bias_init) + " epsilon: " 
     + std::to_string(this->m_epsilon)
     + " dataLayout: " + get_data_layout_string(get_data_layout());
  }

  ~batch_normalization() {
  #ifdef __LIB_CUDNN

    // Destroy cuDNN objects
    if(m_channel_tensor_cudnn_desc) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_channel_tensor_cudnn_desc));
    }

    // Deallocate GPU memory
    this->m_cudnn->deallocate_on_gpus(m_mean_d);
    this->m_cudnn->deallocate_on_gpus(m_var_d);
    this->m_cudnn->deallocate_on_gpus(m_scale_d);
    this->m_cudnn->deallocate_on_gpus(m_bias_d);
    this->m_cudnn->deallocate_on_gpus(m_scale_gradient_d);
    this->m_cudnn->deallocate_on_gpus(m_bias_gradient_d);
    this->m_cudnn->deallocate_on_gpus(m_mean_gradient_d);
    this->m_cudnn->deallocate_on_gpus(m_var_gradient_d);

    // Delete pinned memory workspace
    this->m_cudnn->unpin_matrix(*m_pinned_workspace);
    delete m_pinned_workspace;

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
    if(m_scale_bias_v)        delete m_scale_bias_v;
    if(m_scale_gradient_v)    delete m_scale_gradient_v;
    if(m_bias_gradient_v)     delete m_bias_gradient_v;
    if(m_scale_bias_gradient_v) delete m_scale_bias_gradient_v;
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
    m_scale_bias_v = new StarMat(this->m_comm->get_model_grid());
    m_scale_gradient_v = new StarMat(this->m_comm->get_model_grid());
    m_bias_gradient_v = new StarMat(this->m_comm->get_model_grid());
    m_scale_bias_gradient_v = new StarMat(this->m_comm->get_model_grid());
  #ifdef __LIB_CUDNN
    m_pinned_workspace = new StarMat(this->m_comm->get_model_grid());
  #endif // #ifdef __LIB_CUDNN
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
    El::View(*m_scale_bias_v, *m_parameters, El::ALL, El::IR(4, 6));
    El::Fill(*m_scale_v, m_scale_init);
    El::Fill(*m_bias_v, m_bias_init);

    // Initialize optimizer; since optimizers are element-wise, we use one
    // optimizer for both the scale and bias parameters.
    this->m_optimizer->setup(m_scale_bias_v);
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
    El::View(*m_scale_bias_v, *m_parameters, El::ALL, El::IR(4, 6));

    // Initialize views into parameter gradients
    El::View(*m_mean_gradient_v, *m_parameters_gradient, El::ALL, El::IR(0));
    El::View(*m_var_gradient_v, *m_parameters_gradient, El::ALL, El::IR(1));
    El::View(*m_scale_gradient_v, *m_parameters_gradient, El::ALL, El::IR(2));
    El::View(*m_bias_gradient_v, *m_parameters_gradient, El::ALL, El::IR(3));
    El::View(*m_scale_bias_gradient_v, *m_parameters_gradient, El::ALL,
             El::IR(2, 4));
  }

  void setup_gpu() {
    regularizer_layer::setup_gpu();
  #ifndef __LIB_CUDNN
    throw lbann_exception("convolution_layer: cuDNN not detected");
  #else

    // Set tensor descriptor
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_channel_tensor_cudnn_desc));
    std::vector<int> tensor_dims(this->m_num_neuron_dims+1, 1);
    std::vector<int> tensor_strides(this->m_num_neuron_dims+1, 1);
    tensor_dims[1] = this->m_neuron_dims[0];
    tensor_strides[0] = tensor_dims[1];
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(m_channel_tensor_cudnn_desc,
                                           cudnn::get_cudnn_data_type(),
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
    this->m_cudnn->allocate_on_gpus(m_mean_gradient_d,
                                    m_mean_gradient_v->Height(),
                                    m_mean_gradient_v->Width());
    this->m_cudnn->allocate_on_gpus(m_var_gradient_d,
                                    m_var_gradient_v->Height(),
                                    m_var_gradient_v->Width());

    // Initialize pinned memory workspace
    m_pinned_workspace->Resize(m_parameters->Height(),
                               4 * this->m_cudnn->get_num_gpus());
    this->m_cudnn->pin_matrix(*m_pinned_workspace);

  #endif // __LIB_CUDNN

  }

  void fp_compute() {
    if(this->m_using_gpus) {
      fp_compute_gpu();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() {
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

    // Number of GPUs
    const int num_gpus = this->m_cudnn->get_num_gpus();

    // Get local matrices
    Mat& mean_local = m_mean_v->Matrix();
    Mat& var_local = m_var_v->Matrix();
    Mat& running_mean_local = m_running_mean_v->Matrix();
    Mat& running_var_local = m_running_var_v->Matrix();
    const Mat& scale_local = m_scale_v->LockedMatrix();
    const Mat& bias_local = m_bias_v->LockedMatrix();
    
    // Setup pinned workspace
    Mat& pinned_workspace_local = m_pinned_workspace->Matrix();
    Mat workspace1 = pinned_workspace_local(El::ALL, El::IR(0, num_gpus));
    Mat workspace2 = pinned_workspace_local(El::ALL, El::IR(num_gpus, 2*num_gpus));
    
    // Matrix parameters
    const int height = this->m_prev_activations->Height();
    const int width = this->m_prev_activations->Width();
    const int local_width = this->m_prev_activations->LocalWidth();
    const int num_channels = this->m_neuron_dims[0];
    const int channel_size = this->m_num_neurons / num_channels;

    // Compute statistics
    if(this->get_execution_mode() == execution_mode::training) {

      // Compute sums and sums of squares on each GPU
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
      
      // Reduce sums and sums of squares across GPUs and nodes
      this->m_cudnn->gather_from_gpus(workspace1, m_mean_d, 1);
      this->m_cudnn->gather_from_gpus(workspace2, m_var_d, 1);
      this->m_cudnn->synchronize();
      for(int i=0; i<num_gpus; ++i) {
        if(i == 0) {
          El::Copy(workspace1(El::ALL, El::IR(0)), mean_local);
          El::Copy(workspace2(El::ALL, El::IR(0)), var_local);
        }
        else {
          mean_local += workspace1(El::ALL, El::IR(i));
          var_local += workspace2(El::ALL, El::IR(i));
        }
      }
      if (m_use_global_stats) {
        El::AllReduce(*m_statistics_v,
                      m_statistics_v->RedundantComm(),
                      El::mpi::SUM);
      }

      // Compute minibatch statistics and running statistics
      const DataType num_samples = width * channel_size;
      #pragma omp parallel for
      for(int channel = 0; channel < num_channels; ++channel) {
        const DataType mean = mean_local(channel, 0) / num_samples;
        const DataType sqmean = var_local(channel, 0) / num_samples;
        const DataType var = num_samples / (num_samples - DataType(1)) * std::max(sqmean - mean * mean, DataType(0));
        mean_local(channel, 0) = mean;
        var_local(channel, 0) = var;
        DataType& running_mean = running_mean_local(channel, 0);
        DataType& running_var = running_var_local(channel, 0);
        running_mean = m_decay * running_mean + (DataType(1) - m_decay) * mean;
        running_var = m_decay * running_var + (DataType(1) - m_decay) * var;
      }
      
    }

    // Transfer parameters from CPU to GPUs
    this->m_cudnn->broadcast_to_gpus(m_scale_d, scale_local);
    this->m_cudnn->broadcast_to_gpus(m_bias_d, bias_local);
    if(this->get_execution_mode() == execution_mode::training) {
      this->m_cudnn->broadcast_to_gpus(m_mean_d, mean_local);
      this->m_cudnn->broadcast_to_gpus(m_var_d, var_local);
    } else {
      this->m_cudnn->broadcast_to_gpus(m_mean_d, running_mean_local);
      this->m_cudnn->broadcast_to_gpus(m_var_d, running_var_local);
    }

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
                                        m_mean_d[i],
                                        m_var_d[i],
                                        m_epsilon,
                                        m_scale_d[i],
                                        m_bias_d[i],
                                        this->m_activations_d[i],
                                        this->m_cudnn->get_stream(i));
    }

  #endif // __LIB_CUDNN
  }

  void bp_compute_gpu() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("batch_normalization_layer: cuDNN not detected");
  #else

    // Number of GPUs
    const int num_gpus = this->m_cudnn->get_num_gpus();

    // Get local matrices
    Mat& mean_gradient_local = m_mean_gradient_v->Matrix();
    Mat& var_gradient_local = m_var_gradient_v->Matrix();
    Mat& scale_gradient_local = m_scale_gradient_v->Matrix();
    Mat& bias_gradient_local = m_bias_gradient_v->Matrix();

    // Setup pinned workspace
    Mat& pinned_workspace_local = m_pinned_workspace->Matrix();
    Mat workspace1 = pinned_workspace_local(El::ALL, El::IR(0, num_gpus));
    Mat workspace2 = pinned_workspace_local(El::ALL, El::IR(num_gpus, 2*num_gpus));
    Mat workspace3 = pinned_workspace_local(El::ALL, El::IR(2*num_gpus, 3*num_gpus));
    Mat workspace4 = pinned_workspace_local(El::ALL, El::IR(3*num_gpus, 4*num_gpus));

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
                                                  m_mean_d[i],
                                                  m_var_d[i],
                                                  m_epsilon,
                                                  m_scale_d[i],
                                                  m_scale_gradient_d[i],
                                                  m_bias_gradient_d[i],
                                                  m_mean_gradient_d[i],
                                                  m_var_gradient_d[i],
                                                  this->m_cudnn->get_stream(i));
    }

    // Reduce sums and sums of squares across GPUs and nodes
    this->m_cudnn->gather_from_gpus(workspace1, m_scale_gradient_d, 1);
    this->m_cudnn->gather_from_gpus(workspace2, m_bias_gradient_d, 1);
    this->m_cudnn->gather_from_gpus(workspace3, m_mean_gradient_d, 1);
    this->m_cudnn->gather_from_gpus(workspace4, m_var_gradient_d, 1);
    this->m_cudnn->synchronize();
    for(int i=0; i<num_gpus; ++i) {
      if(i == 0) {
        El::Copy(workspace1(El::ALL, El::IR(i)), scale_gradient_local);
        El::Copy(workspace2(El::ALL, El::IR(i)), bias_gradient_local);
        El::Copy(workspace3(El::ALL, El::IR(i)), mean_gradient_local);
        El::Copy(workspace4(El::ALL, El::IR(i)), var_gradient_local);
      }
      else {
        scale_gradient_local += workspace1(El::ALL, El::IR(i));
        bias_gradient_local += workspace2(El::ALL, El::IR(i));
        mean_gradient_local += workspace3(El::ALL, El::IR(i));
        var_gradient_local += workspace4(El::ALL, El::IR(i));
      }
    }
    scale_gradient_local
      *= DataType(1) / this->m_neural_network_model->get_effective_mini_batch_size();
    bias_gradient_local
      *= DataType(1) / this->m_neural_network_model->get_effective_mini_batch_size();
    if (m_use_global_stats) {
      El::AllReduce(*m_parameters_gradient,
                    m_parameters_gradient->RedundantComm(),
                    El::mpi::SUM);
    }
    this->m_cudnn->broadcast_to_gpus(m_mean_gradient_d, mean_gradient_local);
    this->m_cudnn->broadcast_to_gpus(m_var_gradient_d, var_gradient_local);

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
                                                  m_scale_d[i],
                                                  m_mean_gradient_d[i],
                                                  m_var_gradient_d[i],
                                                  this->m_error_signal_d[i],
                                                  this->m_cudnn->get_stream(i));
    }

  #endif // __LIB_CUDNN
  }

  void fp_compute_cpu() {

    // Local matrices
    const Mat& prev_activations_local = this->m_prev_activations->LockedMatrix();
    Mat& mean_local = m_mean_v->Matrix();
    Mat& var_local = m_var_v->Matrix();
    Mat& running_mean_local = m_running_mean_v->Matrix();
    Mat& running_var_local = m_running_var_v->Matrix();
    Mat& scale_local = m_scale_v->Matrix();
    Mat& bias_local = m_bias_v->Matrix();
    Mat& activations_local = this->m_activations_v->Matrix();
    
    // Matrix parameters
    const int width = this->m_prev_activations->Width();
    const El::Int local_width = this->m_prev_activations->LocalWidth();
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
      if (m_use_global_stats) {
        El::AllReduce(*m_statistics_v,
                      m_statistics_v->RedundantComm(),
                      El::mpi::SUM);
      }

      // Compute minibatch statistics and running statistics
      const DataType num_samples = width * channel_size;
      #pragma omp parallel for
      for(int channel = 0; channel < num_channels; ++channel) {
        const DataType mean = mean_local(channel, 0) / num_samples;
        const DataType sqmean = var_local(channel, 0) / num_samples;
        const DataType var = num_samples / (num_samples - DataType(1)) * std::max(sqmean - mean * mean, DataType(0));
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
    const Mat& prev_activations_local = this->m_prev_activations->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal->LockedMatrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();
    const Mat& mean_local = m_mean_v->LockedMatrix();
    const Mat& var_local = m_var_v->LockedMatrix();
    const Mat& scale_local = m_scale_v->LockedMatrix();
    Mat& mean_gradient_local = m_mean_gradient_v->Matrix();
    Mat& var_gradient_local = m_var_gradient_v->Matrix();
    Mat& scale_gradient_local = m_scale_gradient_v->Matrix();
    Mat& bias_gradient_local = m_bias_gradient_v->Matrix();
    
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
          dvar += - dxhat * (x - mean) * dvar_factor;
        }
      }
      mean_gradient_local(channel, 0) = dmean;
      var_gradient_local(channel, 0) = dvar;
      scale_gradient_local(channel, 0) = dscale;
      bias_gradient_local(channel, 0) = dbias;

    }

    // Get global gradients by accumulating local gradients
    scale_gradient_local
      *= DataType(1) / this->m_neural_network_model->get_effective_mini_batch_size();
    bias_gradient_local
      *= DataType(1) / this->m_neural_network_model->get_effective_mini_batch_size();
    if (m_use_global_stats) {
      El::AllReduce(*m_parameters_gradient,
                    m_parameters_gradient->RedundantComm(),
                    El::mpi::SUM);
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

  bool update_compute() {
    if (this->get_execution_mode() == execution_mode::training) {
      this->m_optimizer->update(m_scale_bias_gradient_v);
    }
    return true;
  }

};

} // namespace lbann

#endif  // LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
