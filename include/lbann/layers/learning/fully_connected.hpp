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
// lbann_layer_fully_connected .hpp .cpp - Dense, fully connected, layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
#define LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED

#include "lbann/layers/learning/learning.hpp"
#include "lbann/layers/activations/activation.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/models/model.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/fan_in_fan_out_initializers.hpp"
#if defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA)
#include "lbann/layers/learning/fully_connected_cuda.hpp"
#include "lbann/utils/cublas_wrapper.hpp"
#include "lbann/base.hpp"
#endif
#include <string>
#include <sstream>

namespace lbann {

enum class device {CPU, CUDA};

template <data_layout T_layout>
class fully_connected_layer : public learning {
 private:

  /** Scaling factor for bias term. 
   *  If the scaling factor is zero, bias is not applied.
   */
  DataType m_bias_scaling_factor;

  /** View into matrix weights. */
  AbsDistMat* m_matrix_weights_v;
  /** View into bias weights. */
  AbsDistMat* m_bias_weights_v;

  /** Matrix weights gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the matrix weights.
   */
  AbsDistMat* m_matrix_weights_gradient;
  /** Bias weights gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the bias weights.
   */
  AbsDistMat* m_bias_weights_gradient;

#if defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA)
  /// GPU memory for activation weights
  std::vector<DataType *> m_weights_d;
  /// View to m_weights_d;
  std::vector<DataType *> m_activation_weights_d;
  /// View to m_weights_d;  
  std::vector<DataType *> m_bias_weights_d;
  std::vector<DataType *> m_weights_gradient_d;
  /// View to m_weights_gradient_d;
  std::vector<DataType *> m_activation_weights_gradient_d;
  Mat m_activation_weights_gradient_host_buf;
  /// View to m_weights_gradient_d;  
  std::vector<DataType *> m_bias_weights_gradient_d;
  cudnnTensorDescriptor_t m_bias_weights_desc;
  cudnnTensorDescriptor_t m_activations_desc;
  std::vector<DataType *> m_work_column_d;
#endif

  /**
   * Do layout-dependent forward propagation computation of the weights.
   */
  template <device Device>
  inline void fp_compute_weights();
  /**
   * Do layout-dependent backward propagation. This handles computing the error
   * signal for the next layer and the gradients for the weights.
   */
  template <device Device>  
  inline void bp_compute_weights();

 public:

  fully_connected_layer(lbann_comm *comm,
                        int num_neurons,  // TODO: accept a vector for neuron dims
                        weights* weight,
                        bool has_bias = true,
                        cudnn::cudnn_manager *cudnn = nullptr)
    : learning(comm) {

    // Setup the data distribution
    initialize_distributed_matrices();

    // Initialize neuron tensor dimensions
    this->m_num_neurons = num_neurons;
    this->m_num_neuron_dims = 1;
    this->m_neuron_dims.assign(1, this->m_num_neurons);

    // Initialize bias
    m_bias_scaling_factor = has_bias ? DataType(1) : DataType(0);

#if defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA)
    if (cudnn && T_layout == data_layout::DATA_PARALLEL) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
#endif
  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    return std::string {} +
     " fully_connected; num_neurons: " 
     + std::to_string(this->m_num_neurons)
     + " has_bias: " + std::to_string(this->m_bias_scaling_factor)
     + " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  static void setup_gpu_activation_bias(std::vector<DataType*> &parameters,
                                        std::vector<DataType*> &activations,
                                        std::vector<DataType*> &bias,
                                        El::Int height, El::Int width) {
    activations = parameters;
    for (unsigned i = 0; i < parameters.size(); ++i) {
      // point to the last column 
      bias.push_back(parameters[i] + height * (width - 1));
    }
  }

  fully_connected_layer(const fully_connected_layer& other) :
    learning(other),
    m_bias_scaling_factor(other.m_bias_scaling_factor) {
    m_matrix_weights_v = other.m_matrix_weights_v->Copy();
    m_bias_weights_v = other.m_bias_weights_v->Copy();
    m_matrix_weights_gradient = other.m_matrix_weights_gradient->Copy();
    m_bias_weights_gradient = other.m_bias_weights_gradient->Copy();
    setup_views();  // Update views.
#if defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA)    
    if (m_cudnn != nullptr) {
      m_weights_d = m_cudnn->copy(other.m_weights_d,
                                  this->m_weights->Height(),
                                  this->m_weights->Width());
      setup_gpu_activation_bias(m_weights_d, m_activation_weights_d,
                                m_bias_weights_d, m_weights->Height(), m_weights->Width());
      m_weights_gradient_d = m_cudnn->copy(other.m_weights_gradient_d,
                                           this->m_weights_gradient->Height(),
                                           this->m_weights_gradient->Width());
      setup_gpu_activation_bias(m_weights_gradient_d,
                                m_activation_weights_gradient_d,
                                m_bias_weights_gradient_d,
                                m_weights_gradient->Width(), m_weights_gradient->Height());
    
      if (!other.m_work_column_d.empty()) {
        m_cudnn->allocate_on_gpus(m_work_column_d, this->m_num_neurons, 1);
      }

      m_bias_weights_desc = nullptr;
      cudnn::copy_tensor_cudnn_desc(other.m_bias_weights_desc,
                                    m_bias_weights_desc);
      m_activations_desc = nullptr;
      cudnn::copy_tensor_cudnn_desc(other.m_activations_desc,
                                    m_activations_desc);

      this->m_cudnn->pin_matrix(*m_weights_gradient);
      setup_views();      
    }
 #endif
    
  }

  fully_connected_layer& operator=(const fully_connected_layer& other) {
    learning::operator=(other);
    m_bias_scaling_factor = other.m_bias_scaling_factor;

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
    COPY_MATRIX(other.m_matrix_weights_v, m_matrix_weights_v);
    COPY_MATRIX(other.m_bias_weights_v, m_bias_weights_v);
    COPY_MATRIX(other.m_matrix_weights_gradient, m_matrix_weights_gradient);
    COPY_MATRIX(other.m_bias_weights_gradient, m_bias_weights_gradient);
  #undef COPY_MATRIX
    setup_views();  // Update views.
#if defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA)
    if (m_cudnn != nullptr) {
      m_cudnn->deallocate_on_gpus(m_weights_d);
      m_weights_d = m_cudnn->copy(other.m_weights_d,
                                  this->m_weights->Height(),
                                  this->m_weights->Width());
      setup_gpu_activation_bias(m_weights_d, m_activation_weights_d,
                                m_bias_weights_d, m_weights->Height(), m_weights->Width());
      m_cudnn->deallocate_on_gpus(m_weights_gradient_d);
      m_weights_gradient_d = m_cudnn->copy(other.m_weights_gradient_d,
                                           this->m_weights_gradient->Height(),
                                           this->m_weights_gradient->Width());
      setup_gpu_activation_bias(m_weights_gradient_d,
                                m_activation_weights_gradient_d,
                                m_bias_weights_gradient_d,
                                m_weights_gradient->Width(), m_weights_gradient->Height());

      m_cudnn->deallocate_on_gpus(m_work_column_d);
      if (!other.m_work_column_d.empty()) {
        m_cudnn->allocate_on_gpus(m_work_column_d, this->m_num_neurons, 1);
      }

      cudnn::copy_tensor_cudnn_desc(other.m_bias_weights_desc,
                                    m_bias_weights_desc);
      cudnn::copy_tensor_cudnn_desc(other.m_activations_desc,
                                    m_activations_desc);

      this->m_cudnn->pin_matrix(*m_weights_gradient);
      setup_views();      
    }
#endif

    return *this;
  }

  ~fully_connected_layer() override {
    if (m_matrix_weights_v != nullptr)        delete m_matrix_weights_v;
    if (m_bias_weights_v != nullptr)          delete m_bias_weights_v;
    if (m_matrix_weights_gradient != nullptr) delete m_matrix_weights_gradient;
    if (m_bias_weights_gradient != nullptr)   delete m_bias_weights_gradient;

#if defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA)
    if (m_cudnn != nullptr) {
      this->m_cudnn->deallocate_on_gpus(m_weights_d);
      this->m_cudnn->deallocate_on_gpus(m_weights_gradient_d);
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_bias_weights_desc));
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_activations_desc));
      this->m_cudnn->unpin_matrix(*m_weights_gradient);
    }
#endif
    
  }

  fully_connected_layer* copy() const override {
    return new fully_connected_layer(*this);
  }

  std::string get_type() const override { return "fully connected"; }

  virtual inline void initialize_distributed_matrices();
  data_layout get_data_layout() const override { return T_layout; }

  void setup_dims() override {
    // Store neuron tensor dimensions
    const int num_neurons = this->m_num_neurons;
    const int num_neuron_dims = this->m_num_neuron_dims;
    const std::vector<int> neuron_dims = this->m_neuron_dims;

    // Initialize previous neuron tensor dimensions
    learning::setup_dims();

    // Initialize neuron tensor dimensions
    this->m_num_neurons = num_neurons;
    this->m_num_neuron_dims = num_neuron_dims;
    this->m_neuron_dims = neuron_dims;
  }

  void setup_data() override {
    learning::setup_data();

    // Initialize default weights if none are provided
    if (this->m_weights.size() > 2) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "attempted to setup " << m_name << " with an invalid number of weights";
      throw lbann_exception(err.str());
    }
    this->m_weights.resize(2, nullptr);
    if (this->m_weights[0] == nullptr) {
      this->m_weights[0] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[0]->set_name(this->m_name + "_matrix_weights");
      this->m_weights[0]->set_initializer(new he_normal_initializer(this->m_comm));
      this->m_weights[0]->set_optimizer(m_neural_network_model->create_optimizer());
      this->m_neural_network_model->add_weights(this->m_weights[0]);
    }
    if (this->m_weights[1] == nullptr) {
      this->m_weights[1] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[1]->set_name(this->m_name + "_bias_weights");
      this->m_weights[1]->set_optimizer(m_neural_network_model->create_optimizer());
      this->m_neural_network_model->add_weights(this->m_weights[1]);
    }

    // Initialize Glorot or He weight initialization
    fan_in_fan_out_initializer* cast_initializer
      = dynamic_cast<fan_in_fan_out_initializer*>(&this->m_weights[0]->get_initializer());
    if (cast_initializer != nullptr) {
      cast_initializer->set_fan_in(this->m_num_prev_neurons);
      cast_initializer->set_fan_out(this->m_num_neurons);
    }

    // Setup weights
    if (this->get_data_layout() == data_layout::MODEL_PARALLEL) {
      this->m_weights[0]->setup(this->m_num_neurons,
                                this->m_num_prev_neurons,
                                El::MC, El::MR);
      this->m_weights[1]->setup(this->m_num_neurons, 1,
                                El::MC, El::STAR);
    }
    else if (this->get_data_layout() == data_layout::DATA_PARALLEL) {
      this->m_weights[0]->setup(this->m_num_neurons,
                                this->m_num_prev_neurons,
                                El::STAR, El::STAR);
      this->m_weights[1]->setup(this->m_num_neurons, 1,
                                El::STAR, El::STAR);
    }
    else {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "attempted to setup " << m_name << " with an invalid data layout";
      throw lbann_exception(err.str());
    }

    // Setup weight gradients
    const AbsDistMat& matrix_weights = this->m_weights[0]->get_values();
    El::Zeros(*this->m_matrix_weights_gradient,
              this->m_weights[0]->get_height(),
              this->m_weights[0]->get_width());
    El::Zeros(*this->m_bias_weights_gradient,
              this->m_weights[1]->get_height(),
              this->m_weights[1]->get_width());

  }

  void setup_views() override {
    learning::setup_views();
    this->m_weights[0]->get_values_view(*m_matrix_weights_v);
    this->m_weights[1]->get_values_view(*m_bias_weights_v);
  }

  void setup_gpu() override {
    learning::setup_gpu();
#if !(defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA))
    throw lbann_exception("fully_connected_layer: CUDA not detected");
#else
    // DATA_PARALLEL is assumed
    // Allocate GPU memory
    this->m_cudnn->allocate_on_gpus(m_weights_d,
                                    m_weights->Height(),
                                    m_weights->Width());
    this->m_cudnn->broadcast_to_gpus(m_weights_d,
                                     m_weights->LockedMatrix());
    setup_gpu_activation_bias(m_weights_d, m_activation_weights_d,
                              m_bias_weights_d, m_weights->Height(), m_weights->Width());

    this->m_cudnn->allocate_on_gpus(m_weights_gradient_d,
                                    m_weights_gradient->Height(),
                                    m_weights_gradient->Width());
    setup_gpu_activation_bias(m_weights_gradient_d,
                              m_activation_weights_gradient_d,
                              m_bias_weights_gradient_d,
                              m_weights_gradient->Width(), m_weights_gradient->Height());

    this->m_cudnn->allocate_on_gpus(m_work_column_d, this->m_num_neurons, 1);


    // CUDNN setup
    // NOTE: Setting tensor dimensions as (1, 1, X, Y), where X is the
    // mini batch size, and bias dimensions as (1, 1, 1, Y) does not
    // work. Calls to cudnnAddTensor return
    // CUDNN_STATUS_NOT_SUPPORTED. Setting X as the dimension of C
    // works, though they should be mathematically the same. 
    FORCE_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_bias_weights_desc));
    FORCE_CHECK_CUDNN(cudnnSetTensor4dDescriptor(m_bias_weights_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 cudnn::get_cudnn_data_type(),
                                                 1, 1, 1,
                                                 m_bias_weights_v->Height()));
    FORCE_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_activations_desc));
    FORCE_CHECK_CUDNN(cudnnSetTensor4dDescriptor(m_activations_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 cudnn::get_cudnn_data_type(),
                                                 1, m_mini_batch_size_per_gpu, 1,
                                                 m_bias_weights_v->Height()));

    this->m_cudnn->pin_matrix(*m_weights_gradient);
    setup_views();
    if (this->m_optimizer != nullptr) {
      this->m_optimizer->setup_gpu(this->m_weights, this->m_weights_d);
    }
    
#endif // __LIB_CUDA
  }


  void fp_compute() override {
#ifdef __LBANN_DEBUG
    if(this->m_using_gpus) {
      this->m_cudnn->synchronize_all();
    }      
#endif
    if(this->m_using_gpus) {
      fp_compute_cuda();
    } else {
      fp_compute_cpu();
    }
#ifdef __LBANN_DEBUG
    if(this->m_using_gpus) {
      this->m_cudnn->synchronize_all();
    }      
#endif    
  }

  void fp_compute_cpu() {  
    // Apply weight matrix
    m_weights[0]->get_values_view(*m_matrix_weights_v);
    fp_compute_weights<device::CPU>();

    // Apply bias if needed
    if(m_bias_scaling_factor != DataType(0)) {
      m_weights[1]->get_values_view(*m_bias_weights_v);
      const Mat& local_bias_weights = m_bias_weights_v->LockedMatrix();
      El::IndexDependentMap(this->m_activations_v->Matrix(),
                            (std::function<DataType(El::Int,El::Int,const DataType&)>)
                            ([this,&local_bias_weights](El::Int r, El::Int c,const DataType& z)
                             ->DataType {
                              return z + m_bias_scaling_factor * local_bias_weights(r, 0);
                            }));
    }
  }

  void fp_compute_cuda() {
#if !(defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA))
    throw lbann_exception("fully_connected: CUDA not detected");
#else
    // Apply weight matrix
    fp_compute_weights<device::CUDA>();

    // Apply bias if needed
    if(m_bias_scaling_factor != DataType(0)) {
      const int num_gpus = this->m_cudnn->get_num_gpus();
      for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
#if 1
        const DataType one = 1;        
        CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                   this->m_cudnn->get_stream(i)));
        FORCE_CHECK_CUDNN(cudnnAddTensor(this->m_cudnn->get_handle(i),
                                         &m_bias_scaling_factor,
                                         m_bias_weights_desc,
                                         m_bias_weights_d[i],
                                         &one,
                                         m_activations_desc,
                                         m_activations_d[i]));
#else
        fully_connected_cuda::add_tensor(m_bias_scaling_factor, m_bias_weights_d[i],
                                         m_bias_weights_v->Height(), 1,
                                         DataType(1), m_activations_d[i],
                                         this->m_activations_v->Height(),
                                         this->m_mini_batch_size_per_gpu);
#endif
      }
    }
#ifdef LBANN_DEBUG
    this->m_cudnn->check_error();
#endif
#endif
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
      bp_compute_cuda();
    } else {
      bp_compute_cpu();
    }
    this->l2_regularize_gradient();
    this->group_lasso_regularize_gradient();
  }

  void bp_compute_cpu() {
    // Compute the error signal and gradients.
    bp_compute_weights<device::CPU>();

    // Compute bias update if needed
    optimizer* bias_optimizer = this->m_weights[1]->get_optimizer();
    if(m_bias_scaling_factor != DataType(0) && bias_optimizer != nullptr) {
      El::RowSum(*this->m_prev_error_signal,
                 *m_bias_weights_gradient);
      El::Scale(m_bias_scaling_factor /
                this->m_neural_network_model->get_effective_mini_batch_size(),
                *m_bias_weights_gradient);
      bias_optimizer->add_to_gradient(*m_bias_weights_gradient);
    }

  }

  void bp_compute_cuda() {
#if !(defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA))
    throw lbann_exception("fully_connected: CUDA not detected");
#else
    // Compute the error signal and gradients.
    bp_compute_weights<device::CUDA>();

    // Compute bias update if needed
    if(m_bias_scaling_factor != DataType(0)) {
      fully_connected_cuda::row_sum(*this->m_cudnn,
                                    m_prev_error_signal_d,
                                    m_prev_error_signal->Height(),
                                    m_mini_batch_size_per_gpu,
                                    m_bias_scaling_factor / this->m_neural_network_model->get_effective_mini_batch_size(),
                                    m_bias_weights_gradient_d);
    }

    // TODO: L2 regularization
#ifdef LBANN_DEBUG
    this->m_cudnn->check_error();
#endif
#endif // __LIB_CUDA    
  }
  

  DataType computeCost(DistMat& deltas) {
    DataType avg_error = 0.0, total_error = 0.0;
    // Compute the L2 norm on the deltas (activation - y)
    ColSumMat norms;
    ColumnTwoNorms(deltas, norms);
    int c = 0;
    // Sum the local, total error
    for(int r = 0; r < norms.LocalHeight(); r++) {
      total_error += norms.GetLocal(r,c);
    }
    mpi::AllReduce(total_error, norms.DistComm());
    avg_error = total_error / norms.Height();
    return avg_error;
  }

};

template<> inline void fully_connected_layer<data_layout::MODEL_PARALLEL>::initialize_distributed_matrices() {
  learning::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
  m_matrix_weights_gradient = new DistMat(this->m_comm->get_model_grid());
  m_bias_weights_gradient = new El::DistMatrix<DataType,MC,STAR>(this->m_comm->get_model_grid());
  m_matrix_weights_v = new DistMat(this->m_comm->get_model_grid());
  m_bias_weights_v = new El::DistMatrix<DataType,MC,STAR>(this->m_comm->get_model_grid());
}

template<> inline void fully_connected_layer<data_layout::DATA_PARALLEL>::initialize_distributed_matrices() {
  learning::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
  m_matrix_weights_gradient = new StarMat(this->m_comm->get_model_grid());
  m_bias_weights_gradient = new StarMat(this->m_comm->get_model_grid());
  m_matrix_weights_v = new StarMat(this->m_comm->get_model_grid());
  m_bias_weights_v = new StarMat(this->m_comm->get_model_grid());
}

template<> template<device Dev> inline void
fully_connected_layer<data_layout::MODEL_PARALLEL>::fp_compute_weights() {
  El::Gemm(NORMAL, NORMAL, DataType(1),
           *this->m_matrix_weights_v,
           *this->m_prev_activations,
           DataType(0),
           *this->m_activations_v);
}

template<> template<> inline void
fully_connected_layer<data_layout::DATA_PARALLEL>::fp_compute_weights<device::CPU>() {
  El::Gemm(NORMAL, NORMAL, DataType(1),
           this->m_matrix_weights_v->LockedMatrix(),
           this->m_prev_activations->LockedMatrix(),
           DataType(0),
           this->m_activations_v->Matrix());
}

#if defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA)
template<> template<> inline void
fully_connected_layer<data_layout::DATA_PARALLEL>::fp_compute_weights<device::CUDA>() {
  const int num_gpus = this->m_cudnn->get_num_gpus();
  for(int i=0; i<num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
    CHECK_CUBLAS(cublas::Gemm<DataType>(this->m_cudnn->get_cublas_handle(i),
                                        CUBLAS_OP_N, CUBLAS_OP_N, 
                                        this->m_activation_weights_v->Height(),
                                        m_mini_batch_size_per_gpu,
                                        this->m_activation_weights_v->Width(),
                                        DataType(1),
                                        this->m_activation_weights_d[i],
                                        this->m_activation_weights_v->Height(),
                                        this->m_prev_activations_d[i],
                                        this->m_prev_activations->Height(),
                                        DataType(0),
                                        this->m_activations_d[i],
                                        this->m_activations_v->Height()));
  }
}
#endif // __LIB_CUDA

template<> template<device dev> inline void
fully_connected_layer<data_layout::MODEL_PARALLEL>::bp_compute_weights() {
  // Compute the partial delta update for the next lower layer
  El::Gemm(El::TRANSPOSE, El::NORMAL, DataType(1),
           *this->m_matrix_weights_v,
           *this->m_prev_error_signal,
           DataType(0),
           *this->m_error_signal_v);

  // Compute update for activation weights
  optimizer* matrix_optimizer = this->m_weights[0]->get_optimizer();
  if (matrix_optimizer != nullptr) {
    El::Gemm(El::NORMAL, El::TRANSPOSE, DataType(1)/
             this->m_neural_network_model->get_effective_mini_batch_size(),
             *this->m_prev_error_signal,
             *this->m_prev_activations,
             DataType(0),
             *this->m_matrix_weights_gradient);
    matrix_optimizer->add_to_gradient(*this->m_matrix_weights_gradient);
  }
}

template<> template<> inline void
fully_connected_layer<data_layout::DATA_PARALLEL>::bp_compute_weights<device::CPU>() {
  El::Gemm(El::TRANSPOSE, El::NORMAL, DataType(1),
           this->m_matrix_weights_v->LockedMatrix(),
           this->m_prev_error_signal->LockedMatrix(),
           DataType(0),
           this->m_error_signal_v->Matrix());

  // Compute update for activation weights
  optimizer* matrix_optimizer = this->m_weights[0]->get_optimizer();
  if (matrix_optimizer != nullptr) {
    El::Gemm(El::NORMAL, El::TRANSPOSE, DataType(1)/
             this->m_neural_network_model->get_effective_mini_batch_size(),
             this->m_prev_error_signal->LockedMatrix(),
             this->m_prev_activations->LockedMatrix(),
             DataType(0),
             this->m_matrix_weights_gradient->Matrix());
    El::AllReduce(*this->m_matrix_weights_gradient,
                  this->m_matrix_weights_gradient->RedundantComm());
    matrix_optimizer->add_to_gradient(*this->m_matrix_weights_gradient);
  }
}

#if defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA)
template<> template<> inline void
fully_connected_layer<data_layout::DATA_PARALLEL>::bp_compute_weights<device::CUDA>() {
  const int num_gpus = this->m_cudnn->get_num_gpus();
  for(int i=0; i<num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
    CHECK_CUBLAS(cublas::Gemm<DataType>(this->m_cudnn->get_cublas_handle(i),
                                        CUBLAS_OP_T, CUBLAS_OP_N, 
                                        this->m_activation_weights_v->Width(),
                                        m_mini_batch_size_per_gpu,
                                        this->m_activation_weights_v->Height(),
                                        DataType(1),
                                        this->m_activation_weights_d[i],
                                        this->m_activation_weights_v->Height(),
                                        this->m_prev_error_signal_d[i],
                                        this->m_prev_error_signal->Height(),
                                        DataType(0),
                                        this->m_error_signal_d[i],
                                        this->m_error_signal_v->Height()));
    
    // Compute update for activation weights
    CHECK_CUBLAS(cublas::Gemm<DataType>(this->m_cudnn->get_cublas_handle(i),
                                        CUBLAS_OP_N, CUBLAS_OP_T,
                                        this->m_prev_error_signal->Height(),
                                        this->m_prev_activations->Height(),
                                        m_mini_batch_size_per_gpu,
                                        DataType(1)/
                                        this->m_neural_network_model->get_effective_mini_batch_size(),
                                        this->m_prev_error_signal_d[i],
                                        this->m_prev_error_signal->Height(),
                                        this->m_prev_activations_d[i],
                                        this->m_prev_activations->Height(),
                                        DataType(0),
                                        this->m_activation_weights_gradient_d[i],
                                        this->m_activation_weights_gradient_v->Height()));

  }

#if 1
  this->m_cudnn->allreduce(m_activation_weights_gradient_d,
                           m_activation_weights_gradient_v->Height(),
                           m_activation_weights_gradient_v->Width());

  // Skip the reduction if there is only one process for this model
  if (this->m_comm->get_procs_per_model() > 1) {

    std::vector<DataType*> t;
    t.push_back(m_activation_weights_gradient_d[0]);
    // Since we assume MPI allreduce only runs with CPU memory, the
    // data must be first copied to host, and then be copied back to GPU
    // after MPI.
    // TODO: Use CUDA-aware MPI to remove manual host-GPU transfers
    this->m_cudnn->gather_from_gpus(m_activation_weights_gradient_v->Matrix(),
                                    t, m_activation_weights_gradient_v->Width());
    this->m_cudnn->synchronize();
    El::AllReduce(*this->m_activation_weights_gradient_v,
                  this->m_activation_weights_gradient_v->RedundantComm());
    this->m_cudnn->broadcast_to_gpus(
        m_activation_weights_gradient_d,
        m_activation_weights_gradient_v->LockedMatrix());
  }
#else
  // New implementation with NCCL 2
  this->m_cudnn->allreduce_nccl(m_activation_weights_gradient_d,
                           m_activation_weights_gradient_v->Height(),
                           m_activation_weights_gradient_v->Width());
#endif
  
}
#endif // __LIB_CUDA

}  // namespace lbann

#endif  // LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
