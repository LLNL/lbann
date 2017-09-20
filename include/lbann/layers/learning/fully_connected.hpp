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
  weight_initialization m_weight_initialization;

  /// Views of the weight matrix that allow you to separate activation weights from bias weights
  AbsDistMat *m_activation_weights_v;
  AbsDistMat *m_bias_weights_v;
  AbsDistMat *m_activation_weights_gradient_v;
  AbsDistMat *m_bias_weights_gradient_v;

  /// Special matrices to allow backprop across the bias term
  AbsDistMat *m_bias_weights_repl;
  AbsDistMat *m_bias_weights_gradient_repl;

  /** Scaling factor for bias term. 
   *  If the scaling factor is zero, bias is not applied.
   */
  DataType m_bias_scaling_factor;
  /** Initial value for bias. */
  DataType m_bias_initial_value;

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
  ////////////////////////////////////////////////////////////////////////////////
  // fully_connected_layer : single network layer class
  ////////////////////////////////////////////////////////////////////////////////
  // WB structure: (num units "neurons / filters" x (num features + 1))
  // Each row represents a neuron / filter
  // There is a column for each feature coming in from the previous layer plus 1 for the bias
  // [W00 ...   B0]
  // [|         |]
  // [Wn0       Bn]
  //
  // WB_D structure:
  // [dW     dB]
  // D structure:
  // [D        ]
  // Z, Zs, Act, Acts structure:
  // [Acts     ]

  fully_connected_layer(int index,
                        lbann_comm *comm,
                        int num_neurons,  // TODO: accept a vector for neuron dims
                        weight_initialization init,
                        optimizer *opt,
                        bool has_bias = true,
                        DataType bias_initial_value = DataType(0),
                        cudnn::cudnn_manager *cudnn = NULL)
    : learning(index, comm, opt),
      m_weight_initialization(init) {

    // Setup the data distribution
    initialize_distributed_matrices();

    // Initialize neuron tensor dimensions
    this->m_num_neurons = num_neurons;
    this->m_num_neuron_dims = 1;
    this->m_neuron_dims.assign(1, this->m_num_neurons);

    // Initialize bias
    m_bias_scaling_factor = has_bias ? DataType(1) : DataType(0);
    m_bias_initial_value = bias_initial_value;

#if defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA)
    if (cudnn && T_layout == data_layout::DATA_PARALLEL) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
#endif
  }

  /** Returns description of ctor params */
  std::string get_description() const {
    return std::string {} +
     " fully_connected; num_neurons: " 
     + std::to_string(this->m_num_neurons)
     + " weight_init: " + get_weight_initialization_name(this->m_weight_initialization)
     + " has_bias: " + std::to_string(this->m_bias_scaling_factor)
     + " bias_initial_value: " + std::to_string(this->m_bias_initial_value)
     + " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }


  fully_connected_layer(const fully_connected_layer& other) :
    learning(other),
    m_weight_initialization(other.m_weight_initialization),
    m_bias_scaling_factor(other.m_bias_scaling_factor),
    m_bias_initial_value(other.m_bias_initial_value) {
    m_bias_weights_repl = other.m_bias_weights_repl->Copy();
    m_bias_weights_gradient_repl = other.m_bias_weights_gradient_repl->Copy();
    m_activation_weights_v = other.m_activation_weights_v->Copy();
    m_activation_weights_gradient_v = other.m_activation_weights_gradient_v->Copy();
    m_bias_weights_v = other.m_bias_weights_v->Copy();
    m_bias_weights_gradient_v = other.m_bias_weights_gradient_v->Copy();
    setup_views();  // Update views.
    // Update optimizer parameters if needed.
    if (this->m_optimizer->get_parameters()) {
#if defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA)
      this->m_optimizer->set_parameters_gpu(this->m_weights,
                                            m_weights_d);
#else
      this->m_optimizer->set_parameters(this->m_weights);
#endif      
    }
  }

  fully_connected_layer& operator=(const fully_connected_layer& other) {
    learning::operator=(other);
    m_weight_initialization = other.m_weight_initialization;
    m_bias_scaling_factor = other.m_bias_scaling_factor;
    m_bias_initial_value = other.m_bias_initial_value;
    if (m_bias_weights_repl) {
      delete m_bias_weights_repl;
      delete m_bias_weights_gradient_repl;
      delete m_activation_weights_v;
      delete m_activation_weights_gradient_v;
      delete m_bias_weights_v;
      delete m_bias_weights_gradient_v;
    }
    m_bias_weights_repl = other.m_bias_weights_repl->Copy();
    m_bias_weights_gradient_repl = other.m_bias_weights_gradient_repl->Copy();
    m_activation_weights_v = other.m_activation_weights_v->Copy();
    m_activation_weights_gradient_v = other.m_activation_weights_gradient_v->Copy();
    m_bias_weights_v = other.m_bias_weights_v->Copy();
    m_bias_weights_gradient_v = other.m_bias_weights_gradient_v->Copy();
    setup_views();  // Update views.
    // Update optimizer parameters if needed.
    if (this->m_optimizer->get_parameters()) {
#if defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA)
      this->m_optimizer->set_parameters_gpu(this->m_weights,
                                            m_weights_d);
#else
      this->m_optimizer->set_parameters(this->m_weights);
#endif      
    }
    return *this;
  }

  ~fully_connected_layer() {
    delete m_bias_weights_repl;
    delete m_bias_weights_gradient_repl;
    delete m_activation_weights_v;
    delete m_activation_weights_gradient_v;
    delete m_bias_weights_v;
    delete m_bias_weights_gradient_v;

#if defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA)
    if (this->m_using_gpus) {
      this->m_cudnn->deallocate_on_gpus(m_weights_d);
      this->m_cudnn->deallocate_on_gpus(m_weights_gradient_d);
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_bias_weights_desc));
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_activations_desc));
    }
#endif
    
  }

  fully_connected_layer* copy() const {
    return new fully_connected_layer(*this);
  }

  std::string get_name() const { return "fully connected"; }

  virtual inline void initialize_distributed_matrices();
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_dims() {
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

  void setup_data() {
    learning::setup_data();
    // Initialize matrices
    // Note: the weights-bias matrix has an extra column so it includes bias term
    El::Zeros(*this->m_weights,
              this->m_num_neurons,
              this->m_num_prev_neurons + 1);
    El::Zeros(*this->m_weights_gradient,
              this->m_num_neurons,
              this->m_num_prev_neurons + 1);

    // Initialize weight matrix
    setup_views();
    initialize_matrix(*m_activation_weights_v, m_weight_initialization,
                      this->m_num_prev_neurons, this->m_num_neurons);
    El::Fill(*m_bias_weights_v, m_bias_initial_value);

    // Initialize optimizer
    if (this->m_optimizer != NULL) {
#if !(defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA))
      this->m_optimizer->setup(this->m_weights);
#else
      // setup_gpu is used when using GPUs
      if (!this->m_using_gpus) {
        this->m_optimizer->setup(this->m_weights);        
      }
#endif
    }
  }

  void setup_views() {
    learning::setup_views();
    El::View(*m_activation_weights_v, *this->m_weights,
             El::ALL, El::IR(0, this->m_num_prev_neurons));
    El::View(*m_activation_weights_gradient_v, *this->m_weights_gradient,
             El::ALL, El::IR(0, this->m_num_prev_neurons));
    El::View(*m_bias_weights_v, *this->m_weights,
             El::ALL, El::IR(this->m_num_prev_neurons));
    El::View(*m_bias_weights_gradient_v, *this->m_weights_gradient,
             El::ALL, El::IR(this->m_num_prev_neurons));
  }

  void setup_gpu() {
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
    
    this->m_cudnn->allocate_on_gpus(m_weights_gradient_d,
                                    m_weights_gradient->Height(),
                                    m_weights_gradient->Width());
    m_activation_weights_d = m_weights_d;
    m_activation_weights_gradient_d = m_weights_gradient_d;
    
    for (int i = 0; i < this->m_cudnn->get_num_gpus(); ++i) {
      // point to the last column 
      m_bias_weights_d.push_back(m_weights_d[i] +
                                 m_weights->Height() * (m_weights->Width() - 1));
      m_bias_weights_gradient_d.push_back(
          m_weights_gradient_d[i] +
          m_weights_gradient->Height() * (m_weights_gradient->Width() - 1));
    }

    this->m_cudnn->allocate_on_gpus(m_work_column_d, this->m_num_neurons, 1);


    // CUDNN setup
    FORCE_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_bias_weights_desc));

    // Two ways to create tensor descriptor. Should be identical.
#if 1
    std::vector<int> bias_dims(4, 1);
    bias_dims[3] = this->m_bias_weights_v->Height();
    std::vector<int> bias_strides(4, 1);
    for (int i = 2; i >=0; --i) {
      bias_strides[i] = bias_strides[i+1] * bias_dims[i+1];
    }
    FORCE_CHECK_CUDNN(cudnnSetTensorNdDescriptor(m_bias_weights_desc,
                                                 this->m_cudnn->get_cudnn_data_type(),
                                                 bias_dims.size(),
                                                 bias_dims.data(),
                                                 bias_strides.data()));
#else    
    FORCE_CHECK_CUDNN(cudnnSetTensor4dDescriptor(m_bias_weights_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 this->m_cudnn->get_cudnn_data_type(),
                                                 1, 1, 1, m_bias_weights_v->Height()));
#endif
    FORCE_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_activations_desc));    
#if 1
    bias_dims[3] = this->m_bias_weights_v->Height();
    bias_dims[2] = m_mini_batch_size_per_gpu;
    for (int i = 2; i >=0; --i) {
      bias_strides[i] = bias_strides[i+1] * bias_dims[i+1];
    }
    FORCE_CHECK_CUDNN(cudnnSetTensorNdDescriptor(m_activations_desc,
                                                 this->m_cudnn->get_cudnn_data_type(),
                                                 bias_dims.size(),
                                                 bias_dims.data(),
                                                 bias_strides.data()));
#else
    FORCE_CHECK_CUDNN(cudnnSetTensor4dDescriptor(m_activations_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 this->m_cudnn->get_cudnn_data_type(),
                                                 1, 1, m_mini_batch_size_per_gpu,
                                                 m_bias_weights_v->Height()));
#endif

    if (this->m_optimizer != NULL) {
      this->m_optimizer->setup_gpu(this->m_weights, this->m_weights_d);
    }
    
#endif // __LIB_CUDA
  }


  void fp_compute() {
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
    l2_regularize_objective_function();
#ifdef __LBANN_DEBUG
    if(this->m_using_gpus) {
      this->m_cudnn->synchronize_all();
    }      
#endif    
  }

  void fp_compute_cpu() {  
  // Apply weight matrix
    fp_compute_weights<device::CPU>();

    // Apply bias if needed
    if(m_bias_scaling_factor != DataType(0)) {
      El::Copy(*m_bias_weights_v, *m_bias_weights_repl);
      const Mat& local_bias_weights = m_bias_weights_repl->LockedMatrix();
      El::IndexDependentMap(this->m_activations_v->Matrix(),
                            (std::function<DataType(El::Int,El::Int,const DataType&)>)
                            ([this,&local_bias_weights](El::Int r, El::Int c,const DataType& z)->DataType {
                              return z + m_bias_scaling_factor * local_bias_weights.Get(r, 0);
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
        FORCE_CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        // CUDNN returns CUDNN_STATUS_NOT_SUPPORTED error.
        // TODO: Investigate why CUDNN returns error.
        // Use a custom CUDA code instead.
#if 0
        const DataType one = 1;        
        FORCE_CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                         this->m_cudnn->get_stream(i)));
        {
          int dims[5];
          int strides[5];
          cudnnDataType_t dt;
          int nbdims;
          // Debug printing
          cudnnGetTensorNdDescriptor(m_bias_weights_desc, 5, &dt,
                                     &nbdims, dims, strides);
          std::cerr << "bias: nbdims: " << nbdims << ", dims: " << dims[0]
                    << ", " << dims[1] << ", " << dims[2] << ", " << dims[3]
                    << ", stride: " << strides[0] << ", " << strides[1] << ", "
                    << strides[2] << ", " << strides[3] << ", type: " << dt << "\n";
          cudnnGetTensorNdDescriptor(m_activations_desc, 5, &dt,
                                     &nbdims, dims, strides);
          std::cerr << "activations: nbdims: " << nbdims << ", dims: " << dims[0]
                    << ", " << dims[1] << ", " << dims[2] << ", " << dims[3]
                    << ", stride: " << strides[0] << ", " << strides[1] << ", "
                    << strides[2] << ", " << strides[3] << ", type: " << dt << "\n";
        }
        FORCE_CHECK_CUDNN(cudnnAddTensor(this->m_cudnn->get_handle(i),
                                         &m_bias_scaling_factor,
                                         m_bias_weights_desc,
                                         m_bias_weights_d[i], &one,
                                         m_activations_desc, m_activations_d[i]));
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

  void bp_compute() {
    if(this->m_using_gpus) {
      bp_compute_cuda();
    } else {
      bp_compute_cpu();
    }
    this->l2_regularize_gradient();
  }

  void bp_compute_cpu() {
    // Compute the error signal and gradients.
    bp_compute_weights<device::CPU>();

    // Compute bias update if needed
    if(m_bias_scaling_factor != DataType(0)) {
      El::RowSum(*this->m_prev_error_signal,
                 *m_bias_weights_gradient_repl);
      El::Scale(m_bias_scaling_factor /
                this->m_neural_network_model->get_effective_mini_batch_size(),
                *m_bias_weights_gradient_repl);
      El::Copy(*m_bias_weights_gradient_repl, *m_bias_weights_gradient_v);
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

  bool update_compute() {
    if(this->m_execution_mode == execution_mode::training) {
#if !(defined(__LIB_CUDA) && defined(LBANN_FULLY_CONNECTED_CUDA))      
      this->m_optimizer->update(this->m_weights_gradient);
#else
      if (this->m_using_gpus) {
        this->m_optimizer->update_gpu(m_weights_gradient_d);
      } else {
        this->m_optimizer->update(this->m_weights_gradient);        
      }
#endif
    }
    return true;
  }

};

/// Matrices should be in MC,MR distributions
template<> inline void fully_connected_layer<data_layout::MODEL_PARALLEL>::initialize_distributed_matrices() {
  learning::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
  m_bias_weights_repl = new El::DistMatrix<DataType,MC,STAR>(this->m_comm->get_model_grid());
  m_bias_weights_gradient_repl = new El::DistMatrix<DataType,MC,STAR>(this->m_comm->get_model_grid());

  // Construct matrix views
  m_activation_weights_v          = m_weights->Construct(m_weights->Grid(),
                                                         m_weights->Root());
  m_activation_weights_gradient_v = m_weights_gradient->Construct(m_weights_gradient->Grid(),
                                                                  m_weights_gradient->Root());
  m_bias_weights_v                = m_weights->Construct(m_weights->Grid(),
                                                         m_weights->Root());
  m_bias_weights_gradient_v       = m_weights_gradient->Construct(m_weights_gradient->Grid(),
                                                                  m_weights_gradient->Root());

}

/// Weight matrices should be in Star,Star and data matrices Star,VC distributions
template<> inline void fully_connected_layer<data_layout::DATA_PARALLEL>::initialize_distributed_matrices() {
  learning::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
  m_bias_weights_repl = new StarMat(this->m_comm->get_model_grid());
  m_bias_weights_gradient_repl = new StarMat(this->m_comm->get_model_grid());

  // Construct matrix views
  m_activation_weights_v          = m_weights->Construct(m_weights->Grid(),
                                                         m_weights->Root());
  m_activation_weights_gradient_v = m_weights_gradient->Construct(m_weights_gradient->Grid(),
                                                                  m_weights_gradient->Root());
  m_bias_weights_v                = m_weights->Construct(m_weights->Grid(),
                                                         m_weights->Root());
  m_bias_weights_gradient_v       = m_weights_gradient->Construct(m_weights_gradient->Grid(),
                                                                  m_weights_gradient->Root());

}

template<> template<device Dev> inline void
fully_connected_layer<data_layout::MODEL_PARALLEL>::fp_compute_weights() {
  El::Gemm(NORMAL, NORMAL, DataType(1),
           *this->m_activation_weights_v,
           *this->m_prev_activations,
           DataType(0),
           *this->m_activations_v);
}

template<> template<> inline void
fully_connected_layer<data_layout::DATA_PARALLEL>::fp_compute_weights<device::CPU>() {
  El::Gemm(NORMAL, NORMAL, DataType(1),
           this->m_activation_weights_v->LockedMatrix(),
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
           *this->m_activation_weights_v,
           *this->m_prev_error_signal,
           DataType(0),
           *this->m_error_signal_v);

  // Compute update for activation weights
  El::Gemm(El::NORMAL, El::TRANSPOSE, DataType(1)/
           this->m_neural_network_model->get_effective_mini_batch_size(),
           *this->m_prev_error_signal,
           *this->m_prev_activations,
           DataType(0),
           *this->m_activation_weights_gradient_v);
}

template<> template<> inline void
fully_connected_layer<data_layout::DATA_PARALLEL>::bp_compute_weights<device::CPU>() {
  El::Gemm(El::TRANSPOSE, El::NORMAL, DataType(1),
           this->m_activation_weights_v->LockedMatrix(),
           this->m_prev_error_signal->LockedMatrix(),
           DataType(0),
           this->m_error_signal_v->Matrix());

  // Compute update for activation weights
  El::Gemm(El::NORMAL, El::TRANSPOSE, DataType(1)/
           this->m_neural_network_model->get_effective_mini_batch_size(),
           this->m_prev_error_signal->LockedMatrix(),
           this->m_prev_activations->LockedMatrix(),
           DataType(0),
           this->m_activation_weights_gradient_v->Matrix());
  El::AllReduce(*this->m_activation_weights_gradient_v,
                this->m_activation_weights_gradient_v->RedundantComm());
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
  
    El::AllReduce(*this->m_activation_weights_gradient_v,
                  this->m_activation_weights_gradient_v->RedundantComm());
    this->m_cudnn->broadcast_to_gpus(
        m_activation_weights_gradient_d,
        m_activation_weights_gradient_v->LockedMatrix());
  }
  
}
#endif // __LIB_CUDA

}  // namespace lbann

#endif  // LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
