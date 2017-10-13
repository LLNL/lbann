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
// lbann_layer_softmax .hpp .cpp - softmax layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_SOFTMAX_HPP_INCLUDED
#define LBANN_LAYER_SOFTMAX_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/models/model.hpp"
#include "lbann/objective_functions/cross_entropy.hpp"
#if defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA)
#include "lbann/layers/activations/softmax_cuda.hpp"
#endif
#include <unistd.h>
#include <string>
#include <typeinfo>
#include <typeindex>

#include <assert.h>

#define LBANN_ENABLE_SOFTMAX_CUTOFF

namespace lbann {

template <data_layout T_layout>
class softmax_layer : public activation_layer {

 private:

  /** Workspace for column-wise reductions. */
  AbsDistMat *m_workspace;
  /** View into workspace for column-wise reductions. */
  AbsDistMat *m_workspace_v;

  /** Lower bound for outputs.
   *  This should be sufficiently large to avoid denormalized
   *  floats.
   */
  DataType m_min_output;

#if defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA)
  cudnnTensorDescriptor_t m_cudnn_desc = nullptr;
#endif

 public:
  softmax_layer(int index,
                lbann_comm *comm,
                cudnn::cudnn_manager *cudnn=nullptr)
      : activation_layer(index, comm) {
    initialize_distributed_matrices();
    m_min_output = std::sqrt(std::numeric_limits<DataType>::min());
    this->m_cudnn = cudnn;
#if defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA)
    if (this->m_cudnn && T_layout == data_layout::DATA_PARALLEL) {
      this->m_using_gpus = true;
    }
#endif
  }

  softmax_layer(const softmax_layer& other) :
    activation_layer(other) {
    m_workspace = other.m_workspace->Copy();
    m_workspace_v = other.m_workspace_v->Copy();
    m_min_output = other.m_min_output;
  }

  softmax_layer& operator=(const softmax_layer& other) {
    activation_layer::operator=(other);
    if (m_workspace) {
      delete m_workspace;
      delete m_workspace_v;
    }
    m_workspace = other.m_workspace->Copy();
    m_workspace_v = other.m_workspace_v->Copy();
    m_min_output = other.m_min_output;
  }

  ~softmax_layer() {
    delete m_workspace;
    delete m_workspace_v;

#if defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA)
    if (m_cudnn_desc) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_cudnn_desc));
    }
#endif
    
  }

  softmax_layer* copy() const { return new softmax_layer(*this); }

  std::string get_name() const { return "softmax"; }

  std::string get_description() const {
    return std::string {} + " softmax" + " dataLayout: " 
           + this->get_data_layout_string(get_data_layout());
  }

  virtual inline void initialize_distributed_matrices();
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_data() {
    activation_layer::setup_data();
    m_workspace->Resize(
      1, this->m_neural_network_model->get_max_mini_batch_size());
  }

  virtual void setup_gpu() {
    activation_layer::setup_gpu();
#if !(defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA))
    throw lbann_exception("softmax: CUDA not detected");
#else
    FORCE_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_cudnn_desc));    
#endif
  }  

  void fp_set_std_matrix_view() {
    Int cur_mini_batch_size = this->m_neural_network_model->get_current_mini_batch_size();
    Layer::fp_set_std_matrix_view();
    El::View(*m_workspace_v, *m_workspace, El::ALL, El::IR(0, cur_mini_batch_size));
  }
  
  void fp_compute() {
    if(this->m_using_gpus) {
      fp_compute_cuda();
    } else {
      fp_compute_cpu();
    }
  }
  
  void fp_compute_cpu() {

    // Get local matrices and parameters
    Mat& workspace_local = m_workspace_v->Matrix();
    const Mat& prev_activations_local = this->m_prev_activations->LockedMatrix();
    Mat& activations_local = this->m_activations_v->Matrix();
    const Int local_height = activations_local.Height();
    const Int local_width = activations_local.Width();

    // Find maximum entry in each column
    #pragma omp parallel for
    for(El::Int col = 0; col < local_width; ++col) {
      DataType max_entry = prev_activations_local(0, col);
      for(El::Int row = 1; row < local_height; ++row) {
        max_entry = std::max(max_entry, prev_activations_local(row,col));
      }
      workspace_local(0, col) = max_entry;
    }
    El::AllReduce(*m_workspace_v, m_workspace_v->RedundantComm(), El::mpi::MAX);

    // Exponentiate activations and compute column sums
    // Note: Subtracting by the column max prevents activations from
    // blowing up. Large negative values underflow to 0.
    #pragma omp parallel for
    for (El::Int col = 0; col < local_width; ++col) {
      const DataType shift = workspace_local(0, col);
      DataType sum = 0;
      for (El::Int row = 0; row < local_height; ++row) {
        const DataType prev_activations_entry = prev_activations_local(row, col);
        const DataType activations_entry = std::exp(prev_activations_entry - shift);
        activations_local(row, col) = activations_entry;
        sum += activations_entry;
      }
      workspace_local(0, col) = sum;
    }
    El::AllReduce(*m_workspace_v, m_workspace_v->RedundantComm(), El::mpi::SUM);

    // Divide activations by column sums
    // Note: Small values are rounded to minimum output value to avoid
    // denormalized floats.
    El::IndexDependentMap(activations_local,
                          (std::function<DataType(El::Int,El::Int,const DataType&)>)
                          ([this,&workspace_local](El::Int r, El::Int c, const DataType& x)
                           ->DataType {
                            const DataType sum = workspace_local(0, c);
                            const DataType y = x / sum;
#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
                            return y > m_min_output ? y : m_min_output;
#else
                            return y;
#endif
                          }));

  }

  void fp_compute_cuda();

  void bp_compute() {
    objective_functions::cross_entropy* obj
        = dynamic_cast<objective_functions::cross_entropy*>(this->m_neural_network_model->m_obj_fn);
    if(obj != nullptr && obj->get_shortcut_softmax_layer() == this) {
      bp_compute_cross_entropy_shortcut();
      return;
    }
    
    if(this->m_using_gpus) {
      bp_compute_cuda();
    } else {
      bp_compute_cpu();
    }
  }
  
  void bp_compute_cross_entropy_shortcut() {
    if(this->m_using_gpus) {
#if defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA)
      Mat& error_signal_local = this->m_error_signal_v->Matrix();
      int height = error_signal_local.Height();
      int width = this->m_mini_batch_size_per_gpu;
      softmax_cuda::bp_compute_cross_entropy_shortcut(*this->m_cudnn,
                                                      this->m_activations_d,
                                                      this->m_prev_error_signal_d,
                                                      this->m_error_signal_d,
                                                      height, width,
                                                      m_min_output);
#else
      throw lbann_exception("softmax: CUDA not detected");
#endif
    } else {
      bp_compute_cross_entropy_shortcut_cpu();
    }
  }

  void bp_compute_cross_entropy_shortcut_cpu() {  
    // Get local matrices and parameters
    const Mat& activations_local = this->m_activations_v->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal->Matrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();

    El::IndexDependentFill(error_signal_local,
                           (std::function<DataType(El::Int,El::Int)>)
                           ([this,&activations_local,&prev_error_signal_local]
                            (El::Int r, El::Int c)->DataType {
                             const DataType activations_entry = activations_local(r,c);
                             const DataType prev_error_signal_entry = prev_error_signal_local(r,c);
#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF                               
                             if(activations_entry > m_min_output) {
                               return activations_entry - prev_error_signal_entry;
                             }
                             else {
                               return DataType(0);
                             }
#else
                             return activations_entry - prev_error_signal_entry;
#endif
                           }));
    return;
  }

  void bp_compute_cpu() {    
    const Mat& activations_local = this->m_activations_v->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal->Matrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();
    Mat& workspace_local = m_workspace_v->Matrix();
    const El::Int local_width = activations_local.Width();
    
    // Compute dot products
    // Note: prev_error_signal^T activations
    for(El::Int c=0; c<local_width; ++c) {
      workspace_local(0, c) = El::Dot(prev_error_signal_local(El::ALL,El::IR(c)),
                                      activations_local(El::ALL,El::IR(c)));
    }
    El::AllReduce(*m_workspace_v, m_workspace_v->RedundantComm(), El::mpi::SUM);

    // Update error signal
    // Note: error_signal := activations * (prev_error_signal - prev_error_signal^T activations)
    El::IndexDependentFill(error_signal_local,
                           (std::function<DataType(El::Int,El::Int)>)
                           ([this,&activations_local,&prev_error_signal_local,&workspace_local]
                            (El::Int r, El::Int c)->DataType {
                             const DataType activations_entry = activations_local(r,c);
                             const DataType prev_error_signal_entry = prev_error_signal_local(r,c);
                             const DataType dot_product_entry = workspace_local(Int(0),c);
#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF                            
                             if(activations_entry > m_min_output) {
                               return activations_entry * (prev_error_signal_entry
                                                           - dot_product_entry);
                             }
                             else {
                               return DataType(0);
                             }
#else
                               return activations_entry * (prev_error_signal_entry
                                                           - dot_product_entry);
#endif
                           }));

  }

  void bp_compute_cuda();

  bool update_compute() {
    return true;
  }

  bool saveToCheckpoint(int fd, const char *filename, size_t *bytes) {
    return Layer::saveToCheckpoint(fd, filename, bytes);
  }

  bool loadFromCheckpoint(int fd, const char *filename, size_t *bytes) {
    return Layer::loadFromCheckpoint(fd, filename, bytes);
  }

  bool saveToCheckpointShared(lbann::persist& p) {
    return Layer::saveToCheckpointShared(p);
  }

  bool loadFromCheckpointShared(lbann::persist& p) {
    return Layer::loadFromCheckpointShared(p);
  }
};

/// Matrices should be in MC,MR distributions
template<> inline void softmax_layer<data_layout::MODEL_PARALLEL>::initialize_distributed_matrices() {
  activation_layer::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
  m_workspace = new StarMRMat(this->m_comm->get_model_grid());
  m_workspace_v = new StarMRMat(this->m_comm->get_model_grid());
}

/// Weight matrices should be in Star,Star and data matrices Star,VC distributions
template<> inline void softmax_layer<data_layout::DATA_PARALLEL>::initialize_distributed_matrices() {
  activation_layer::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
  m_workspace = new StarVCMat(this->m_comm->get_model_grid());
  m_workspace_v = new StarVCMat(this->m_comm->get_model_grid());
}

template<> inline void softmax_layer<data_layout::DATA_PARALLEL>::fp_compute_cuda() {
#if !(defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA))
    throw lbann_exception("softmax: CUDA not detected");
#else
    const DataType one = 1;
    const DataType zero = 0;

    Mat& activations_local = this->m_activations_v->Matrix();
    const Int local_height = activations_local.Height();
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(m_cudnn_desc,
                                           CUDNN_TENSOR_NCHW,
                                           cudnn::get_cudnn_data_type(),                                           
                                           this->m_mini_batch_size_per_gpu,
                                           local_height, 1, 1));

    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      FORCE_CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      FORCE_CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                       this->m_cudnn->get_stream(i)));
      FORCE_CHECK_CUDNN(cudnnSoftmaxForward(this->m_cudnn->get_handle(i),
                                            CUDNN_SOFTMAX_ACCURATE,
                                            CUDNN_SOFTMAX_MODE_CHANNEL,
                                            &one,
                                            m_cudnn_desc,
                                            this->m_prev_activations_d[i],
                                            &zero,
                                            m_cudnn_desc,
                                            this->m_activations_d[i]));
    }
    softmax_cuda::fp_cutoff(*this->m_cudnn, this->m_activations_d,
                            local_height, this->m_mini_batch_size_per_gpu,
                            this->m_min_output);
#endif
}

template<> inline void softmax_layer<data_layout::MODEL_PARALLEL>::fp_compute_cuda() {
#if !(defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA))
    throw lbann_exception("softmax: CUDA not detected");
#else
    throw lbann_exception("softmax: model-parallel CUDA not implemented");
#endif
}

template<> inline void softmax_layer<data_layout::DATA_PARALLEL>::bp_compute_cuda() {
#if !(defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA))
    throw lbann_exception("softmax: CUDA not detected");
#else
    const DataType one = 1;
    const DataType zero = 0;

    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      FORCE_CHECK_CUDNN(cudnnSoftmaxBackward(this->m_cudnn->get_handle(i),
                                             CUDNN_SOFTMAX_ACCURATE,
                                             CUDNN_SOFTMAX_MODE_CHANNEL,
                                             &one,
                                             m_cudnn_desc,
                                             this->m_activations_d[i],
                                             m_cudnn_desc,
                                             this->m_prev_error_signal_d[i],
                                             &zero,
                                             m_cudnn_desc,
                                             this->m_error_signal_d[i]));
    }
    softmax_cuda::bp_cutoff(*this->m_cudnn, this->m_activations_d,
                            this->m_error_signal_d,
                            this->m_activations_v->Matrix().Height(),
                            this->m_mini_batch_size_per_gpu,
                            this->m_min_output);
    
#endif
}

template<> inline void softmax_layer<data_layout::MODEL_PARALLEL>::bp_compute_cuda() {
#if !(defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA))
    throw lbann_exception("softmax: CUDA not detected");
#else
    throw lbann_exception("softmax: model-parallel CUDA not implemented");    
#endif
}

}  // namespace lbann

#endif  // LBANN_LAYER_SOFTMAX_HPP_INCLUDED
