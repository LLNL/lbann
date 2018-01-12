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
#if defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA)
#include "lbann/layers/activations/softmax_cuda.hpp"
#endif
#include <unistd.h>
#include <string>
#include <typeinfo>
#include <typeindex>

#include <cassert>

// Output has minimum value to avoid denormalized floats
#define LBANN_ENABLE_SOFTMAX_CUTOFF

namespace lbann {

/** Softmax layer. */
template <data_layout T_layout>
class softmax_layer : public activation_layer {

 private:

  /** Workspace for column-wise reductions. */
  AbsDistMat *m_workspace;

  /** Lower bound for outputs.
   *  This should be sufficiently large to avoid denormalized
   *  floats.
   */
  DataType m_min_output;

#if defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA)
  cudnnTensorDescriptor_t m_cudnn_desc = nullptr;
#endif

 public:

  /** Constructor. */
  softmax_layer(lbann_comm *comm,
                cudnn::cudnn_manager *cudnn=nullptr)
    : activation_layer(comm),
      m_workspace(nullptr),
      m_min_output(std::sqrt(std::numeric_limits<DataType>::min())) {
    this->m_cudnn = cudnn;
#if defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA)
    if (this->m_cudnn && T_layout == data_layout::DATA_PARALLEL) {
      this->m_using_gpus = true;
    }
#endif
  }

  softmax_layer(const softmax_layer& other)
    : activation_layer(other),
      m_min_output(other.m_min_output) {

    // Matrix deep copy
    m_workspace = other.m_workspace;
    if (m_workspace != nullptr) { m_workspace = m_workspace->Copy(); }

    // Copy GPU objects
    this->m_cudnn = other.m_cudnn;
    this->m_using_gpus = other.m_using_gpus;
#if defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA)
    if (this->m_using_gpus) {
      cudnn::copy_tensor_cudnn_desc(other.m_cudnn_desc,
                                    m_cudnn_desc);
    }
#endif

  }

  softmax_layer& operator=(const softmax_layer& other) {
    activation_layer::operator=(other);
    m_min_output = other.m_min_output;

    // Deep matrix copy
    if (m_workspace != nullptr) { delete m_workspace; }
    m_workspace = other.m_workspace;
    if (m_workspace != nullptr) { m_workspace = m_workspace->Copy(); }

    this->m_cudnn = other.m_cudnn;
    this->m_using_gpus = other.m_using_gpus;
#if defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA)    
    if (this->m_using_gpus) {
      cudnn::copy_tensor_cudnn_desc(other.m_cudnn_desc,
                                    m_cudnn_desc);
    }
#endif
  }

  ~softmax_layer() override {
    if (m_workspace != nullptr) { delete m_workspace; }
#if defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA)
    if (m_cudnn_desc) {
      CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_cudnn_desc));
    }
#endif
    
  }

  softmax_layer* copy() const override { return new softmax_layer(*this); }

  std::string get_type() const override { return "softmax"; }

  std::string get_description() const override {
    return std::string {} + " softmax" + " dataLayout: " 
           + this->get_data_layout_string(get_data_layout());
  }

  data_layout get_data_layout() const override { return T_layout; }

  void setup_matrices(const El::Grid& grid) override {
    activation_layer::setup_matrices(grid);
    if (m_workspace != nullptr) { delete m_workspace; }
    switch (get_data_layout()) {
    case data_layout::MODEL_PARALLEL:
      m_workspace = new StarMRMat(grid);
      break;
    case data_layout::DATA_PARALLEL:
      m_workspace = new StarVCMat(grid);
      break;
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid distributed matrix layout";
      throw lbann_exception(err.str());
    }
  }

  void setup_data() override {
    activation_layer::setup_data();
    const int mini_batch_size = this->m_model->get_max_mini_batch_size();
    m_workspace->Resize(1, mini_batch_size);
  }

  void setup_gpu() override {
    activation_layer::setup_gpu();
#if !(defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA))
    throw lbann_exception("softmax: CUDA not detected");
#else
    FORCE_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_cudnn_desc));    
#endif
  }  

  void fp_setup_data(int mini_batch_size) override {
    activation_layer::fp_setup_data(mini_batch_size);
    m_workspace->Resize(1, mini_batch_size);
  }
  
  void fp_compute() override {
    if(this->m_using_gpus) {
      fp_compute_cuda();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
      bp_compute_cuda();
    } else {
      bp_compute_cpu();
    }
  }
  
  virtual void fp_compute_cpu() {

    // Local matrices
    const auto& local_input = get_local_prev_activations();
    auto& local_output = get_local_activations();
    auto& local_workspace = m_workspace->Matrix();
    
    // Matrix parameters
    const El::Int local_height = local_input.Height();
    const El::Int local_width = local_input.Width();

    // Find maximum entry in each column
    #pragma omp parallel for
    for(El::Int col = 0; col < local_width; ++col) {
      DataType max_entry = local_input(0, col);
      for(El::Int row = 1; row < local_height; ++row) {
        max_entry = std::max(max_entry, local_input(row, col));
      }
      local_workspace(0, col) = max_entry;
    }
    m_comm->allreduce(*m_workspace, m_workspace->RedundantComm(),
                      El::mpi::MAX);

    // Exponentiate activations and compute column sums
    // Note: Subtracting by the column max prevents activations from
    // blowing up. Large negative values underflow to 0.
    #pragma omp parallel for
    for (El::Int col = 0; col < local_width; ++col) {
      const DataType shift = local_workspace(0, col);
      DataType sum = 0;
      for (El::Int row = 0; row < local_height; ++row) {
        const DataType x = local_input(row, col);
        const DataType y = std::exp(x - shift);
        local_output(row, col) = y;
        sum += y;
      }
      local_workspace(0, col) = sum;
    }
    m_comm->allreduce(*m_workspace, m_workspace->RedundantComm());

    // Divide activations by column sums
    // Note: Small values are rounded to minimum output value to avoid
    // denormalized floats.
    #pragma omp parallel for
    for (El::Int col = 0; col < local_width; ++col) {
      const DataType scale = DataType(1) / local_workspace(0, col);
      for (El::Int row = 0; row < local_height; ++row) {
        DataType& y = local_output(row, col);
        y *= scale;
#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
        y = std::max(y, m_min_output);
#endif
      }
    }

  }

  virtual void bp_compute_cpu() {

    // Local matrices
    const auto& local_output = get_local_activations();
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_gradient_wrt_input = get_local_error_signals();
    auto& local_workspace = m_workspace->Matrix();
    
    // Matrix parameters
    const El::Int local_height = local_output.Height();
    const El::Int local_width = local_output.Width();

    // Compute dot products between output and gradient w.r.t. output
    for (El::Int col = 0; col < local_width; ++col) {
      const auto& y = local_output(El::ALL, El::IR(col));
      const auto& dy = local_gradient_wrt_output(El::ALL, El::IR(col));
      local_workspace(0, col) = El::Dot(y, dy);
    }
    m_comm->allreduce(*m_workspace, m_workspace->RedundantComm());

    // Compute gradient w.r.t. input
    #pragma omp parallel for
    for (El::Int col = 0; col < local_width; ++col) {
      const DataType y_dot_dy = local_workspace(0, col);
      for (El::Int row = 0; row < local_height; ++row) {
        const DataType y = local_output(row, col);
        const DataType dy = local_gradient_wrt_output(row, col);
        DataType dx = y * (dy - y_dot_dy);
#ifdef LBANN_ENABLE_SOFTMAX_CUTOFF
        if (y <= m_min_output) { dx = DataType(0); }
#endif
        local_gradient_wrt_input(row, col) += dx;
      }
    }

  }

  void fp_compute_cuda();
  void bp_compute_cuda();


};

template<> inline void softmax_layer<data_layout::DATA_PARALLEL>::fp_compute_cuda() {
#if !(defined(__LIB_CUDA) && defined(LBANN_SOFTMAX_CUDA))
    throw lbann_exception("softmax: CUDA not detected");
#else
    const DataType one = 1;
    const DataType zero = 0;

    Mat& activations_local = this->m_activations_v->Matrix();
    const El::Int local_height = activations_local.Height();
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
                                            this->m_prev_activations_dv[i],
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
                                             this->m_prev_error_signal_dv[i],
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
