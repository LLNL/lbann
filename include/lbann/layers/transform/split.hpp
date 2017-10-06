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
// split.hpp - Split layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_SPLIT_HPP_INCLUDED
#define LBANN_LAYER_SPLIT_HPP_INCLUDED

#include <vector>
#include "lbann/base.hpp"
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/// Split layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class split_layer : public transform {
 private:

 public:
  /// Constructor
  split_layer(int index,
              lbann_comm *comm,
              cudnn::cudnn_manager *cudnn = NULL)
    : transform(index, comm) {

    // Setup the data distribution
    initialize_distributed_matrices();

    // Split layer has no limit on children
    m_max_num_child_layers = -1;

  #ifdef __LIB_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN

  }

  split_layer(const split_layer&) = default;
  split_layer& operator=(const split_layer&) = default;
  ~split_layer() {
  #ifdef __LIB_CUDNN
    // GPU memory for activations is a copy of previous layer's activations
    this->m_activations_d.clear();
  #endif // __LIB_CUDNN
  }

  /** Returns description of ctor params */
  std::string get_description() const {
    std::stringstream s;
    s << " split; children: ";
    for (size_t h=0; h<this->m_child_layers.size(); h++) {
      s << this->m_child_layers[h]->get_index() << " " << this->m_child_layers[h]->get_name() << " ";
    }
    s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    return s.str();
  }

  split_layer* copy() const { return new split_layer(*this); }

  std::string get_name() const { return "split"; }

  virtual inline void initialize_distributed_matrices() {
    transform::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_gpu() {
    transform::setup_gpu();
  #ifndef __LIB_CUDNN
    throw lbann_exception("split_layer: cuDNN not detected");
  #else

    // Copy forward propagation output from GPUs if a child layer is
    // not using GPU implementation
    for(size_t i=1; i<this->m_child_layers.size(); ++i) {
      if(!this->m_child_layers[i]->using_gpus()) {
        m_copy_fp_output_from_gpus = true;
      }
    }

  #endif // #ifndef __LIB_CUDNN
  }

  protected:

  void fp_compute() {
    if(this->m_using_gpus) {
  #ifndef __LIB_CUDNN
      throw lbann_exception("split_layer: cuDNN not detected");
  #else
      this->m_cudnn->copy_on_gpus(this->m_activations_d,
                                  this->m_prev_activations_d,
                                  this->m_num_prev_neurons,
                                  this->m_mini_batch_size_per_gpu);
  #endif // __LIB_CUDNN
    }
    else {
      El::LockedView(*this->m_activations_v, *this->m_prev_activations);
    }
  }

  void bp_compute() {
    if(this->m_using_gpus) {
      bp_compute_cudnn();
    } else {
      bp_compute_cpu();
    }
  }

  void bp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("split_layer: cuDNN not detected");
  #else

    // Useful constant
    const DataType one = 1;

    // Copy error signal from first child layer
    this->m_cudnn->copy_on_gpus(this->m_error_signal_d,
                                this->m_prev_error_signal_d,
                                this->m_num_neurons,
                                this->m_mini_batch_size_per_gpu);

    // Iterate through child layers
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(size_t child_index = 1;
        child_index < this->m_child_layers.size();
        ++child_index) {
      const Layer* child = this->m_child_layers[child_index];

      // Get child error signal on GPUs
      if(child->using_gpus()) {
        child->get_gpu_bp_output(this->m_prev_error_signal_d, this);
      }
      else {
        child->get_bp_output(*this->m_prev_error_signal, this);
        this->m_cudnn->scatter_to_gpus(this->m_prev_error_signal_d,
                                       this->m_prev_error_signal->LockedMatrix(),
                                       this->m_mini_batch_size_per_gpu);
      }

      // Add child error signal to this layer's error signal
      for(int i=0; i<num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                   this->m_cudnn->get_stream(i)));
        CHECK_CUDNN(cudnnAddTensor(this->m_cudnn->get_handle(i),
                                   &one,
                                   this->m_neurons_cudnn_desc,
                                   this->m_prev_error_signal_d[i],
                                   &one,
                                   this->m_prev_neurons_cudnn_desc,
                                   this->m_error_signal_d[i]));
      }

    }

  #endif // #ifndef __LIB_CUDNN
  }

  void bp_compute_cpu() {
    El::Copy(*this->m_prev_error_signal, *this->m_error_signal_v);
    for(size_t i=1; i<this->m_child_layers.size(); ++i) {
      this->m_child_layers[i]->get_bp_output(*this->m_prev_error_signal, this);
      El::Axpy(DataType(1),
               *this->m_prev_error_signal,
               *this->m_error_signal_v);
    }
  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_SPLIT_HPP_INCLUDED
