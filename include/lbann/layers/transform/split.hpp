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

  /// List of child layers
  std::vector<const Layer*> m_children;

 public:
  /// Constructor
  split_layer(int index,
              lbann_comm *comm,
              std::vector<const Layer*> children,
              cudnn::cudnn_manager *cudnn = NULL)
    : transform(index, comm) {

    // Setup the data distribution
    initialize_distributed_matrices();

    // Initialize list of children
    for(size_t i=0; i<children.size(); ++i) {
      add_child(children[i]);
    }

  #ifdef __LIB_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      // this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN

  }

  split_layer(const split_layer&) = default;
  split_layer& operator=(const split_layer&) = default;
  ~split_layer() = default;

  split_layer* copy() const { return new split_layer(*this); }

  std::string get_name() const { return "split"; }

  virtual inline void initialize_distributed_matrices() {
    transform::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void add_child(const Layer *child) {

    // Check if child layer is null pointer
    if(child == NULL) {
      if(m_comm->am_world_master()) {
        std::cerr << "split_layer: could not add child layer since pointer is null" << "\n";
      }
      return;
    }

    // Add child layer if it isn't in list of children
    auto child_pos = std::find(m_children.begin(), m_children.end(), child);
    if(child_pos == m_children.end()) {
      m_children.push_back(child);
    }
    else {
      if(m_comm->am_world_master()) {
        std::cerr << "split_layer: could not add child layer since it is already in list of children" << "\n";
      }
    }

  }

  void remove_child(const Layer *child) {
    
    // Check if child layer is null pointer
    if(child == NULL) {
      if(m_comm->am_world_master()) {
        std::cerr << "split_layer: could not remove child layer since pointer is null" << "\n";
      }
      return;
    }

    // Remove child layer if it is in list of children
    auto child_pos = std::find(m_children.begin(), m_children.end(), child);
    if(child_pos != m_children.end()) {
      m_children.erase(child_pos);
    }
    else {
      throw lbann_exception("split_layer: could not remove child layer since it isn't in list of children");
    }

  }

  void setup_pointers(const Layer *prev_layer, const Layer *next_layer) {
    transform::setup_pointers(prev_layer, next_layer);

    // Add "next" layer to list of children
    if(this->m_next_layer != NULL) {
      add_child(this->m_next_layer);
    }

    // Make the first child layer the "next" layer
    this->m_next_layer = m_children.front();

  }

  protected:

  void fp_compute() {
    if(this->m_using_gpus) {
  #ifndef __LIB_CUDNN
      throw lbann_exception("split_layer: cuDNN not detected");
  #else
      this->m_activations_d = this->m_prev_activations_d;
  #endif // __LIB_CUDNN
    }
    else {
      El::LockedView(*this->m_activations, *this->m_prev_activations);
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
  #if 0
  #ifndef __LIB_CUDNN
    throw lbann_exception("split_layer: cuDNN not detected");
  #else

    // Useful constant
    const DataType one = 1;

    // Copy error signal from first child layer
    this->m_cudnn->copy_on_gpus(this->m_error_signal_d,
                                this->m_prev_error_signal_d);

    // Iterate through child layers
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(size_t child_index=1; child_index<m_children.size(); ++child_index) {

      // Get error signal from current child layer
      const std::vector<DataType*> input = m_children[child_index]->gpu_bp_output(this);

      // Perform addition on each GPU
      for(int i=0; i<num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                   this->m_cudnn->get_stream(i)));
        CHECK_CUDNN(cudnnAddTensor(this->m_cudnn->get_handle(i),
                                   &one,
                                   this->m_neurons_cudnn_desc,
                                   input[i],
                                   &one,
                                   this->m_prev_neurons_cudnn_desc,
                                   this->m_error_signal_d[i]));
        
      }
    }

  #endif // #ifndef __LIB_CUDNN
  #endif // 0
  }

  void bp_compute_cpu() {
    El::Copy(*this->m_prev_error_signal, *this->m_error_signal);
    for(size_t i=1; i<m_children.size(); ++i) {
      El::Axpy(DataType(1),
               m_children[i]->bp_output(this),
               *this->m_error_signal);
    }
  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_SPLIT_HPP_INCLUDED
