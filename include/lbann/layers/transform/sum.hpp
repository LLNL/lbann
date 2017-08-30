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
// sum.hpp - Sum layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_SUM_HPP_INCLUDED
#define LBANN_LAYER_SUM_HPP_INCLUDED

#include <vector>
#include "lbann/base.hpp"
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/// Sum layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class sum_layer : public transform {
 private:

  /// List of parent layers
  std::vector<const Layer*> m_parents;

 public:
  /// Constructor
  sum_layer(int index,
            lbann_comm *comm,
            std::vector<const Layer*> parents,
            cudnn::cudnn_manager *cudnn = NULL)
    : transform(index, comm) {

    // Setup the data distribution
    initialize_distributed_matrices();

    // Initialize list of parents
    for(size_t i=0; i<parents.size(); ++i) {
      add_parent(parents[i]);
    }

  #ifdef __LIB_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN

  }

  sum_layer(const sum_layer&) = default;
  sum_layer& operator=(const sum_layer&) = default;
  ~sum_layer() {
  #ifdef __LIB_CUDNN
    // GPU memory for activations is a copy of previous layer's activations
    this->m_error_signal_d.clear();
  #endif // __LIB_CUDNN
  }

  /** Returns description of ctor params */
  std::string get_description() const {
    std::stringstream s;
     s << " sum; parents: ";
     for (size_t i=0; i<m_parents.size(); i++) {
       s << m_parents[i]->get_index() << " " << m_parents[i]->get_name() << " ";
     }
     s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
     return s.str();
  }

  sum_layer* copy() const { return new sum_layer(*this); }

  std::string get_name() const { return "sum"; }

  virtual inline void initialize_distributed_matrices() {
    transform::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void add_parent(const Layer *parent) {

    // Check if parent layer is null pointer
    if(parent == NULL) {
      if(m_comm->am_world_master()) {
        std::cerr << "sum_layer: could not add parent layer since pointer is null" << "\n";
      }
      return;
    }

    // Add parent layer if it isn't in list of parents
    auto parent_pos = std::find(m_parents.begin(), m_parents.end(), parent);
    if(parent_pos == m_parents.end()) {
      m_parents.push_back(parent);
    }
    else {
      if(m_comm->am_world_master()) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ 
          << " :: sum_layer: could not add parent layer since it is already in list of parents;\n"
          << "my index: " << this->get_index()
          << " parent index: " << parent->get_index() << " name: " << parent->get_name() << "\n"
          << "existing parent list: ";
      for (auto t : m_parents) {
        err << " index: " << t->get_index() << " name: " << t->get_name();
      }
      throw lbann_exception(err.str());
      }
    }

  }

  void remove_parent(const Layer *parent) {
    
    // Check if parent layer is null pointer
    if(parent == NULL) {
      if(m_comm->am_world_master()) {
        std::cerr << "sum_layer: could not remove parent layer since pointer is null" << "\n";
      }
      return;
    }

    // Remove parent layer if it is in list of parents
    auto parent_pos = std::find(m_parents.begin(), m_parents.end(), parent);
    if(parent_pos != m_parents.end()) {
      m_parents.erase(parent_pos);
    }
    else {
      throw lbann_exception("sum_layer: could not remove parent layer since it isn't in list of parents");
    }

  }

  void setup_pointers(const Layer *prev_layer, const Layer *next_layer) {
    transform::setup_pointers(prev_layer, next_layer);

    // Add "previous" layer to list of parents
    if(this->m_prev_layer != NULL) {
      add_parent(this->m_prev_layer);
    }

    // Make the first parent layer the "previous" layer
    this->m_prev_layer = m_parents.front();

  }

  void setup_gpu() {
    transform::setup_gpu();
  #ifndef __LIB_CUDNN
    throw lbann_exception("sum_layer: cuDNN not detected");
  #else

    // Copy backward propagation output from GPUs if a parent layer is
    // not using GPU implementation
    for(size_t i=1; i<m_parents.size(); ++i) {
      if(!m_parents[i]->using_gpus()) {
        m_copy_bp_output_from_gpus = true;
      }
    }

  #endif // #ifndef __LIB_CUDNN
  }

  protected:

  void fp_compute() {
    if(this->m_using_gpus) {
      fp_compute_cudnn();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() {
    if(this->m_using_gpus) {
  #ifndef __LIB_CUDNN
      throw lbann_exception("sum_layer: cuDNN not detected");
  #else
      this->m_cudnn->copy_on_gpus(this->m_error_signal_d,
                                  this->m_prev_error_signal_d,
                                  this->m_num_neurons,
                                  this->m_mini_batch_size_per_gpu);
  #endif // __LIB_CUDNN
    }
    else {
      El::LockedView(*this->m_error_signal_v, *this->m_prev_error_signal);
    }
  }

  void fp_compute_cudnn() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("sum_layer: cuDNN not detected");
  #else

    // Useful constant
    const DataType one = 1;

    // Copy error signal from first child layer
    this->m_cudnn->copy_on_gpus(this->m_activations_d,
                                this->m_prev_activations_d,
                                this->m_num_prev_neurons,
                                this->m_mini_batch_size_per_gpu);

    // Iterate through child layers
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(size_t parent_index=1; parent_index<m_parents.size(); ++parent_index) {
      const Layer* parent = m_parents[parent_index];

      // Get child error signal on GPUs
      if(parent->using_gpus()) {
        parent->get_gpu_fp_output(this->m_prev_activations_d, this);
      }
      else {
        parent->get_fp_output(*this->m_prev_activations, this);
        this->m_cudnn->scatter_to_gpus(this->m_prev_activations_d,
                                       this->m_prev_activations->LockedMatrix(),
                                       this->m_mini_batch_size_per_gpu);
      }

      // Add child error signal to this layer's error signal
      for(int i=0; i<num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                   this->m_cudnn->get_stream(i)));
        CHECK_CUDNN(cudnnAddTensor(this->m_cudnn->get_handle(i),
                                   &one,
                                   this->m_prev_neurons_cudnn_desc,
                                   this->m_prev_activations_d[i],
                                   &one,
                                   this->m_neurons_cudnn_desc,
                                   this->m_activations_d[i]));
      }

    }

  #endif // #ifndef __LIB_CUDNN
  }

  void fp_compute_cpu() {
    El::Copy(*this->m_prev_activations, *this->m_activations_v);
    for(size_t i=1; i<m_parents.size(); ++i) {
      m_parents[i]->get_fp_output(*this->m_prev_activations, this);
      El::Axpy(DataType(1),
               *this->m_prev_activations,
               *this->m_activations_v);
    }
  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_SUM_HPP_INCLUDED
