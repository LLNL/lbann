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
// concatenation.hpp - Concatenation layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_CONCATENATION_HPP_INCLUDED
#define LBANN_LAYER_CONCATENATION_HPP_INCLUDED

#include <vector>
#include "lbann/base.hpp"
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/// Concatenation layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class concatenation_layer : public transform {
 private:

  /** Tensor dimension to concatenate. */
  int m_concatenation_axis;
  /** Concatenation points for each parent layer. */
  std::vector<int> m_concatenation_points;

  /** View of back prop output, as seen by parent layers. */
  AbsDistMat* m_bp_output;
  /** View into an input tensor slice.
   *  Used in forward and backward propagation.
   */
  AbsDistMat* m_input_slice_v;
  /** View into an output tensor slice.
   *  Used in forward and backward propagation.
   */
  AbsDistMat* m_output_slice_v;

 public:
  /// Constructor
  concatenation_layer(int index,
                      lbann_comm *comm,
                      std::vector<const Layer*> parents,
                      int concatenation_axis,
                      cudnn::cudnn_manager *cudnn = NULL)
    : transform(index, comm),
      m_concatenation_axis(concatenation_axis) {

    // Setup the data distribution
    initialize_distributed_matrices();

    // Initialize list of parents
    for(size_t i=0; i<parents.size(); ++i) {
      push_back_parent(parents[i]);
    }

  #ifdef __LIB_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN

  }

  concatenation_layer(const concatenation_layer& other) :
    transform(other) {
    m_bp_output = other.m_bp_output->Copy();
    m_input_slice_v = other.m_input_slice_v->Copy();
    m_output_slice_v = other.m_output_slice_v->Copy();
  }

  concatenation_layer& operator=(const concatenation_layer& other) {
    transform::operator=(other);
    if(m_bp_output)      delete m_bp_output;
    if(m_input_slice_v)  delete m_input_slice_v;
    if(m_output_slice_v) delete m_output_slice_v;
    m_bp_output = other.m_bp_output->Copy();
    m_input_slice_v = other.m_input_slice_v->Copy();
    m_output_slice_v = other.m_output_slice_v->Copy();
  }

  ~concatenation_layer() {
    delete m_bp_output;
    delete m_input_slice_v;
    delete m_output_slice_v;

  #ifdef __LIB_CUDNN
    // GPU memory for error signal is a copy of previous layer's error signal
    this->m_error_signal_d.clear();
  #endif // __LIB_CUDNN

  }

  concatenation_layer* copy() const { return new concatenation_layer(*this); }

  std::string get_name() const { return "concatenation"; }

  virtual inline void initialize_distributed_matrices();
  virtual data_layout get_data_layout() const { return T_layout; }

  void push_back_parent(const Layer *parent) {

    // Check if parent layer is null pointer
    if(parent == NULL) {
      if(m_comm->am_world_master()) {
        std::cerr << "concatenation_layer: could not add parent layer since pointer is null" << "\n";
      }
      return;
    }

    // Add parent layer if it isn't in list of parents
    auto parent_pos = std::find(m_parents.begin(), m_parents.end(), parent);
    if(parent_pos == m_parents.end()) {
      m_parents.push_back(parent);
    }
    else {
      throw lbann_exception("concatenation_layer: could not add parent layer since it is already in list of parents");
    }

  }

  void pop_back_parent() {
    if(m_parents.empty()) {
      throw lbann_exception("concatenation_layer: could not remove parent since this layer has no parents");
    }
    m_parents.pop_back();
    m_concatenation_points.pop_back();
  }

  void setup_pointers(const Layer *prev_layer, const Layer *next_layer) {
    transform::setup_pointers(prev_layer, next_layer);

    // Add "previous" layer to list of parents
    if(this->m_prev_layer != NULL) {
      push_back_parent(this->m_prev_layer);
    }

    // Make the first parent layer the "previous" layer
    this->m_prev_layer = m_parents.front();

  }

  void setup_dims() {

    // Initialize previous layer dimensions with first parent layer
    transform::setup_dims();

    // Check if concatenation axis is valid
    if(m_concatenation_axis < 0
       || m_concatenation_axis >= this->m_num_neuron_dims) {
      throw lbann_exception("concatenation_layer: invalid concatenation axis");
    }

    // Get concatenation axis indices corresponding to each parent layer
    m_concatenation_points.empty();
    m_concatenation_points.push_back(0);
    m_concatenation_points.push_back(this->m_neuron_dims[m_concatenation_axis]);
    for(size_t i=1; i<m_parents.size(); ++i) {

      // Get parent layer dimensions
      std::vector<int> parent_dims = m_parents[i]->fp_output_dims(this);

      // Check if parent layer has valid dimensions
      if((int) parent_dims.size() != this->m_num_neuron_dims) {
        throw lbann_exception("concatenation_layer: parent layer has invalid number of dimensions");
      }
      for(size_t d=0; d<parent_dims.size(); ++d) {
        if((int) d != m_concatenation_axis
           && this->m_neuron_dims[d] != parent_dims[d]) {
          throw lbann_exception("concatenation_layer: parent layer has invalid dimensions");
        }
      }

      // Get concatentation axis upper bound for parent layer
      m_concatenation_points.push_back(m_concatenation_points.back()
                                       + parent_dims[m_concatenation_axis]);

    }

    // Update neuron dimensions
    this->m_neuron_dims[m_concatenation_axis] = m_concatenation_points.back();
    this->m_num_neurons = std::accumulate(m_neuron_dims.begin(),
                                          m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

  }

  void setup_gpu() {
    transform::setup_gpu();
  #ifndef __LIB_CUDNN
    throw lbann_exception("concatenation_layer: cuDNN not detected");
  #else

    // Copy backward propagation output from GPUs if a parent layer is
    // not using GPU implementation
    for(size_t i=1; i<m_parents.size(); ++i) {
      if(!m_parents[i]->using_gpus()) {
        m_copy_bp_output_from_gpus = true;
      }
    }

    // Allocate workspace if needed
    if(m_copy_bp_output_from_gpus) {
      int max_slice_dim = 0;
      for(size_t parent_index=1; parent_index<m_parents.size(); ++parent_index) {
        if(!m_parents[parent_index]->using_gpus()) {
          max_slice_dim = std::max(max_slice_dim,
                                   m_concatenation_points[parent_index+1]
                                   - m_concatenation_points[parent_index]);
        }
      }
      int max_slice_size = (this->m_num_neurons
                            / this->m_neuron_dims[m_concatenation_axis]
                            * max_slice_dim);
      size_t required_work_space = (max_slice_size
                                    * this->m_mini_batch_size_per_gpu
                                    * sizeof(DataType));
      for(int i=0; i<this->m_cudnn->get_num_gpus(); ++i) {
        if(required_work_space > this->m_cudnn->get_work_space_size(i)) {
          this->m_cudnn->set_work_space_size(i, required_work_space);
        }
      }
    }

  #endif // #ifndef __LIB_CUDNN
  }

  protected:

  void fp_compute() {
    if(this->m_using_gpus) {
      fp_compute_gpu();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() {
    if(this->m_using_gpus) {
  #ifndef __LIB_CUDNN
      throw lbann_exception("concatenation_layer: cuDNN not detected");
  #else
      this->m_cudnn->copy_on_gpus(this->m_error_signal_d,
                                  this->m_prev_error_signal_d,
                                  this->m_num_neurons,
                                  this->m_mini_batch_size_per_gpu);
  #endif // __LIB_CUDNN
    }
    else {
      El::LockedView(*this->m_error_signal_v, *this->m_prev_error_signal_v);
    }
  }

  void fp_compute_gpu() {
  #ifndef __LIB_CUDNN
    throw lbann_exception("concatenation_layer: cuDNN not detected");
  #else

    // Split the activations tensor into slices of width 1 along the
    // concatenation axis
    const int num_slices
      = std::accumulate(this->m_neuron_dims.begin(),
                        this->m_neuron_dims.begin() + m_concatenation_axis,
                        1,
                        std::multiplies<int>());
    const int slice_unit_size
      = std::accumulate(this->m_neuron_dims.begin() + m_concatenation_axis + 1,
                        this->m_neuron_dims.end(),
                        1,
                        std::multiplies<int>());
    const int output_slice_dim = this->m_neuron_dims[m_concatenation_axis];
    const int output_slice_size = output_slice_dim * slice_unit_size;

    // Copy entries in each child to error signal tensor
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(size_t parent_index = 0; parent_index < m_parents.size(); ++parent_index) {
      const Layer* parent = m_parents[parent_index];

      // Get child error signal on GPUs
      std::vector<DataType*> input;
      if(parent_index == 0) {
        input = this->m_prev_activations_d;
      }
      else {
        std::vector<void*> work_spaces = this->m_cudnn->get_work_spaces();
        for(int i=0; i<num_gpus; ++i) {
          input.push_back((DataType*) work_spaces[i]);
        }
        if(parent->using_gpus()) {
          parent->get_gpu_fp_output(input, this);
        }
        else {
          parent->get_fp_output(*this->m_prev_activations, this);
          this->m_cudnn->scatter_to_gpus(input,
                                         this->m_prev_activations->LockedMatrix(),
                                         this->m_mini_batch_size_per_gpu);
        }
      }

      // Split previous error signal tensor into slices
      const int input_slice_dim = m_concatenation_points[parent_index+1] - m_concatenation_points[parent_index];
      const int input_slice_size = input_slice_dim * slice_unit_size;
      const int input_size = num_slices * input_slice_size;
      const int slice_offset = m_concatenation_points[parent_index] * slice_unit_size;

      // Copy slices from previous error signal tensor into error signal tensor
      for(int slice = 0; slice < num_slices; ++slice) {
        std::vector<DataType*> input_slice(num_gpus), output_slice(num_gpus);
        for(int i = 0; i < num_gpus; ++i) {
          input_slice[i] = input[i] + slice * input_slice_size;
          output_slice[i] = this->m_activations_d[i] + slice * output_slice_size + slice_offset;
        }
        this->m_cudnn->copy_on_gpus(output_slice,
                                    input_slice,
                                    input_slice_size,
                                    this->m_mini_batch_size_per_gpu,
                                    input_size,
                                    this->m_num_neurons);
      }

    }
    
  #endif // #ifndef __LIB_CUDNN
  }

  void fp_compute_cpu() {

    // Split the neuron tensor into slices of width 1 along the
    // concatenation axis
    const int num_slices
      = std::accumulate(this->m_neuron_dims.begin(),
                        this->m_neuron_dims.begin() + m_concatenation_axis,
                        1,
                        std::multiplies<int>());
    const int slice_unit_size
      = std::accumulate(this->m_neuron_dims.begin() + m_concatenation_axis + 1,
                        this->m_neuron_dims.end(),
                        1,
                        std::multiplies<int>());
    const int output_slice_dim = this->m_neuron_dims[m_concatenation_axis];
    const int output_slice_size = output_slice_dim * slice_unit_size;

    // Copy entries in each parent to neuron tensor
    for(size_t i = 0; i < m_parents.size(); ++i) {

      // Split previous neuron tensor into slices
      m_parents[i]->get_fp_output(*this->m_prev_activations, this);
      const int input_slice_dim = m_concatenation_points[i+1] - m_concatenation_points[i];
      const int input_slice_size = input_slice_dim * slice_unit_size;
      const int slice_offset_start = m_concatenation_points[i] * slice_unit_size;
      const int slice_offset_end = m_concatenation_points[i+1] * slice_unit_size;

      // Copy slices from previous neuron tensor into neuron tensor
      for(int slice = 0; slice < num_slices; ++slice) {
        El::LockedView(*m_input_slice_v,
                       *this->m_prev_activations,
                       El::IR(slice * input_slice_size,
                              (slice+1) * input_slice_size),
                       El::ALL);
        El::View(*m_output_slice_v,
                 *this->m_activations_v,
                 El::IR(slice * output_slice_size + slice_offset_start,
                        slice * output_slice_size + slice_offset_end),
                 El::ALL);
        El::Copy(*m_input_slice_v, *m_output_slice_v);
      }

    }

  }

  void get_bp_output(AbsDistMat& bp_output, const Layer* prev_layer) const {

    // Check if input is in the list of parent layers
    const int parent_index = (std::find(m_parents.begin(),
                                        m_parents.end(),
                                        prev_layer)
                              - m_parents.begin());
    if(parent_index >= (int) m_parents.size()) {
      transform::get_bp_output(bp_output, prev_layer);
    }

    // Split the error signal tensor into slices of width 1 along the
    // concatenation axis
    const int num_slices
      = std::accumulate(this->m_neuron_dims.begin(),
                        this->m_neuron_dims.begin() + m_concatenation_axis,
                        1,
                        std::multiplies<int>());
    const int slice_unit_size
      = std::accumulate(this->m_neuron_dims.begin() + m_concatenation_axis + 1,
                        this->m_neuron_dims.end(),
                        1,
                        std::multiplies<int>());
    const int input_slice_dim = this->m_neuron_dims[m_concatenation_axis];
    const int output_slice_dim = m_concatenation_points[parent_index+1] - m_concatenation_points[parent_index];
    const int input_slice_size = input_slice_dim * slice_unit_size;
    const int output_slice_size = output_slice_dim * slice_unit_size;
    const int slice_offset_start = m_concatenation_points[parent_index] * slice_unit_size;
    const int slice_offset_end = m_concatenation_points[parent_index+1] * slice_unit_size;

    if(num_slices == 1
       && m_error_signal_v->DistData() == bp_output.DistData()) {
      // Return view of error signal slice
      El::LockedView(bp_output,
                     *this->m_error_signal_v,
                     El::IR(slice_offset_start, slice_offset_end),
                     El::ALL);
    }
    else {
      // Copy slices from error signal tensor into output
      bp_output.Empty(false);
      bp_output.Resize(output_slice_size, m_error_signal_v->Width());
      AbsDistMat* output_slice_v
        = bp_output.Construct(bp_output.Grid(), bp_output.Root());
      for(int slice = 0; slice < num_slices; ++slice) {
        El::LockedView(*m_input_slice_v,
                       *this->m_error_signal_v,
                       El::IR(slice * input_slice_size + slice_offset_start,
                              slice * input_slice_size + slice_offset_end),
                       El::ALL);
        El::View(*output_slice_v,
                 bp_output,
                 El::IR(slice * output_slice_size,
                        (slice+1) * output_slice_size),
                 El::ALL);
        El::Copy(*m_input_slice_v, *output_slice_v);
      }
      delete output_slice_v;
    }

  }

  #ifdef __LIB_CUDNN
  void get_gpu_bp_output(std::vector<DataType*>& bp_output, const Layer* prev_layer) const {

    // Check if input is in the list of child layers
    const int parent_index = (std::find(m_parents.begin(),
                                        m_parents.end(),
                                        prev_layer)
                              - m_parents.begin());
    if(parent_index >= (int) m_parents.size()) {
      transform::get_gpu_bp_output(bp_output, prev_layer);
    }

    // Split the error signal tensor into slices of width 1 along the
    // concatenation axis
    const int num_slices
      = std::accumulate(this->m_neuron_dims.begin(),
                        this->m_neuron_dims.begin() + m_concatenation_axis,
                        1,
                        std::multiplies<int>());
    const int slice_unit_size
      = std::accumulate(this->m_neuron_dims.begin() + m_concatenation_axis + 1,
                        this->m_neuron_dims.end(),
                        1,
                        std::multiplies<int>());
    const int input_slice_dim = this->m_neuron_dims[m_concatenation_axis];
    const int output_slice_dim = m_concatenation_points[parent_index+1] - m_concatenation_points[parent_index];
    const int input_slice_size = input_slice_dim * slice_unit_size;
    const int output_slice_size = output_slice_dim * slice_unit_size;
    const int output_size = num_slices * output_slice_size;
    const int slice_offset = m_concatenation_points[parent_index] * slice_unit_size;
    
    // Copy slices from previous activations tensor into output
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int slice = 0; slice < num_slices; ++slice) {
      std::vector<DataType*> input_slice(num_gpus), output_slice(num_gpus);
      for(int i = 0; i < num_gpus; ++i) {
        input_slice[i] = this->m_error_signal_d[i] + slice * input_slice_size + slice_offset;
        output_slice[i] = bp_output[i] + slice * output_slice_size;
      }
      this->m_cudnn->copy_on_gpus(output_slice,
                                  input_slice,
                                  output_slice_size,
                                  this->m_mini_batch_size_per_gpu,
                                  this->m_num_neurons,
                                  output_size);
    }

  }
  #endif // __LIB_CUDNN

};


/// Matrices should be in MC,MR distributions
template<> inline void concatenation_layer<data_layout::MODEL_PARALLEL>::initialize_distributed_matrices() {
  transform::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
  m_bp_output = new DistMat(this->m_comm->get_model_grid());
  m_input_slice_v = new DistMat(this->m_comm->get_model_grid());
  m_output_slice_v = new DistMat(this->m_comm->get_model_grid());
}

/// Matrices should be in Star,VC distributions
template<> inline void concatenation_layer<data_layout::DATA_PARALLEL>::initialize_distributed_matrices() {
  transform::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
  m_bp_output = new StarVCMat(this->m_comm->get_model_grid());
  m_input_slice_v = new StarVCMat(this->m_comm->get_model_grid());
  m_output_slice_v = new StarVCMat(this->m_comm->get_model_grid());
}

} // namespace lbann

#endif // LBANN_LAYER_CONCATENATION_HPP_INCLUDED
