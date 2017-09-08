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
// slice.hpp - Slice layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_SLICE_HPP_INCLUDED
#define LBANN_LAYER_SLICE_HPP_INCLUDED

#include <vector>
#include "lbann/base.hpp"
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/// Slice layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class slice_layer : public transform {
 private:

  /** List of child layers. */
  std::vector<const Layer*> m_children;
  /** Tensor dimension to slice. */
  int m_slice_axis;
  /** Slice points for each child layer. */
  std::vector<int> m_slice_points;

  /** View of back prop output, as seen by child layers. */
  AbsDistMat* m_fp_output;
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
  slice_layer(int index,
              lbann_comm *comm,
              std::vector<const Layer*> children,
              int slice_axis,
              std::vector<int> slice_points,
              cudnn::cudnn_manager *cudnn = NULL)
    : transform(index, comm),
      m_slice_axis(slice_axis) {

    // Setup the data distribution
    initialize_distributed_matrices();

    // Check that number of slice points is valid
    if(!children.empty() && children.size()-1 != slice_points.size()) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: slice_layer:  number of slice points should be one less than number of children";
      throw lbann_exception(err.str());
    }

    // Initialize list of children
    if(!children.empty()) {
      push_back_child(children.front(), 0);
    }
    for(size_t i=1; i<children.size(); ++i) {
      push_back_child(children[i], slice_points[i-1]);
    }

  #ifdef __LIB_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN

  }

  slice_layer(const slice_layer& other) :
    transform(other) {
    m_fp_output = other.m_fp_output->Copy();
    m_input_slice_v = other.m_input_slice_v->Copy();
    m_output_slice_v = other.m_output_slice_v->Copy();
  }

  slice_layer& operator=(const slice_layer& other) {
    transform::operator=(other);
    if(m_fp_output)      delete m_fp_output;
    if(m_input_slice_v)  delete m_input_slice_v;
    if(m_output_slice_v) delete m_output_slice_v;
    m_fp_output = other.m_fp_output->Copy;
    m_input_slice_v = other.m_input_slice_v->Copy();
    m_output_slice_v = other.m_output_slice_v->Copy();
  }

  ~slice_layer() {
    delete m_fp_output;
    delete m_input_slice_v;
    delete m_output_slice_v;

  #ifdef __LIB_CUDNN
    // GPU memory for activations is a copy of previous layer's activations
    this->m_activations_d.clear();
  #endif // __LIB_CUDNN

  }

  /** Returns description of ctor params */
  std::string get_description() const {
    std::stringstream s;
    s << " slice; slice_axis: "
      << m_slice_axis << " children: ";
    for (size_t h=0; h<this->m_children.size(); h++) {
      s << this->m_children[h]->get_index() << " " << this->m_children[h]->get_name() << " ";
    }
    s << " slice_points: ";
    for (size_t h=0; h<this->m_slice_points.size(); h++) {
      s << this->m_slice_points[h] << " ";
    }
    s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    return s.str();
  }

  slice_layer* copy() const { return new slice_layer(*this); }

  std::string get_name() const { return "slice"; }

  virtual inline void initialize_distributed_matrices();
  virtual data_layout get_data_layout() const { return T_layout; }

  void push_back_child(const Layer *child, int slice_point) {
    std::stringstream err;

    // Check if child layer is null pointer
    if(child == NULL) {
      if(m_comm->am_world_master()) {
        err << __FILE__ << " " << __LINE__ << " :: slice_layer: could not add child layer since pointer is null";
        throw lbann_exception(err.str());
      }
      return;
    }

    // Add first child
    if(m_children.empty()) {
      if(m_comm->am_world_master()) {
        if(slice_point > 0) {
          err << __FILE__ << " " << __LINE__ << " :: slice_layer: first child should have a slice point of zero";
          throw lbann_exception(err.str());
        }
      }
      m_children.push_back(child);
      m_slice_points.push_back(0);
    }

    // Add subsequent children
    else {
      auto child_pos = std::find(m_children.begin(), m_children.end(), child);
      if(child_pos != m_children.end()) {
        err << __FILE__ << " " << __LINE__ << " :: slice_layer:  number of slice points should be one less than number of children";
        throw lbann_exception(err.str());
      }
      if(slice_point <= m_slice_points.back()) {
        err << __FILE__ << " " << __LINE__ << " :: slice_layer:  invalid slice point";
        throw lbann_exception(err.str());
      }
      m_children.push_back(child);
      m_slice_points.push_back(slice_point);
    }

  }

  void pop_back_child() {
    if(m_children.empty()) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: slice_layer: could not remove child since this layer has no children";
      throw lbann_exception(err.str());
    }
    m_children.pop_back();
    m_slice_points.pop_back();
  }

  void setup_pointers(const Layer *prev_layer, const Layer *next_layer) {
    transform::setup_pointers(prev_layer, next_layer);
    std::stringstream err;

    // Error if "next" layer isn't already in list of children
    if(next_layer != NULL
       && (std::find(m_children.begin(), m_children.end(), this->m_next_layer)
           == m_children.end())) {
      err << __FILE__ << " " << __LINE__ << " :: slice_layer: can not add child layer during setup phase";
      throw lbann_exception(err.str());
    }
    if(m_children.empty()) {
      throw lbann_exception("slice_layer: can not setup layer since it has no children");
      err << __FILE__ << " " << __LINE__ << " :: slice_layer: can not setup layer since it has no children";
      throw lbann_exception(err.str());
    }

    // Make the first child layer the "next" layer
    this->m_next_layer = m_children.front();
  }

  void setup_dims() {
    std::stringstream err;

    // Initialize previous neuron tensor dimensions
    transform::setup_dims();

    // Check if slice axis and slice points are valid
    if(m_slice_axis < 0 || m_slice_axis >= this->m_num_neuron_dims) {
      err << __FILE__ << " " << __LINE__ << " :: slice_layer: invalid slice axis";
      throw lbann_exception(err.str());
    }
    if(m_slice_points.back() >= this->m_neuron_dims[m_slice_axis] - 1) {
      err << __FILE__ << " " << __LINE__ << " :: slice_layer: slice points are greater than slice axis dimensions";
      throw lbann_exception(err.str());
    }

    // Add slice axis dimension to slice point list
    m_slice_points.push_back(this->m_neuron_dims[m_slice_axis]);

  }

  void setup_gpu() {
    transform::setup_gpu();
  #ifndef __LIB_CUDNN
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: slice_layer: cuDNN not detected";
    throw lbann_exception(err.str());
  #else

    // Copy forward propagation output from GPUs if a child layer is
    // not using GPU implementation
    for(size_t i=1; i<m_children.size(); ++i) {
      if(!m_children[i]->using_gpus()) {
        m_copy_fp_output_from_gpus = true;
      }
    }

    // Allocate workspace if needed
    if(m_copy_fp_output_from_gpus) {
      int max_slice_dim = 0;
      for(size_t child_index=1; child_index<m_children.size(); ++child_index) {
        if(!m_children[child_index]->using_gpus()) {
          max_slice_dim = std::max(max_slice_dim,
                                   m_slice_points[child_index+1]
                                   - m_slice_points[child_index]);
        }
      }
      int max_slice_size = (this->m_num_prev_neurons
                            / this->m_prev_neuron_dims[m_slice_axis]
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
  #ifndef __LIB_CUDNN
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: slice_layer: cuDNN not detected";
    throw lbann_exception(err.str());
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
      bp_compute_gpu();
    } else {
      bp_compute_cpu();
    }
  }

  void bp_compute_gpu() {
  #ifndef __LIB_CUDNN
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: slice_layer: cuDNN not detected";
    throw lbann_exception(err.str());
  #else

    // Split the error signal tensor into slices of width 1 along the
    // slice axis
    const int num_slices
      = std::accumulate(this->m_prev_neuron_dims.begin(),
                        this->m_prev_neuron_dims.begin() + m_slice_axis,
                        1,
                        std::multiplies<int>());
    const int slice_unit_size
      = std::accumulate(this->m_prev_neuron_dims.begin() + m_slice_axis + 1,
                        this->m_prev_neuron_dims.end(),
                        1,
                        std::multiplies<int>());
    const int output_slice_dim = this->m_prev_neuron_dims[m_slice_axis];
    const int output_slice_size = output_slice_dim * slice_unit_size;

    // Copy entries in each child to error signal tensor
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(size_t child_index = 0; child_index < m_children.size(); ++child_index) {
      const Layer* child = m_children[child_index];

      // Get child error signal on GPUs
      std::vector<DataType*> input;
      if(child_index == 0) {
        input = this->m_prev_error_signal_d;
      }
      else {
        std::vector<void*> work_spaces = this->m_cudnn->get_work_spaces();
        for(int i=0; i<num_gpus; ++i) {
          input.push_back((DataType*) work_spaces[i]);
        }
        if(child->using_gpus()) {
          child->get_gpu_bp_output(input, this);
        }
        else {
          child->get_bp_output(*this->m_prev_error_signal, this);
          this->m_cudnn->scatter_to_gpus(input,
                                         this->m_prev_error_signal->LockedMatrix(),
                                         this->m_mini_batch_size_per_gpu);
        }
      }

      // Split previous error signal tensor into slices
      const int input_slice_dim = m_slice_points[child_index+1] - m_slice_points[child_index];
      const int input_slice_size = input_slice_dim * slice_unit_size;
      const int input_size = num_slices * input_slice_size;
      const int slice_offset = m_slice_points[child_index] * slice_unit_size;

      // Copy slices from previous error signal tensor into error signal tensor
      for(int slice = 0; slice < num_slices; ++slice) {
        std::vector<DataType*> input_slice(num_gpus), output_slice(num_gpus);
        for(int i = 0; i < num_gpus; ++i) {
          input_slice[i] = input[i] + slice * input_slice_size;
          output_slice[i] = this->m_error_signal_d[i] + slice * output_slice_size + slice_offset;
        }
        this->m_cudnn->copy_on_gpus(output_slice,
                                    input_slice,
                                    input_slice_size,
                                    this->m_mini_batch_size_per_gpu,
                                    input_size,
                                    this->m_num_prev_neurons);
      }

    }
    
  #endif // #ifndef __LIB_CUDNN
  }

  void bp_compute_cpu() {

    // Split the error signal tensor into slices of width 1 along the
    // slice axis
    const int num_slices
      = std::accumulate(this->m_prev_neuron_dims.begin(),
                        this->m_prev_neuron_dims.begin() + m_slice_axis,
                        1,
                        std::multiplies<int>());
    const int slice_unit_size
      = std::accumulate(this->m_prev_neuron_dims.begin() + m_slice_axis + 1,
                        this->m_prev_neuron_dims.end(),
                        1,
                        std::multiplies<int>());
    const int output_slice_dim = this->m_prev_neuron_dims[m_slice_axis];
    const int output_slice_size = output_slice_dim * slice_unit_size;

    // Copy entries in each child to error signal tensor
    for(size_t i = 0; i < m_children.size(); ++i) {

      // Split previous error signal tensor into slices
      m_children[i]->get_bp_output(*this->m_prev_error_signal, this);
      const int input_slice_dim = m_slice_points[i+1] - m_slice_points[i];
      const int input_slice_size = input_slice_dim * slice_unit_size;
      const int slice_offset_start = m_slice_points[i] * slice_unit_size;
      const int slice_offset_end = m_slice_points[i+1] * slice_unit_size;

      // Copy slices from previous error signal tensor into error signal tensor
      for(int slice = 0; slice < num_slices; ++slice) {
        El::LockedView(*m_input_slice_v,
                       *this->m_prev_error_signal,
                       El::IR(slice * input_slice_size,
                              (slice+1) * input_slice_size),
                       El::ALL);
        El::View(*m_output_slice_v,
                 *this->m_error_signal_v,
                 El::IR(slice * output_slice_size + slice_offset_start,
                        slice * output_slice_size + slice_offset_end),
                 El::ALL);
        El::Copy(*m_input_slice_v, *m_output_slice_v);
      }

    }

  }

  void get_fp_output(AbsDistMat& fp_output, const Layer* next_layer) const {

    // Check if input is in the list of child layers
    const int child_index = (std::find(m_children.begin(),
                                       m_children.end(),
                                       next_layer)
                             - m_children.begin());
    if(child_index >= (int) m_children.size()) {
      transform::get_fp_output(fp_output, next_layer);
    }

    // Split the activations tensor into slices of width 1 along the
    // slice axis
    const int num_slices
      = std::accumulate(this->m_prev_neuron_dims.begin(),
                        this->m_prev_neuron_dims.begin() + m_slice_axis,
                        1,
                        std::multiplies<int>());
    const int slice_unit_size
      = std::accumulate(this->m_prev_neuron_dims.begin() + m_slice_axis + 1,
                        this->m_prev_neuron_dims.end(),
                        1,
                        std::multiplies<int>());
    const int input_slice_dim = this->m_prev_neuron_dims[m_slice_axis];
    const int output_slice_dim = m_slice_points[child_index+1] - m_slice_points[child_index];
    const int input_slice_size = input_slice_dim * slice_unit_size;
    const int output_slice_size = output_slice_dim * slice_unit_size;
    const int slice_offset_start = m_slice_points[child_index] * slice_unit_size;
    const int slice_offset_end = m_slice_points[child_index+1] * slice_unit_size;
    
    if(num_slices == 1
       && m_activations_v->DistData() == fp_output.DistData()) {
      // Return view of activations tensor slice
      El::LockedView(fp_output,
                     *this->m_activations_v,
                     El::IR(slice_offset_start, slice_offset_end),
                     El::ALL);
    }
    else {
      // Copy slices from activations tensor into output
      fp_output.Empty(false);
      fp_output.Resize(output_slice_size, m_activations_v->Width());
      AbsDistMat* output_slice_v
        = fp_output.Construct(fp_output.Grid(), fp_output.Root());
      for(int slice = 0; slice < num_slices; ++slice) {
        El::LockedView(*m_input_slice_v,
                       *this->m_activations_v,
                       El::IR(slice * input_slice_size + slice_offset_start,
                              slice * input_slice_size + slice_offset_end),
                       El::ALL);
        El::View(*output_slice_v,
                 fp_output,
                 El::IR(slice * output_slice_size,
                        (slice+1) * output_slice_size),
                 El::ALL);
        El::Copy(*m_input_slice_v, *output_slice_v);
      }
      delete output_slice_v;
    }

  }

  #ifdef __LIB_CUDNN
  void get_gpu_fp_output(std::vector<DataType*>& fp_output, const Layer* next_layer) const {

    // Check if input is in the list of child layers
    const int child_index = (std::find(m_children.begin(),
                                       m_children.end(),
                                       next_layer)
                             - m_children.begin());
    if(child_index >= (int) m_children.size()) {
      transform::get_gpu_fp_output(fp_output, next_layer);
    }

    // Split the activations tensor into slices of width 1 along the
    // slice axis
    const int num_slices
      = std::accumulate(this->m_prev_neuron_dims.begin(),
                        this->m_prev_neuron_dims.begin() + m_slice_axis,
                        1,
                        std::multiplies<int>());
    const int slice_unit_size
      = std::accumulate(this->m_prev_neuron_dims.begin() + m_slice_axis + 1,
                        this->m_prev_neuron_dims.end(),
                        1,
                        std::multiplies<int>());
    const int input_slice_dim = this->m_prev_neuron_dims[m_slice_axis];
    const int output_slice_dim = m_slice_points[child_index+1] - m_slice_points[child_index];
    const int input_slice_size = input_slice_dim * slice_unit_size;
    const int output_slice_size = output_slice_dim * slice_unit_size;
    const int output_size = num_slices * output_slice_size;
    const int slice_offset = m_slice_points[child_index] * slice_unit_size;
    
    // Copy slices from previous activations tensor into output
    const int num_gpus = this->m_cudnn->get_num_gpus();
    for(int slice = 0; slice < num_slices; ++slice) {
      std::vector<DataType*> input_slice(num_gpus), output_slice(num_gpus);
      for(int i = 0; i < num_gpus; ++i) {
        input_slice[i] = this->m_activations_d[i] + slice * input_slice_size + slice_offset;
        output_slice[i] = fp_output[i] + slice * output_slice_size;
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

  const std::vector<int> fp_output_dims(const Layer* next_layer) const {

    // Return all neurons if input is null
    if(next_layer == NULL) {
      return m_neuron_dims;
    }

    // Check if input is in the list of child layers
    const int child_index = (std::find(m_children.begin(),
                                       m_children.end(),
                                       next_layer)
                             - m_children.begin());
    if(child_index >= (int) m_children.size()) {
      return m_neuron_dims;
    }

    // Return slice dimensions
    std::vector<int> neuron_dims = m_neuron_dims;
    neuron_dims[m_slice_axis] = m_slice_points[child_index+1] - m_slice_points[child_index];
    return neuron_dims;

  }

};

/// Matrices should be in MC,MR distributions
template<> inline void slice_layer<data_layout::MODEL_PARALLEL>::initialize_distributed_matrices() {
  transform::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
  m_fp_output = new DistMat(this->m_comm->get_model_grid());
  m_input_slice_v = new DistMat(this->m_comm->get_model_grid());
  m_output_slice_v = new DistMat(this->m_comm->get_model_grid());
}

/// Matrices should be in Star,VC distributions
template<> inline void slice_layer<data_layout::DATA_PARALLEL>::initialize_distributed_matrices() {
  transform::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
  m_fp_output = new StarVCMat(this->m_comm->get_model_grid());
  m_input_slice_v = new StarVCMat(this->m_comm->get_model_grid());
  m_output_slice_v = new StarVCMat(this->m_comm->get_model_grid());
}

} // namespace lbann

#endif // LBANN_LAYER_SLICE_HPP_INCLUDED
