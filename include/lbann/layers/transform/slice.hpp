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

  /// List of child layers
  std::vector<const Layer*> m_children;
  /// Tensor dimension to slice
  int m_slice_axis;
  /// Slice points for each child layer
  std::vector<int> m_slice_points;

 public:
  /// Constructor
  slice_layer(int index,
              lbann_comm *comm,
              int mini_batch_size,
              std::vector<const Layer*> children,
              int slice_axis,
              std::vector<int> slice_points,
              cudnn::cudnn_manager *cudnn = NULL)
    : transform(index, comm, mini_batch_size),
      m_slice_axis(slice_axis) {

    // Setup the data distribution
    initialize_distributed_matrices();

    // Initialize list of children
  #if LBANN_DEBUG
    if(!children.empty() && children.size()-1 != slice_points.size()) {
      throw lbann_exception("slice_layer: number of slice points should be one less than number of children");
    }
  #endif
    if(!children.empty()) {
      push_back_child(children.front(), 0);
    }
    for(size_t i=1; i<children.size(); ++i) {
      push_back_child(children[i], slice_points[i-1]);
    }

  }

  slice_layer(const slice_layer&) = default;
  slice_layer& operator=(const slice_layer&) = default;
  ~slice_layer() = default;

  slice_layer* copy() const { return new slice_layer(*this); }

  std::string get_name() const { return "slice"; }

  virtual inline void initialize_distributed_matrices() {
    transform::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void push_back_child(const Layer *child, int slice_point) {

    // Check if child layer is null pointer
    if(child == NULL) {
    #ifdef LBANN_DEBUG
      if(m_comm->am_world_master()) {
        std::cerr << "slice_layer: could not add child layer since pointer is null" << "\n";
      }
    #endif
      return;
    }

    // Add first child
    if(m_children.empty()) {
    #ifdef LBANN_DEBUG
      if(slice_point > 0) {
        std::cerr << "slice_layer: first child should have a slice point of zero" << "\n";
      }
    #endif // LBANN_DEBUG
      m_children.push_back(child);
      m_slice_points.push_back(0);
    }

    // Add subsequent children
    else {
    #ifdef LBANN_DEBUG
      auto child_pos = std::find(m_children.begin(), m_children.end(), child);
      if(child_pos != m_children.end()) {
        throw lbann_exception("slice_layer: child is already in list of children");
      }
      if(slice_point <= m_slice_points.back()) {
        throw lbann_exception("slice_layer: invalid slice point");
      }
    #endif // LBANN_DEBUG
      m_children.push_back(child);
      m_slice_points.push_back(slice_point);
    }

  }

  void pop_back_child() {
  #ifdef LBANN_DEBUG
    if(m_children.empty()) {
      std::cerr << "slice_layer: could not remove child since this layer has no children" << "\n";
    }
  #endif // LBANN_DEBUG
    m_children.pop_back();
    m_slice_points.pop_back();
  }

  void setup_pointers(const Layer *prev_layer, const Layer *next_layer) {
    transform::setup_pointers(prev_layer, next_layer);

  #ifdef LBANN_DEBUG
    // Error if "next" layer isn't already in list of children
    if(next_layer != NULL
       && (std::find(m_children.begin(), m_children.end(), this->m_next_layer)
           == m_children.end())) {
      throw lbann_exception("slice_layer: can not add child layer during setup phase");
    }
    if(m_children.empty()) {
      throw lbann_exception("slice_layer: can not setup layer since it has no children");
    }
  #endif // LBANN_DEBUG

    // Make the first child layer the "next" layer
    this->m_next_layer = m_children.front();

  }

  void setup_dims() {

    // Initialize previous neuron tensor dimensions
    transform::setup_dims();

  #ifdef LBANN_DEBUG
    // Check if slice axis and slice points are valid
    if(m_slice_axis < 0 || m_slice_axis >= this->m_num_neuron_dims) {
      throw lbann_exception("slice_layer: invalid slice axis");
    }
    if(m_slice_points.back() >= this->m_neuron_dims[m_slice_axis] - 1) {
      throw lbann_exception("slice_layer: slice points are greater than slice axis dimensions");
    }
  #endif

    // Add slice axis dimension to slice point list
    m_slice_points.push_back(this->m_neuron_dims[m_slice_axis]);

  }

  protected:

  void fp_compute() {
    El::View(*this->m_activations, *this->m_prev_activations);
  }

  void bp_compute() {

    if(m_slice_axis == 0) {
      const int slice_size = this->m_num_neurons / this->m_neuron_dims[m_slice_axis];
      for(size_t i=0; i<m_children.size(); ++i) {
        auto error_signal_slice
          = El::View(*this->m_error_signal,
                     El::IR(slice_size * m_slice_points[i],
                            slice_size * m_slice_points[i+1]),
                     El::ALL);
        El::Copy(m_children[i]->bp_output(this), error_signal_slice);
      }
    }
    else {
      // TODO: implement general slice layer
      throw lbann_exception("slice_layer: currently only implemented with slice axis 0");
    }

  }

  const AbsDistMat& fp_output(const Layer* next_layer) const {

    // Return all neurons if input is null
    if(next_layer == NULL) {
      return *m_activations;
    }

    // Check if input is in the list of child layers
    const int child_index = (std::find(m_children.begin(),
                                       m_children.end(),
                                       next_layer)
                             - m_children.begin());
    if(child_index >= m_children.size()) {
      throw lbann_exception("slice_layer: unexpected next layer");
    }
    
    if(m_slice_axis == 0) {
      const int slice_size = this->m_num_neurons / this->m_neuron_dims[m_slice_axis];
      return El::LockedView(*this->m_activations,
                            El::IR(slice_size * m_slice_points[child_index],
                                   slice_size * m_slice_points[child_index+1]),
                            El::ALL);
    }
    else {
      // TODO: implement general slice layer
      throw lbann_exception("slice_layer: currently only implemented with slice axis 0");
    }

  }

  const AbsDistMat& bp_input(const Layer* next_layer) const {

    // Return all neurons if input is null
    if(next_layer == NULL) {
      return *m_prev_error_signal;
    }

    // Check if input is in the list of child layers
    const int child_index = (std::find(m_children.begin(),
                                       m_children.end(),
                                       next_layer)
                             - m_children.begin());
    if(child_index >= m_children.size()) {
      throw lbann_exception("slice_layer: unexpected next layer");
    }
    
    if(m_slice_axis == 0) {
      const int slice_size = this->m_num_neurons / this->m_neuron_dims[m_slice_axis];
      return El::LockedView(*this->m_prev_error_signal,
                            El::IR(slice_size * m_slice_points[child_index],
                                   slice_size * m_slice_points[child_index+1]),
                            El::ALL);
    }
    else {
      // TODO: implement general slice layer
      throw lbann_exception("slice_layer: currently only implemented with slice axis 0");
    }

  }  

};

}  // namespace lbann

#endif  // LBANN_LAYER_SLICE_HPP_INCLUDED
