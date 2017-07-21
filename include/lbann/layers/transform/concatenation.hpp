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

  /// List of parent layers
  std::vector<const Layer*> m_parents;
  /// Tensor dimension to concatenate
  int m_concatenation_axis;
  /// Concatenation points for each parent layer
  std::vector<int> m_concatenation_points;

  /// View of forward prop input from parent layers
  AbsDistMat* m_fp_input_v;
  /// View of back prop output, as seen by parent layers
  AbsDistMat* m_bp_output_v;

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

  }

  concatenation_layer(const concatenation_layer& other) :
    transform(other) {
    m_fp_input_v = other.m_fp_input_v->Copy();
    m_bp_output_v = other.m_bp_output_v->Copy();
  }

  concatenation_layer& operator=(const concatenation_layer& other) {
    transform::operator=(other);
    if(m_fp_input_v) {
      delete m_fp_input_v;
    }
    if(m_bp_output_v) {
      delete m_bp_output_v;
    }
    m_fp_input_v = other.m_fp_input_v->Copy;
    m_bp_output_v = other.m_bp_output_v->Copy;
  }

  ~concatenation_layer() {
    delete m_fp_input_v;
    delete m_bp_output_v;
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

  protected:

  void fp_compute() {

    if(m_concatenation_axis == 0) {
      const int channel_size = this->m_num_neurons / this->m_neuron_dims[m_concatenation_axis];
      for(size_t i=0; i<m_parents.size(); ++i) {
        El::View(*m_fp_input_v,
                 *this->m_activations,
                 El::IR(channel_size * m_concatenation_points[i],
                        channel_size * m_concatenation_points[i+1]),
                 El::ALL);
        El::Copy(m_parents[i]->fp_output(this), *m_fp_input_v);
      }
    }
    else {
      // TODO: implement general concatenation layer
      throw lbann_exception("concatenation_layer: currently only implemented with concatenation axis 0");
    }

  }

  void bp_compute() {
    El::LockedView(*this->m_error_signal, *this->m_prev_error_signal);
  }

  const AbsDistMat& fp_input(const Layer* prev_layer) const {
      return *m_prev_activations;
  }  

  const AbsDistMat& bp_output(const Layer* prev_layer) const {

    // Return entire error signal if input is null
    if(prev_layer == NULL) {
      return *m_error_signal;
    }

    // Check if input is in the list of parent layers
    const int parent_index = (std::find(m_parents.begin(),
                                       m_parents.end(),
                                       prev_layer)
                             - m_parents.begin());
    if(parent_index >= (int) m_parents.size()) {
      return *m_error_signal;
    }
    
    if(m_concatenation_axis == 0) {
      const int slice_size = this->m_num_neurons / this->m_neuron_dims[m_concatenation_axis];
      El::LockedView(*m_bp_output_v,
                     *this->m_error_signal,
                     El::IR(slice_size * m_concatenation_points[parent_index],
                            slice_size * m_concatenation_points[parent_index+1]),
                     El::ALL);
      return *m_bp_output_v;
    }
    else {
      // TODO: implement general concatenation layer
      throw lbann_exception("concatenation_layer: currently only implemented with concatenation axis 0");
    }

  }

};

/// Matrices should be in MC,MR distributions
template<> inline void concatenation_layer<data_layout::MODEL_PARALLEL>::initialize_distributed_matrices() {
  transform::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
  m_fp_input_v = new DistMat(this->m_comm->get_model_grid());
  m_bp_output_v = new DistMat(this->m_comm->get_model_grid());
}

/// Matrices should be in Star,VC distributions
template<> inline void concatenation_layer<data_layout::DATA_PARALLEL>::initialize_distributed_matrices() {
  transform::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
  m_fp_input_v = new StarVCMat(this->m_comm->get_model_grid());
  m_bp_output_v = new StarVCMat(this->m_comm->get_model_grid());
}

}  // namespace lbann

#endif  // LBANN_LAYER_CONCATENATION_HPP_INCLUDED
