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
            int mini_batch_size,
            std::vector<const Layer*> parents,
            cudnn::cudnn_manager *cudnn = NULL)
    : transform(index, comm, mini_batch_size) {

    // Setup the data distribution
    initialize_distributed_matrices();

    // Initialize list of parents
    for(size_t i=0; i<parents.size(); ++i) {
      add_parent(parents[i]);
    }

  }

  sum_layer(const sum_layer&) = default;
  sum_layer& operator=(const sum_layer&) = default;
  ~sum_layer() = default;

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
      throw lbann_exception("sum_layer: could not add parent layer since it is already in list of parents");
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

  protected:

  void fp_compute() {
    if(m_parents.size() == 1) {
      El::View(*this->m_activations, *this->m_prev_activations);
    }
    else {
      El::Copy(*this->m_prev_activations, *this->m_activations);
      for(size_t i=1; i<m_parents.size(); ++i) {
        El::Axpy(DataType(1),
                 m_parents[i]->fp_output(this),
                 *this->m_activations);
      }
    }
  }

  void bp_compute() {
    El::View(*this->m_error_signal, *this->m_prev_error_signal);
  }

  const AbsDistMat& fp_input(const Layer* prev_layer) const {
  #ifdef LBANN_DEBUG
    if(prev_layer != NULL
       && (std::find(m_parents.begin(), m_parents.end(), prev_layer)
           == m_parents.end())) {
      throw lbann_exception("sum_layer: unexpected previous layer");
    }
  #endif // LBANN_DEBUG
    return *m_prev_activations;
  }

  const AbsDistMat& bp_output(const Layer* prev_layer) const {
  #ifdef LBANN_DEBUG
    if(prev_layer != NULL
       && (std::find(m_parents.begin(), m_parents.end(), prev_layer)
           == m_parents.end())) {
      throw lbann_exception("sum_layer: unexpected previous layer");
    }
  #endif // LBANN_DEBUG
    return *m_error_signal;
  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_SUM_HPP_INCLUDED
