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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYERS_GENERIC_TARGET_LAYER_HPP_INCLUDED
#define LBANN_LAYERS_GENERIC_TARGET_LAYER_HPP_INCLUDED

#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/io/data_buffers/partitioned_io_buffer.hpp"
#include "lbann/io/data_buffers/distributed_io_buffer.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lbann {
class generic_target_layer : public Layer {
 protected:
  /** Ground truth matrix. */
  AbsDistMat* m_ground_truth;

 public:
  generic_target_layer(lbann_comm *comm)
    : Layer(comm),
      m_ground_truth(nullptr) {
    // Target layers have two parents, the layer that it will feed
    // back the response to, and the input layer where it gets the response
    m_expected_num_parent_layers = 2;
    // Target layers have no children
    m_expected_num_child_layers = 0;
  }

  generic_target_layer(const generic_target_layer& other)
    : Layer(other),
      m_ground_truth(other.m_ground_truth) {
    if (m_ground_truth != nullptr) { m_ground_truth = m_ground_truth->Copy(); }
  }

  generic_target_layer& operator=(const generic_target_layer& other) {
    Layer::operator=(other);

    // Deep copy matrix
    if (m_ground_truth != nullptr) { delete m_ground_truth; }
    m_ground_truth = other.m_ground_truth;
    if (m_ground_truth != nullptr) { m_ground_truth = m_ground_truth->Copy(); }

    return *this;
  }


  ~generic_target_layer() override {
    if (m_ground_truth != nullptr) { delete m_ground_truth; }
  };

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::string s = get_topo_description();
    return std::string {} + " target_layer "
           + " dataLayout: " + this->get_data_layout_string(get_data_layout())
           + " (" + s + ")";
  }

  void setup_matrices(const El::Grid& grid) override {
    Layer::setup_matrices(grid);
    if (m_ground_truth != nullptr) delete m_ground_truth;
    // Inherit the data layout of the input layer
    m_ground_truth = get_prev_activations(1).Copy();
  }

  void fp_setup_data(int mini_batch_size) override {
    Layer::fp_setup_data(mini_batch_size);
  }

  void fp_compute() override {
    Copy(get_prev_activations(1), *m_ground_truth);
  }

  void bp_compute() override {}

  bool update_compute() override {
    return true;
  }

  virtual AbsDistMat& get_prediction() { return get_prev_activations(); }
  virtual const AbsDistMat& get_prediction() const { return get_prev_activations(); }
  virtual AbsDistMat& get_ground_truth() { return *m_ground_truth; }
  virtual const AbsDistMat& get_ground_truth() const { return *m_ground_truth; }

  std::vector<Layer*> get_layer_pointers() override {
    std::vector<Layer*> layers = Layer::get_layer_pointers();
    return layers;
  }

  void set_layer_pointers(std::vector<Layer*> layers) override {
    Layer::set_layer_pointers(layers);
  }
};

}  // namespace lbann

#endif  // LBANN_LAYERS_GENERIC_TARGET_LAYER_HPP_INCLUDED
