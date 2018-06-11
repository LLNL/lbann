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

#ifndef LBANN_LAYERS_RECONSTRUCTION_HPP_INCLUDED
#define LBANN_LAYERS_RECONSTRUCTION_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/layers/io/target/generic_target_layer.hpp"
#include "lbann/models/model.hpp"
#include <string>
#include "lbann/utils/random.hpp"

namespace lbann {
template <data_layout T_layout, El::Device Dev>
class reconstruction_layer : public generic_target_layer {
 private:

 public:
  reconstruction_layer(lbann_comm *comm)
    :  generic_target_layer(comm) {
  }

  reconstruction_layer* copy() const override {
    throw lbann_exception("reconstruction_layer can't be copied");
    return nullptr;
  }

  std::string get_type() const override { return "reconstruction"; }

  std::string get_description() const override {
    return std::string{} + " reconstruction_layer " +
                           " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  data_layout get_data_layout() const override { return T_layout; }

  El::Device get_device_allocation() const override { return Dev; }

 protected:

  void fp_compute() override {}

  void bp_compute() override {}

  virtual AbsDistMat& get_ground_truth() { return get_prev_activations(1); }
  virtual const AbsDistMat& get_ground_truth() const { return get_prev_activations(1); }

public:

  void summarize_stats(lbann_summary& summarizer, int step) override {
    std::string tag = this->m_name + "/ReconstructionCost";
    execution_mode mode = this->m_model->get_execution_mode();
    summarizer.reduce_scalar(tag, this->m_model->get_objective_function()->get_mean_value(mode), step);
    // Skip target layer (for now).
    //    io_layer::summarize_stats(summarizer, step);
  }
};

}

#endif  // LBANN_LAYERS_RECONSTRUCTION_HPP_INCLUDED
