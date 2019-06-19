////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
// lbann_callback_io .hpp .cpp - Callback hooks for I/O monitoring
////////////////////////////////////////////////////////////////////////////////

#include <utility>

#include "lbann/callbacks/callback_io.hpp"
#include "lbann/layers/io/input/generic_input_layer.hpp"

namespace lbann {

lbann_callback_io::lbann_callback_io() : lbann_callback() {}

lbann_callback_io::lbann_callback_io(
  std::unordered_set<Layer *> layers) : lbann_callback(), m_layer_indices(std::move(layers)) {}

void lbann_callback_io::on_epoch_end(model *m) {
  lbann_comm *comm = m->get_comm();
  for (Layer *layer : m->get_layers()) {
    if(m_layer_indices.size() == 0
       || m_layer_indices.find(layer) != m_layer_indices.end()) {
      auto *input = (generic_input_layer *) dynamic_cast<generic_input_layer *> (layer);
      if(input != nullptr) {
        std::cout << "Rank " << comm->get_trainer_rank() << "." << comm->get_rank_in_trainer() << " processed "
                  << input->get_num_samples_trained() << " training samples of "
                  << input->get_total_num_training_samples() << " ("
                  << input->get_num_samples_trained() / m->get_epoch() << " per epoch)" << std::endl;
      }
    }
  }
}

void lbann_callback_io::on_test_end(model *m) {
  lbann_comm *comm = m->get_comm();
  for (Layer *layer : m->get_layers()) {
    if(m_layer_indices.size() == 0
       || m_layer_indices.find(layer) != m_layer_indices.end()) {
      auto *input = (generic_input_layer *) dynamic_cast<generic_input_layer *> (layer);
      if(input != nullptr) {
        std::cout << "Rank " << comm->get_trainer_rank() << "." << comm->get_rank_in_trainer() << " processed "
                  << input->get_num_samples_tested() << " test samples of "
                  << input->get_total_num_testing_samples() << " ("
                  << input->get_num_samples_tested() / m->get_epoch() << " per epoch)" << std::endl;
      }
    }
  }
}

}  // namespace lbann
