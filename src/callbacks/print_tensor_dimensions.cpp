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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/print_tensor_dimensions.hpp"

namespace lbann {
namespace callback {

void print_tensor_dimensions::on_train_begin(model *m) {
  const auto comm = m->get_comm();
  if(!comm->am_trainer_master())
    return;

  // Return "NxCxDxHxW (tensor_name)"
  const auto tensor_dims_str =
      [](const std::vector<int> dims, const Layer& l) {
        std::stringstream ss;
        for(unsigned int i = 0; i < dims.size(); i++)
          ss << (i ? "x" : "") << dims[i];
        ss << " ("<< l.get_name() << ")";
        return ss.str();
      };

  std::stringstream ss;
  ss << "print_tensor_dimensions callback:";
  const auto layers = m->get_layers();
  for(const Layer* layer : layers) {
    ss << "   " << layer->get_name() << ": ";
    if(layer->get_num_parents()) {
      for(int i = 0; i < layer->get_num_parents(); i++) {
        const std::vector<int> input_dims = layer->get_input_dims(i);
        ss << (i ? ", " : "") << tensor_dims_str(input_dims, layer->get_parent_layer(i));
      }
    } else {
      ss << "no input";
    }
    ss << " -> ";
    if(layer->get_num_children()) {
      for(int i = 0; i < layer->get_num_children(); i++) {
        const std::vector<int> output_dims = layer->get_output_dims(i);
        ss << (i ? ", " : "") << tensor_dims_str(output_dims, layer->get_child_layer(i));
      }
    } else {
      ss << "no output";
    }
    ss << std::endl;
  }
  std::cout << ss.str();

}

} // namespace callback
} // namespace lbann
