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

#include <vector>
#include "lbann/callbacks/callback_dump_activations.hpp"
#include "lbann/layers/io/target/target_layer.hpp"

namespace lbann {

void lbann_callback_dump_activations::on_forward_prop_end(model *m, Layer *l) {

  // Skip if we are interested in saving inferences at (a) given layer(s)
  if (!m_layer_names.empty()) { return; }

  // Skip if layer has no activations
  if (l->get_num_children() == 0) { return; }

  // Create matrix for matrix views
  const auto& acts = l->get_activations();
  std::unique_ptr<AbsDistMat> acts_v(acts.Construct(acts.Grid(),
                                                    acts.Root()));

  // Write each activation matrix to file
  const auto& children = l->get_child_layers();
  for (size_t i = 0; i < children.size(); ++i) {

    // File name
    std::stringstream file;
    file << m_basename
         << "model" << m->get_comm()->get_model_rank() << "-"
         << "epoch" << m->get_cur_epoch() << "-"
         << "step" << m->get_cur_step() << "-"
         << l->get_name() << "-"
         << "Activations";
    if (children.size() > 1) { file << i; }

    // Write activations to file
    // Note: This forces a memory transfer from GPU to CPU
    l->get_fp_output(*acts_v, children[i]);
    El::Write(*acts_v, file.str(), El::ASCII);

  }

}

void lbann_callback_dump_activations::on_epoch_end(model *m) {
  auto tag = "epoch" + std::to_string(m->get_cur_epoch());
  dump_activations(*m,tag);
}

void lbann_callback_dump_activations::on_test_end(model *m) {
  dump_activations(*m,"test");
}

void lbann_callback_dump_activations::dump_activations(model& m,
                                            std::string tag) {
  //Skip if layer list is empty, user may interested in the saving all activations
  //Use method above 
  if (m_layer_names.empty()) { return; }

  const auto layers = m.get_layers();
  for(auto& l: layers) {
    if(std::find(std::begin(m_layer_names), std::end(m_layer_names),
                  l->get_name()) != std::end(m_layer_names)) {
      //@todo: generalize to support different format
      const std::string file
        = (m_basename
           + "model" + std::to_string(m.get_comm()->get_model_rank())
           + "-" + l->get_name()
           + "-" + tag
           + "-Activations");
       El::Write(l->get_activations(), file, El::ASCII);
      }
    }
}


}  // namespace lbann
