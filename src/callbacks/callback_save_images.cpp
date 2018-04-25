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
// lbann_callback_save_images .hpp .cpp - Callbacks to save images
////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include "lbann/callbacks/callback_save_images.hpp"
#include "lbann/data_readers/image_utils.hpp"

namespace lbann {


void lbann_callback_save_images::on_epoch_end(model *m) {
  auto tag = "epoch" + std::to_string(m->get_cur_epoch());
  save_image(*m,tag);
}


void lbann_callback_save_images::on_phase_end(model *m) {
  const auto phase = m->get_current_phase();
  auto tag = "phase" + std::to_string(phase);
  save_image(*m,tag);
}

void lbann_callback_save_images::on_test_end(model *m) {
  save_image(*m,"test");
}

void lbann_callback_save_images::save_image(model& m,
                                            std::string tag) {

  // Save image
  if(m_layer_names.empty()) {
    if(m.get_comm()->am_world_master()) 
      std::cout << "Layer list empty, images not saved " << std::endl;
    return;
  }
 //@todo: check that number of neurons (linearized) equal mat heigth? 
 if(m.get_comm()->am_world_master()) 
      std::cout << "Saving images to " << m_image_dir << std::endl;
  
  const auto layers = m.get_layers();
  for(auto& l: layers) {
    auto layer_name = l->get_name();
    if(std::find(std::begin(m_layer_names), std::end(m_layer_names),
                  layer_name) != std::end(m_layer_names)) {

      AbsDistMat* input_col = l->get_activations().Construct(
                                          l->get_activations().Grid(),
                                          l->get_activations().Root());
      El::View(*input_col, l->get_activations(), El::ALL, El::IR(0));
      CircMat input_circ = *input_col;
      delete input_col;

      if(m.get_comm()->am_world_master()) 
        m_reader->save_image(input_circ.Matrix(), 
                             m_image_dir+tag+"-"+layer_name+"."+m_extension);
    }
  }  
}
  

}  // namespace lbann
