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
// lbann_callback_save_images .hpp .cpp - Callbacks to save images, currently used in autoencoder
////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include "lbann/callbacks/callback_save_images.hpp"
#include "lbann/data_readers/image_utils.hpp"

namespace lbann {


void lbann_callback_save_images::on_epoch_end(model *m) {
  auto epoch = m->get_cur_epoch();
  auto layers = m->get_layers();
  //@todo: generalize to two matching layers
  //@todo: use view so we can save arbitrary number of valid images
  //using layer just before reconstruction layer because prev_layer_act is protected
  save_image(*m,
             layers[0]->get_activations(),
             layers[layers.size()-2]->get_activations(),
             epoch);
}


void lbann_callback_save_images::on_phase_end(model *m) {
  auto layers = m->get_layers();
  auto phase = m->get_current_phase();
  auto epoch = m->get_cur_epoch();
  auto index = phase*epoch + epoch;
  save_image(*m,
             layers[phase]->get_activations(),
             layers[phase+2]->get_activations(),
             index);
}

void lbann_callback_save_images::save_image(model& m,
                                            AbsDistMat& input,
                                            AbsDistMat& output,
                                            int index) {

  // Save input image
  AbsDistMat* input_col = input.Construct(input.Grid(),
                                          input.Root());
  El::View(*input_col, input, El::ALL, El::IR(0));
  CircMat input_circ = *input_col;
  delete input_col;
  if(m.get_comm()->am_world_master()) {
    m_reader->save_image(input_circ.Matrix(), m_image_dir+"input_"+ std::to_string(index)+"."+m_extension);
  }
  
  // Save output image if it is a reconstruction
  if(output.Height() == input.Height()) {
    AbsDistMat* output_col = output.Construct(output.Grid(),
                                              output.Root());
    El::View(*output_col, output, El::ALL, El::IR(0));
    CircMat output_circ = *output_col;
    delete output_col;
    if(m.get_comm()->am_world_master()) {
      m_reader->save_image(output_circ.Matrix(), m_image_dir+"output_"+ std::to_string(index)+"."+m_extension);
    }
  }

  // Output message
  if(m.get_comm()->am_world_master()) {
    std::cout << "Saved images to " << m_image_dir << std::endl;
  }

}


}  // namespace lbann
