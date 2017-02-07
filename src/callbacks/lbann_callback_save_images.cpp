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
#include "lbann/callbacks/lbann_callback_save_images.hpp"
#include "lbann/data_readers/lbann_image_utils.hpp"

namespace lbann {

void lbann_callback_save_images::on_phase_end(model* m) {
  auto layers = m->get_layers();
  auto phase = m->get_current_phase();
  save_images(layers[phase]->m_activations, layers[phase+2]->m_activations,phase*10);
}

void lbann_callback_save_images::save_images(ElMat* input, ElMat* output,int phase) {

  //check that num_images is less than minibatch size
  Int num_images = std::min(m_num_images,std::min(input->Width(),output->Width()));

  Int num_neurons = std::min(input->Height(),output->Height());

  uint y_dim = floor(std::sqrt(num_neurons));
  uint x_dim = floor(std::sqrt(num_neurons));

  uchar* pixels_gt = new uchar[x_dim * num_images * y_dim];
  uchar* pixels_rc = new uchar[x_dim * num_images * y_dim];

  //@todo: add depth
  for (size_t n = 0; n < num_images; n++) {
    vector<uchar> in_pixels(num_neurons);
    vector<uchar> out_pixels(num_neurons);
    for (uint m = 0; m < num_neurons; m++){
      in_pixels[m] = input->Get(m,n) * 255;
      out_pixels[m] = output->Get(m,n) * 255;
    }

    for (size_t y = 0; y < y_dim; y++) {
      for (size_t x = 0; x < x_dim; x++) {
        pixels_gt[y * y_dim * num_images + x + x_dim * n] = in_pixels[y * y_dim + x]; //input/ground truth
        pixels_rc[y * y_dim * num_images + x + x_dim * n] = out_pixels[y * y_dim + x]; //output/reconstructed
      }
    }
  }

	char imagepath_gt[512];
  char imagepath_rc[512];
  sprintf(imagepath_gt, "%slbann_image_gt_%02d.pgm", m_image_dir.c_str(),phase);
  sprintf(imagepath_rc, "%slbann_image_rc_%02d.pgm", m_image_dir.c_str(), phase);
  lbann::image_utils::savePGM(imagepath_gt, y_dim * num_images, x_dim, 1, false, pixels_gt);
  lbann::image_utils::savePGM(imagepath_rc, y_dim * num_images, x_dim, 1, false, pixels_rc);

  delete [] pixels_gt;
  delete [] pixels_rc;
}

}  // namespace lbann
