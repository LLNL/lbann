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
  //save_image(layers[phase]->m_activations, layers[phase+2]->m_activations);
}

void lbann_callback_save_images::save_image(ElMat* input, ElMat* output){
  DistMat in_col,out_col;
  View(in_col,*input, ALL, IR(0));//@todo: remove hardcoded 0, save any image (index) you want, 0 as default
  View(out_col,*output, ALL, IR(0));
  CircMat in_pixel = in_col;
  CircMat out_pixel = out_col;
  //m_reader->save_image(in_pixel.Matrix(), m_image_dir+"gt."+m_extension);
  m_reader->save_image(in_pixel.Matrix(), m_image_dir+"gt."+m_extension,false);
  m_reader->save_image(out_pixel.Matrix(), m_image_dir+"rc."+m_extension,false);
}

void lbann_callback_save_images::save_images(ElMat* input, ElMat* output,int phase) {

  //check that num_images is less than minibatch size
  Int num_images = std::min(m_num_images,std::min(input->Width(),output->Width()));

  Int num_neurons = std::min(input->Height(),output->Height());

  Int y_dim = floor(std::sqrt(num_neurons));
  Int x_dim = floor(std::sqrt(num_neurons));

  uchar* pixels_gt = new uchar[x_dim * num_images * y_dim];
  uchar* pixels_rc = new uchar[x_dim * num_images * y_dim];

  //@todo: add depth
  for (Int n = 0; n < num_images; n++) {
    DistMat in_col,out_col;
    View(in_col,*input, ALL, IR(n));
    View(out_col,*output, ALL, IR(n));
    CircMat in_pixel_circ = in_col;
    CircMat out_pixel_circ = out_col;
    Mat in_pixels = in_pixel_circ.Matrix();
    Mat out_pixels = out_pixel_circ.Matrix();

    for (Int y = 0; y < y_dim; y++) {
      for (Int x = 0; x < x_dim; x++) {
        //pixels_gt[y * y_dim * num_images + x + x_dim * n] = in_pixels[y * y_dim + x]; //input/ground truth
        pixels_gt[y * y_dim * num_images + x + x_dim * n] = in_pixels.Get(y * y_dim + x,0) * 255; //input/ground truth
        pixels_rc[y * y_dim * num_images + x + x_dim * n] = out_pixels.Get(y * y_dim + x,0) * 255; //output/reconstructed
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
