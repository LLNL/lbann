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

#ifndef LBANN_CALLBACKS_CALLBACK_SAVE_IMAGES_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_SAVE_IMAGES_HPP_INCLUDED

#include <utility>

#include "lbann/callbacks/callback.hpp"
#include "lbann/data_readers/data_reader.hpp"

namespace lbann {

/**
 * Save images to file
 */
class lbann_callback_save_images : public lbann_callback {
 public:
  /**
   * @param data reader type e.g., imagenet, mnist, cifar10....
   * @param image_dir directory to save image
   * @param layer_names list of layers from which to save images 
   * @param image extension e.g., jpg, png, pgm......
   */
  lbann_callback_save_images(generic_data_reader *reader, std::string image_dir,
                             std::vector<std::string> layer_names,
                             std::string extension="jpg") :
    lbann_callback(), m_image_dir(std::move(image_dir)), m_extension(std::move(extension)),
    m_reader(reader), m_layer_names(layer_names) {}
  lbann_callback_save_images(const lbann_callback_save_images&) = default;
  lbann_callback_save_images& operator=(
    const lbann_callback_save_images&) = default;
  lbann_callback_save_images* copy() const override {
    return new lbann_callback_save_images(*this);
  }
  void on_epoch_end(model *m) override;
  void on_phase_end(model *m) override;
  void on_test_end(model *m) override;
  std::string name() const override { return "save images"; }
 private:
  std::string m_image_dir; //directory to save images
  std::string m_extension; //image extension; pgm, jpg, png etc
  generic_data_reader *m_reader;
  /** List of layers at which to save images*/
  std::vector<std::string> m_layer_names;
  void save_image(model& m, std::string tag);
};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_SAVE_IMAGES_HPP_INCLUDED
