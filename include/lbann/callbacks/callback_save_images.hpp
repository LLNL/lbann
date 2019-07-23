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

#ifndef LBANN_CALLBACKS_CALLBACK_SAVE_IMAGES_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_SAVE_IMAGES_HPP_INCLUDED

#include <string>
#include <vector>
#include "lbann/callbacks/callback.hpp"

namespace lbann {

/** Save layer outputs as image files.
 *  Image files are in the form
 *  "<prefix><tag>-<layer name>.<format>".
 */
class lbann_callback_save_images : public lbann_callback {
public:

  /** Constructor.
   *  @param layer_names  List of layer names to save as images.
   *  @param image_format Image file format (e.g. jpg, png, pgm).
   *  @param image_prefix Prefix for image file names.
   */
  lbann_callback_save_images(std::vector<std::string> layer_names,
                             std::string image_format = "jpg",
                             std::string image_prefix = "");
  lbann_callback_save_images(const lbann_callback_save_images&) = default;
  lbann_callback_save_images& operator=(
    const lbann_callback_save_images&) = default;
  lbann_callback_save_images* copy() const override {
    return new lbann_callback_save_images(*this);
  }
  void on_epoch_end(model *m) override;
  void on_test_end(model *m) override;
  std::string name() const override { return "save images"; }

private:

  /** List of layer names to save as images. */
  std::vector<std::string> m_layer_names;
  /** Image file format.
   *  Valid options: jpg, png, pgm.
   */
  std::string m_image_format;
  /** Prefix for saved image files. */
  std::string m_image_prefix;

};

// Builder function
std::unique_ptr<lbann_callback>
build_callback_save_images_from_pbuf(
  const google::protobuf::Message&, lbann_summary*);

} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_SAVE_IMAGES_HPP_INCLUDED
