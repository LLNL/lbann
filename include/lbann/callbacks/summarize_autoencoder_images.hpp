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
// summarize_autoencoder_images .hpp .cpp - Callback hooks to dump
// results of image testing to event files
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_SUMMARIZE_AUTOENCODER_IMAGES_HPP_INCLUDED
#define LBANN_CALLBACKS_SUMMARIZE_AUTOENCODER_IMAGES_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

#include <google/protobuf/message.h>

#include <string>
#include <vector>

namespace lbann {
namespace callback {

/** @class lbann_callback_summarize_autoencoder_images
 *  @brief Dump images with testing results to event files
 */
class summarize_autoencoder_images : public callback_base {
 public:

public:
  /** @brief Constructor.
   *  @param summarizer Pointer to lbann_summary object
   *  @param reconstruction_layer_name Name of reconstruction layer
   *  @param img_layer_name Name of image layer
   *  @param input_layer_name Name of input layer
   *  @param interval Interval of epochs to dump images
   *  @param image_format Image file format (e.g. .jpg, .png, .pgm)
   *  @param num_images Number of images to track
   */
  summarize_autoencoder_images(std::shared_ptr<lbann_summary> const& summarizer,
                               std::string const& reconstruction_layer_name,
                               std::string const& img_layer_name,
                               std::string const& input_layer_name,
                               uint64_t interval,
                               std::string const& img_format = ".jpg",
                               size_t const& num_images = 15);

  /** @brief Copy constructor */
  callback_base* copy() const override { return new summarize_autoencoder_images(*this); }

  /** @brief Return name of callback */
  std::string name() const override { return "summarize_autoencoder_images"; }

  /** @brief Hook to pull data from lbann run */
  void on_batch_evaluate_end(model* m) override;

private:
  /** @brief setup layers */
  void setup(model* m) override;

  /** @brief Gets layers from model based on name
   *  @param layers Vector with pointers to the Layers
   *  @param layer_name Name of layer
   */
  Layer const* get_layer_by_name(const std::vector<Layer*>& layers, const std::string& layer_name);

  /** @brief Get vector containing indices of images to be dumped.
   *  @returns std::vector<int> Vector with indices of images to dump.
   */
  std::vector<El::Int> get_image_indices();

  /** @brief Add images to event file */
  void dump_images_to_summary(const Layer& layer,
                              const uint64_t& step,
                              const El::Int& epoch = -1);
  /** @brief Construct tag for image */
  std::string get_tag(El::Int index, El::Int epoch);


private:

  /* lbann_summary object */
  std::shared_ptr<lbann_summary> m_summarizer;

  /* Names layers */
  std::string m_reconstruction_layer_name;
  std::string m_img_layer_name;
  std::string m_label_layer_name;
  std::string m_input_layer_name;

  /** lbann::Layer objects */
  Layer const* m_reconstruction_layer = nullptr;
  Layer const* m_img_layer = nullptr;
  Layer const* m_label_layer = nullptr;
  Layer const* m_input_layer = nullptr;

  /* Interval for dumping images */
  uint64_t m_interval;

  /** Image file format.
   *  Valid options: jpg, png, pgm.
   */
  std::string m_img_format;

  /** Sample indices of images to track */
  std::unordered_set<El::Int> m_tracked_images;

  /** Number of images to track */
  size_t m_num_images;

  /** Number of images in mini-batch **/
  size_t m_mini_batch_size;
};

std::unique_ptr<callback_base>
build_summarize_autoencoder_images_callback_from_pbuf(
  const google::protobuf::Message&,
  const std::shared_ptr<lbann_summary>& summarizer);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_SUMMARIZE_AUTOENCODER_IMAGES_HPP_INCLUDED
