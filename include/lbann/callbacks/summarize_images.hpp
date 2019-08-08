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
// summarize_images .hpp .cpp - Callback hooks to dump
// results of image testing to event files
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_SUMMARIZE_IMAGES_HPP_INCLUDED
#define LBANN_CALLBACKS_SUMMARIZE_IMAGES_HPP_INCLUDED


#include "lbann/callbacks/callback.hpp"

#include <google/protobuf/message.h>
#include <lbann/base.hpp>
#include <string>
#include <vector>
namespace lbann {
namespace callback {

class ImageOutputStrategy {

public:
  virtual std::vector<El::Int> get_image_indices() = 0;
  virtual ~ImageOutputStrategy() = default;

}; //class OutputStrategy


/** @class CategoricalAccuracy Subclass of ImageOutputStrategy to dump categorized images
 *  @brief Dump images to event files based on categorization criteria
 */
class CategoricalAccuracy : ImageOutputStrategy {

public:

  enum class MatchType
  {
    NOMATCH=0,
    MATCH=1,
    ALL=2
  };// enum class MatchType

  /** @brief summarize_images Constructor.
   *  @param cat_accuracy_layer_name Name of categorical accuracy layer
   *  @param match_type Criteria for dumping images (MATCH, NOMATCH, or ALL)
   */
  CategoricalAccuracy(model* m,
                      std::string const& cat_accuracy_layer_name,
                      MatchType match_type,
                      size_t num_images)
    : m_model(m),
      m_cat_accuracy_layer_name(cat_accuracy_layer_name),
      m_match_type(match_type),
      m_num_images(num_images) {}

  /** @brief Get vector containing indices of images to be dumped.
  *  @returns std::vector<int> Vector with indices of images to dump.
  */
  std::vector<El::Int> get_image_indices() final;

private:
   /** @brief Tests whether image should be dumped based on criteria
   *  @returns bool Value is true if matches criteria and false otherwise
   */
  bool meets_criteria( const DataType& match );

  /** Model */
  model* m_model;

  /** Name of categorical accuracy layer*/
  std::string const m_cat_accuracy_layer_name;

  /** lbann::Layer object */
  Layer const* m_cat_accuracy_layer = nullptr;

  /** Criterion to dump images */
  MatchType m_match_type;

  /* Number of images to be dumped per epoch */
  size_t m_num_images;

}; // class CategoricalAccuracy : ImageOutputStrategy


/** @class Autoencoder Subclass of ImageOutputStrategy to dump autoencoder images
 *  @brief Dump images to event files based on strategy
 */
class Autoencoder : ImageOutputStrategy {

public:

  /** @brief Autoencoder : ImageOutputStrategy Constructor.
   *  @param sample_indices Vector of sample indices for images
   */
  Autoencoder(El::Matrix<El::Int>* sample_indices, size_t num_images = 10)
    : m_sample_indices(sample_indices),
      m_num_images(num_images) {}

  /** @brief Get vector containing indices of images to be dumped.
   *  @returns std::vector<int> Vector with indices of images to dump.
   */
  std::vector<El::Int> get_image_indices() final;

private:

  /** Sample indices of images to track */
  std::unordered_set<El::Int> m_tracked_images;

  /** Sample indices of images */
  El::Matrix<El::Int>* m_sample_indices;

  /* Number of images to be tracked */
  size_t m_num_images;

}; // class Autoencoder : ImageOutputStrategy

/** @class summarize_images
 *  @brief Callback to dump images to event files based on strategy
 */
class summarize_images : public callback_base {

public:
  /** @brief summarize_images Constructor.
   *  @param strategy Pointer to image ImageOutputStrategy
   *  @param summarizer Pointer to lbann_summary object
   *  @param img_source_layer_name Name of image layer
   *  @param input_layer_name Name of input layer
   *  @param num_images Max number of images to dump per epoch
   *  @param interval Interval of epochs to dump images
   *  @param img_format Image file format (e.g. .jpg, .png, .pgm)
   */
  summarize_images(std::shared_ptr<lbann_summary> const& summarizer,
                   std::shared_ptr<ImageOutputStrategy> const& strategy,
                   std::string const& img_source_layer_name,
                   std::string const& input_layer_name,
                   uint64_t interval = 1,
                   uint64_t num_images = 10,
                   std::string const& img_format = ".jpg");

  /** @brief Copy constructor */
callback_base* copy() const override { return new summarize_images(*this); }

  /** @brief Return name of callback */
  std::string name() const override { return "summarize_images"; }

  /** @brief Hook to pull data from lbann run */
  void on_batch_evaluate_end(model* m) override;

private:
  /** @brief setup layers */
  void setup(model* m);

  /** @brief Add image to event file */
  void dump_images_to_summary(const Layer& layer,
                             const uint64_t& step,
                             const El::Int& epoch);

  /** @brief Construct tag for image */
  std::string get_tag(El::Int index, El::Int epoch, size_t img_number = 0);


private:

  /* lbann_summary object */
  std::shared_ptr<lbann_summary> m_summarizer;

  /* ImageOutputStrategy object */
  std::shared_ptr<ImageOutputStrategy> m_strategy;

  /* Names of layers */
  std::string const m_img_source_layer_name;
  std::string const m_input_layer_name;

  /** lbann::Layer objects */
  Layer const* m_img_source_layer = nullptr;
  Layer const* m_input_layer = nullptr;

  /* Size of mini-batch */
  size_t m_mini_batch_size;

  /* Interval for dumping images */
  uint64_t m_interval;

  /* Number of images to be dumped */
  size_t m_num_images;

  /** Image file format. Valid options: .jpg, .png, .pgm. */
  std::string m_img_format;
}; // class summarize_images

/** @brief Free function - gets layers from model based on name
 *  @param layers Vector with pointers to the Layers
 *  @param layer_name Name of layer
 */
Layer const* get_layer_by_name(const std::vector<Layer*>& layers,
                                 const std::string& layer_name);

std::unique_ptr<callback_base>
build_summarize_images_callback_from_pbuf(
  const google::protobuf::Message&,
  const std::shared_ptr<lbann_summary>& summarizer);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_SUMMARIZE_IMAGES_HPP_INCLUDED
