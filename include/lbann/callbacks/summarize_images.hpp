////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

// Forward-declare protobuf classes
namespace lbann_data {
class Callback_CallbackSummarizeImages;
}

namespace lbann {
namespace callback {

/** @class image_output_strategy
 *  @brief Interface for strategies for determining which images
 *  to output to the summarizer.
 */
class image_output_strategy
{

public:
  virtual std::vector<std::pair<size_t, El::Int>>
  get_image_indices(model const&) const = 0;
  virtual std::string get_tag(std::string const& layer_name,
                              El::Int index,
                              El::Int epoch) const = 0;
  virtual void write_strategy_proto(
    lbann_data::Callback_CallbackSummarizeImages& msg) const = 0;
  virtual ~image_output_strategy() = default;

}; // class image_output_strategy

/** @class CategoricalAccuracy
 *  @brief Subclass of image_output_strategy to dump categorized
 *  images to event files based on categorization criteria
 */
class categorical_accuracy_strategy : public image_output_strategy
{
public:
  enum class MatchType
  {
    NOMATCH = 0,
    MATCH = 1,
    ALL = 2
  }; // enum class MatchType

  /** @brief summarize_images Constructor.
   *  @param cat_accuracy_layer_name Name of categorical accuracy layer
   *  @param match_type Criteria for dumping images (MATCH, NOMATCH, or ALL)
   *  @param num_images Number of images to summarize per epoch
   */
  categorical_accuracy_strategy(std::string const& cat_accuracy_layer_name,
                                MatchType match_type = MatchType::NOMATCH,
                                size_t num_images = 10)
    : m_cat_accuracy_layer_name(cat_accuracy_layer_name),
      m_match_type(match_type),
      m_num_images(num_images)
  {}

  /** @brief Get vector containing indices of images to be dumped.
   *  @returns std::vector<int> Vector with indices of images to dump.
   */
  std::vector<std::pair<size_t, El::Int>>
  get_image_indices(model const& m) const final;

  /** @brief Construct tag for image */
  std::string get_tag(std::string const& layer_name,
                      El::Int index,
                      El::Int epoch) const final;

private:
  /** @brief Write strategy specific data to prototext */
  void write_strategy_proto(
    lbann_data::Callback_CallbackSummarizeImages& msg) const final;

  /** @brief Tests whether image should be dumped based on criteria
   *  @returns bool Value is true if matches criteria and false otherwise
   */
  bool meets_criteria(const DataType& match) const noexcept;

  /** @brief Name of categorical accuracy layer*/
  std::string const m_cat_accuracy_layer_name;

  /** @brief Criterion to dump images */
  MatchType m_match_type;

  /** @brief Number of images to be dumped per epoch */
  size_t m_num_images;

}; // class categorical_accuracy_strategy : image_output_strategy

std::unique_ptr<image_output_strategy>
build_categorical_accuracy_strategy_from_pbuf(google::protobuf::Message const&);

/** @class autoencoder_strategy
 *  @brief Subclass of image_output_strategy to dump autoencoder images.
 *  @details Dump images to event files based on strategy
 */
class autoencoder_strategy : public image_output_strategy
{

public:
  /** @brief Constructor
   *  @param input_layer_name The input layer.
   *  @param num_images The number of images to dump.
   */
  autoencoder_strategy(std::string const& input_layer_name,
                       size_t num_images = 10)
    : m_input_layer_name{input_layer_name}, m_num_images{num_images}
  {}

  /** @brief Get vector containing indices of images to be dumped.
   *  @returns std::vector<int> Vector with indices of images to dump.
   */
  std::vector<std::pair<size_t, El::Int>>
  get_image_indices(model const& m) const final;

  /** @brief Construct tag for image */
  std::string get_tag(std::string const& layer_name,
                      El::Int index,
                      El::Int epoch) const final;

  /** @brief Write strategy specific data to prototext */
  void write_strategy_proto(
    lbann_data::Callback_CallbackSummarizeImages& msg) const final;

private:
  /** @brief Name of input layer */
  std::string m_input_layer_name;

  /** @brief Number of images to be tracked */
  size_t m_num_images;

  /** @brief Sample indices of images to track */
  mutable std::unordered_set<El::Int> m_tracked_images;

  /** @brief A map from models to shuffled indices */
  mutable std::unordered_map<model const*, std::vector<size_t>>
    m_shuffled_indices;

}; // class Autoencoder : image_output_strategy

std::unique_ptr<image_output_strategy>
build_track_sample_ids_strategy_from_pbuf(google::protobuf::Message const&);

/** @class summarize_images
 *  @brief Callback to dump images to event files based on strategy
 */
class summarize_images : public callback_base
{

public:
  /** @brief summarize_images Constructor.
   *  @param summarizer Pointer to lbann_summary object
   *  @param strategy Pointer to image image_output_strategy
   *  @param img_source_layer_name Name of image layer
   *  @param interval Interval of epochs to dump images
   *  @param img_format Image file format (e.g. .jpg, .png, .pgm)
   */
  summarize_images(std::shared_ptr<lbann_summary> const& summarizer,
                   std::unique_ptr<image_output_strategy> strategy,
                   std::string const& img_source_layer_name,
                   uint64_t interval = 1,
                   std::string const& img_format = ".jpg");

  /** @brief Copy constructor */
  callback_base* copy() const override;

  /** @brief Return name of callback */
  std::string name() const override { return "summarize_images"; }

  /** @brief Hook to pull data from lbann run */
  void on_batch_evaluate_end(model* m) override;

  /** Add callback specific data to prototext */
  void write_specific_proto(lbann_data::Callback& proto) const final;

private:
  /** @brief Add image to event file */
  void dump_images_to_summary(model const& m) const;

private:
  /* @brief lbann_summary object */
  std::shared_ptr<lbann_summary> m_summarizer;

  /* @brief image_output_strategy object */
  std::unique_ptr<image_output_strategy> m_strategy;

  /* @brief Names of layers */
  std::string m_img_source_layer_name;

  /* @brief Interval for dumping images */
  uint64_t m_epoch_interval;

  /** @brief Image file format. Valid options: .jpg, .png, .pgm. */
  std::string m_img_format;

}; // class summarize_images

/** @brief Get a layer from model based on name
 *  @param m The model
 *  @param layer_name Name of layer
 */
Layer const& get_layer_by_name(model const& m, std::string const& layer_name);

std::unique_ptr<callback_base> build_summarize_images_callback_from_pbuf(
  const google::protobuf::Message&,
  const std::shared_ptr<lbann_summary>& summarizer);

} // namespace callback
} // namespace lbann

#endif // LBANN_CALLBACKS_SUMMARIZE_IMAGES_HPP_INCLUDED
