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
// lbann_callback_summarize_images .hpp .cpp - Callback hooks to dump
// results of image testing to event files
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_SUMMARIZE_IMAGES_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_SUMMARIZE_IMAGES_HPP_INCLUDED

#include <string>
#include <vector>
#include "lbann/callbacks/callback.hpp"

#include <google/protobuf/message.h>

namespace lbann {

/** @class lbann_callback_summarize_images
 *  @brief Dump images with testing results to event files
 */
class lbann_callback_summarize_images : public lbann_callback{
 public:

  enum class MatchType
  {
    NOMATCH=0,
    MATCH=1,
    ALL=2
  };// enum class MatchType

public:
  /** @brief Constructor.
   *  @param cat_accuracy_layer Categorical accuracy layer object
   *  @param img_layer Image layer object
   *  @param summarizer lbann_summary object
   *  @param image_format Image file format (e.g. .jpg, .png, .pgm)
   */
  lbann_callback_summarize_images(lbann_summary *summarizer,
                                    std::string const& cat_accuracy_layer_name,
                                    std::string const& img_layer_name,
                                    MatchType match_type,
                                    uint64_t interval,
                                    std::string img_format = ".jpg");
  /** @brief Destructor */
  ~lbann_callback_summarize_images() {}

  lbann_callback* copy() const override { return new lbann_callback_summarize_images(*this); }
  std::string name() const override { return "callback_summarize_images"; }

  /** @brief Hook to pull data from lbann run */
  void on_batch_evaluate_end(model* m) override;

private:
  /** @brief Gets layers from model based on name */
  Layer const* get_layer_by_name(const std::vector<Layer*>& layers, const std::string& layer_name);

  /** @brief Get vector containing indices of images to be dumped.
   *  @returns std::vector<int> Vector with indices of images to dump.
   */
  std::vector<El::Int> get_image_indices();

  /** @brief Tests whether image should be dumped based on criteria
   *  @returns bool Value is true if matches criteria and false otherwise
   */
  bool meets_criteria( const DataType& match );

  /** @brief Add image to event file */
  void dump_image_to_summary(const std::vector<El::Int>& img_indices,
                             const uint64_t& step,
                             const El::Int& epoch);


private:
  /* Name of categorical accuracy layer */
  std::string m_cat_accuracy_layer_name;
  std::string m_img_layer_name;
  std::string m_label_layer_name;
//FIXME: What is this layer??
  std::string m_classifier_layer_name;

  /* Criterion for selecting images to dump */
  MatchType m_match_type;
  /** lbann::Layer objects */
  Layer const* m_cat_accuracy_layer = nullptr;
  Layer const* m_img_layer = nullptr;
  Layer const* m_label_layer = nullptr;
  Layer const* m_classifier_layer = nullptr;
  Layer const* m_input_layer = nullptr;

  /* Interval for dumping images */
  uint64_t m_interval;

  /** Image file format.
   *  Valid options: jpg, png, pgm.
   */
  std::string m_img_format;
};

std::unique_ptr<lbann_callback>
build_callback_summarize_images_from_pbuf(
  const google::protobuf::Message&, lbann_summary*);

} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_SUMMARIZE_IMAGES_HPP_INCLUDED
