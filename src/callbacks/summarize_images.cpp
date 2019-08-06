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

#include <lbann_config.hpp>
#include <lbann/utils/image.hpp>
#include <lbann/utils/summary.hpp>
#include "lbann/callbacks/summarize_images.hpp"
#include <lbann/layers/io/input/generic_input_layer.hpp>

#include <callbacks.pb.h>

#include <iostream>

namespace lbann {
namespace callback {

summarize_images::summarize_images(
  lbann_summary *summarizer,
  std::string const& cat_accuracy_layer_name,
  std::string const& img_layer_name,
  std::string const& input_layer_name,
  MatchType match_type,
  size_t num_images,
  uint64_t interval,
  std::string const& img_format)
  : callback_base(1),
    m_summarizer(summarizer),
    m_cat_accuracy_layer_name(cat_accuracy_layer_name),
    m_img_layer_name(img_layer_name),
    m_input_layer_name(input_layer_name),
    m_match_type(match_type),
    m_num_images(num_images),
    m_interval(interval),
    m_img_format(img_format)
{
#ifndef LBANN_HAS_OPENCV
  LBANN_ERROR("OpenCV not detected");
#endif // LBANN_HAS_OPENCV
}

namespace
{
template <typename... Ts>
void ThrowLBANNError(Ts... args)
{
  std::ostringstream oss;
  int dummy[] = { (oss << args, 0)... };
  (void) dummy;
  LBANN_ERROR(oss.str());
}
}

void summarize_images::on_batch_evaluate_end(model* m) {
  if (m->get_step() % m_interval != 0)
    return;

  if (m_cat_accuracy_layer == nullptr) {
    setup(m);
  }

  dump_image_to_summary(m->get_step(), m->get_epoch());
}

void summarize_images::setup(model* m){
  auto layers = m->get_layers();
  /* find layers in model based on string */
  m_cat_accuracy_layer = get_layer_by_name(layers, m_cat_accuracy_layer_name);
  m_img_layer = get_layer_by_name(layers, m_img_layer_name);
  m_input_layer = get_layer_by_name(layers, m_input_layer_name);

  // Check widths of img_layer.activations and cat_accuracy_layer are equal
  const AbsDistMat& cat_accuracy_activations = m_cat_accuracy_layer->get_activations();
  const AbsDistMat& img_layer_activations = m_img_layer->get_activations();
  if( cat_accuracy_activations.Width() != img_layer_activations.Width() )
    ThrowLBANNError("Invalid data. Categorical accuracy activations and image activations widths do not match.");

  if (auto gil = dynamic_cast<generic_input_layer const*>(m_input_layer)){
    m_num_images = std::min(static_cast<long>(m_num_images),
                            gil->get_dataset(execution_mode::validation).get_total_samples());
//FIXME: Use if m_num_images exceeds mini-batch size
  //m_mini_batch_size = gil->get_current_mini_batch_size();
  }
}

Layer const* summarize_images::get_layer_by_name(
  const std::vector<Layer*>& layers,
  const std::string& layer_name)
{
  for(auto const* l : layers)
      if( l->get_name() == layer_name)
        return l;

  ThrowLBANNError("Layer ", layer_name, " not found.");
  return nullptr;
}

std::vector<El::Int> summarize_images::get_image_indices() {
  const AbsDistMat& categorized_correctly_dist = m_cat_accuracy_layer->get_activations();
  CircMat<El::Device::CPU> categorized_correctly(
    categorized_correctly_dist.Grid(), categorized_correctly_dist.Root());
  categorized_correctly = categorized_correctly_dist;

  if (categorized_correctly.Height() != El::Int(1))
    LBANN_ERROR("Tom was wrong about this matrix. Oops.");

  // Create return value
  std::vector<El::Int> img_indices;

  // Fill return value if root process
  if (categorized_correctly.CrossRank() == categorized_correctly.Root()) {
    // Loop over all samples -- samples are the *width* of the matrix
    auto const num_samples = categorized_correctly.LocalWidth();
    for (auto sample = decltype(num_samples){0}; sample < num_samples; ++sample) {
      auto const& correctness_value = categorized_correctly.LockedMatrix()(0, sample);

      if ((correctness_value != DataType(0))
          && (correctness_value != DataType(1)))
        ThrowLBANNError("Invalid data from ", m_cat_accuracy_layer->get_name(),
                        ". Received ", correctness_value, ", expected 0 or 1.");

      if (meets_criteria(correctness_value))
        img_indices.push_back(sample);

      if(img_indices.size() > m_num_images)
        break;
    }
  }

  return img_indices;
}

bool summarize_images::meets_criteria( const DataType& match ) {
  if( (match && (m_match_type == MatchType::MATCH)) ||
      (!match && (m_match_type == MatchType::NOMATCH)) ||
      (m_match_type == MatchType::ALL))
    return true;

  return false;

}

void summarize_images::dump_image_to_summary( const uint64_t& step,
                                              const El::Int& epoch) {

  static size_t img_number = 0;

  const AbsDistMat& img_layer_activations = m_img_layer->get_activations();

  CircMat<El::Device::CPU> all_images(
      img_layer_activations.Grid(), img_layer_activations.Root());
  all_images = img_layer_activations;

  if (all_images.CrossRank() == all_images.Root()) {
    auto const& local_images = all_images.LockedMatrix();
    auto dims = m_img_layer->get_output_dims();

    std::vector<El::Int> img_indices = get_image_indices();

    for (const El::Int& col_index : img_indices) {
      if (col_index >= local_images.Width())
        LBANN_ERROR("Bad col index.");

      auto sample_indices = const_cast<Layer&>(*m_input_layer).get_sample_indices_per_mb();
      auto sample_index = sample_indices->Get(0, col_index);
      auto const local_image = local_images(El::ALL, El::IR(col_index));
      std::string image_tag("epoch-" + std::to_string(epoch) +
                            "/ sample_index-" + std::to_string(sample_index) +
                            "/ image-" + std::to_string(img_number++));
      this->m_summarizer->report_image(image_tag, m_img_format, local_image, dims, step);
    }
  }
}

std::unique_ptr<callback_base>
build_summarize_images_callback_from_pbuf(
  const google::protobuf::Message& proto_msg, lbann_summary* summarizer){

  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackSummarizeImages&>(proto_msg);

  /* Add criterion->MatchType function */
  auto ConvertToLbannType = [](lbann_data::Callback_CallbackSummarizeImages_MatchType a)
    {
      return static_cast<summarize_images::MatchType>(a);
    };

  return make_unique<summarize_images>(
    summarizer,
    params.cat_accuracy_layer(),
    params.image_layer(),
    params.input_layer(),
    ConvertToLbannType(params.criterion()),
    params.interval());
}

} // namespace callback
} // namespace lbann
