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

std::vector<El::Int> CategoricalAccuracy::get_image_indices() {

  auto layers = m_model->get_layers();
  m_cat_accuracy_layer = get_layer_by_name(layers, m_cat_accuracy_layer_name);

  const AbsDistMat& categorized_correctly_dist = m_cat_accuracy_layer->get_activations();
  CircMat<El::Device::CPU> categorized_correctly(
    categorized_correctly_dist.Grid(), categorized_correctly_dist.Root());
  categorized_correctly = categorized_correctly_dist;

//FIXME: Should width of img_layer abd accuracy_layer activations be tested here?

  if (categorized_correctly.Height() != El::Int(1))
    LBANN_ERROR("Tom was wrong about this matrix. Oops.");

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

bool CategoricalAccuracy::meets_criteria( const DataType& match ) {
  if( (match && (m_match_type == MatchType::MATCH)) ||
      (!match && (m_match_type == MatchType::NOMATCH)) ||
      (m_match_type == MatchType::ALL))
    return true;

  return false;

}

std::vector<El::Int> Autoencoder::get_image_indices() {

  for(El::Int ii = 0; ii < m_sample_indices->Height(); ii++){
    if (ii >= m_sample_indices->Height())
      LBANN_ERROR(
        BuildErrorMessage(
          "col_index: ", ii, " is greater than Matrix height: ",
          m_sample_indices->Height()));

    std::vector<El::Int> img_indices;
    if (m_tracked_images.find(m_sample_indices->Get(ii,0)) != m_tracked_images.end()){
      std::cout << "I found a tracked index! Idx = " << m_sample_indices->Get(ii,0)
                << "\n";
      img_indices.push_back(ii);
    }
    else if(m_tracked_images.size() < m_num_images){
      m_tracked_images.insert(m_sample_indices->Get(ii,0));
      std::cout << "Adding to tracked indices Idx = " << m_sample_indices->Get(ii,0)
                << "\n";
      img_indices.push_back(ii);
    }

    flush(std::cout);
  }
  return img_indices;

}

summarize_images::summarize_images(
  std::shared_ptr<lbann_summary> const& summarizer,
  std::string const& img_layer_name,
  std::string const& input_layer_name,
  uint64_t interval,
  size_t num_images,
  std::string const& img_format)
: callback_base(1),
    m_summarizer(summarizer),
    m_img_layer_name(img_layer_name),
    m_input_layer_name(input_layer_name),
    m_interval(interval),
    m_num_images(num_images),
    m_img_format(img_format)

{
#ifndef LBANN_HAS_OPENCV
  LBANN_ERROR("OpenCV not detected");
#endif // LBANN_HAS_OPENCV
}

void summarize__images::on_batch_evaluate_end(model* m) {


  if (m->get_epoch() % m_interval != 0)
    return;

  if (m_img_source_layer == nullptr) {
    setup(m);
  }

  dump_images_to_summary(*m_img_source_layer, m->get_step(), m->get_epoch());
// FIXME: Dump original image for Autoencoder Strategy
//  if(m->get_epoch() > 1)
//    dump_images_to_summary(*m_img_layer, m->get_step());
}

void summarize_autoencoder_images::setup(model* m)
{
  auto layers = m->get_layers();
  /* find layers in model based on string */
  m_img_source_layer = get_layer_by_name(layers, m_img_source_layer_name);
  if (m_img_source_layer == nullptr)
    LBANN_ERROR(BuildErrorMessage("get_layer_by_name() failed on layer ",
                                  m_img_source_layer_name));

  m_input_layer = get_layer_by_name(layers, m_input_layer_name);
  if (m_input_layer == nullptr)
    LBANN_ERROR(BuildErrorMessage("get_layer_by_name() failed on layer ", m_input_layer_name));

//FIXME: Does this make sense? Error is supposed to catch reconstruction/img layer.
//       Should this be moved somewhere else for autoencoder strategy?
  // Check widths of img_layer.activations and reconstruction_layer are equal
  const AbsDistMat& img_source_activations = m_img_source_layer->get_activations();
  const AbsDistMat& input_layer_activations = m_input_layer->get_activations();
  if( img_source_activations.Width() != img_layer_activations.Width() )
    LBANN_ERROR(
      BuildErrorMessage(
        "Invalid data. Reconstruction layer activations and image activations widths "
        "do not match."));

  if (auto gil = dynamic_cast<generic_input_layer const*>(m_input_layer)){
    m_num_images = std::min(static_cast<long>(m_num_images),
                            gil->get_dataset(execution_mode::validation).get_total_samples());
//FIXME: Need to use this to conrol for more images than minibatch size in cat_accuracy
    m_mini_batch_size = gil->get_current_mini_batch_size();
  }
}

void summarize_autoencoder_images::dump_images_to_summary(
  const Layer& layer, const uint64_t& step, const El::Int& epoch) {

  auto sample_indices = const_cast<Layer&>(*m_input_layer).get_sample_indices_per_mb();
  if (sample_indices == nullptr)
    LBANN_ERROR(BuildErrorMessage("NULL SAMPLE INDICES"));


//FIXME: Is this right?
  std::vector<El::Int> img_indices = m_strategy->get_image_indices();

  const AbsDistMat& layer_activations = layer.get_activations();

  CircMat<El::Device::CPU> all_images(
    layer_activations.Grid(), layer_activations.Root());
  all_images = layer_activations;

  if (all_images.CrossRank() == all_images.Root()) {
    auto const& local_images = all_images.LockedMatrix();
    auto dims = layer.get_output_dims();

    for (const El::Int& col_index : img_indices) {
      if (col_index >= local_images.Height())
        LBANN_ERROR(
          BuildErrorMessage(
            "col_index: ", col_index, " is greater than Matrix height: ",
            local_images.Height()));
      auto sample_index = sample_indices->Get(col_index, 0);
      auto image_tag =  get_tag(sample_index,epoch);
      auto const local_image = local_images(El::ALL, El::IR(col_index));

      this->m_summarizer->report_image(image_tag, m_img_format, local_image, dims, step);
    }
  }
}

std::string summarize_images::get_tag(El::Int index, El::Int epoch, size_t img_number){
  std::string image_tag;

  image_tag = "epoch: " + std::to_string(epoch) +
              "/ sample_index-" + std::to_string(index) +
              "/ image number: " + std::to_string(img_number);

  return image_tag;
}

Layer const* summarize_autoencoder_images::get_layer_by_name(
  const std::vector<Layer*>& layers,
  const std::string& layer_name)
{
  for(auto const* l : layers)
    if( l->get_name() == layer_name)
      return l;

  LBANN_ERROR(BuildErrorMessage("Layer named ", layer_name, " not found."));
  return nullptr;
}

std::unique_ptr<callback_base>
build_summarize_autoencoder_images_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>& summarizer) {

  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackSummarizeImages&>(proto_msg);

  return make_unique<summarize__images>(
    strategy,
    summarizer,
    params.img_layer(),
    params.input_layer(),
    params.interval());
}

}// namespace callback
}// namespace lbann
