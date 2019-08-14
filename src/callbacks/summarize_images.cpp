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
#include "lbann/callbacks/summarize_images.hpp"

#include <lbann/layers/io/input/generic_input_layer.hpp>
#include <lbann/proto/helpers.hpp>
#include <lbann/utils/factory.hpp>
#include <lbann/utils/image.hpp>
#include <lbann/utils/summary.hpp>

#include <callbacks.pb.h>

#include <iostream>

namespace lbann {
namespace callback {

// Strategy construction
namespace {

// Define the factory type.
using factory_type = lbann::generic_factory<
  image_output_strategy,
  std::string,
  proto::generate_builder_type<image_output_strategy,
                               google::protobuf::Message const&>,
  default_key_error_policy>;

void register_default_builders(factory_type& factory) {
  factory.register_builder("CategoricalAccuracyStrategy",
                           build_categorical_accuracy_strategy_from_pbuf);
  factory.register_builder("AutoencoderStrategy",
                           build_autoencoder_strategy_from_pbuf);
}

// Manage a global factory
struct factory_manager {
  factory_type factory_;

  factory_manager() {
    register_default_builders(factory_);
  }
};

factory_manager factory_mgr_;
factory_type const& get_strategy_factory() noexcept {
  return factory_mgr_.factory_;
}

std::unique_ptr<image_output_strategy>
construct_strategy(google::protobuf::Message const& proto_msg) {
  auto const& factory = get_strategy_factory();
  auto const& msg =
    proto::helpers::get_oneof_message(proto_msg, "strategy_type");
  return factory.create_object(msg.GetDescriptor()->name(), msg);
}

}// namespace

std::vector<El::Int> categorical_accuracy_strategy::get_image_indices(model const& m) {

  static size_t img_counter = 0;
  static size_t epoch_counter = 0;
  if(static_cast<size_t>(m.get_epoch()) > epoch_counter){
    epoch_counter++;
    img_counter = 0;
  }
  std::vector<El::Int> img_indices;

  if(!m_cat_accuracy_layer)
    m_cat_accuracy_layer = get_layer_by_name(m, m_cat_accuracy_layer_name);

  const AbsDistMat& categorized_correctly_dist = m_cat_accuracy_layer->get_activations();
  CircMat<El::Device::CPU> categorized_correctly(
    categorized_correctly_dist.Grid(), categorized_correctly_dist.Root());
  categorized_correctly = categorized_correctly_dist;

//FIXME: Should width of img_layer and accuracy_layer activations be tested here?

  if (categorized_correctly.Height() != El::Int(1))
    LBANN_ERROR("Tom was wrong about this matrix. Oops.");

  // Fill return value if root process
  if (categorized_correctly.CrossRank() == categorized_correctly.Root()) {
    // Loop over all samples -- samples are the *width* of the matrix
    auto const num_samples = categorized_correctly.LocalWidth();
    for (auto sample = decltype(num_samples){0}; sample < num_samples; ++sample) {
      auto const& correctness_value = categorized_correctly.LockedMatrix()(0, sample);

      if ((correctness_value != DataType(0))
          && (correctness_value != DataType(1)))
        LBANN_ERROR("Invalid data from ", m_cat_accuracy_layer->get_name(),
                        ". Received ", correctness_value, ", expected 0 or 1.");

      if (meets_criteria(correctness_value)){
        img_indices.push_back(sample);
        img_counter++;
      }

      if(img_indices.size() > static_cast<size_t>(num_samples) || img_counter >= m_num_images)
        break;
    }
  }

  return img_indices;

}

bool categorical_accuracy_strategy::meets_criteria( const DataType& match ) {
  if( (match && (m_match_type == MatchType::MATCH)) ||
      (!match && (m_match_type == MatchType::NOMATCH)) ||
      (m_match_type == MatchType::ALL))
    return true;

  return false;

}

// Builder function
std::unique_ptr<image_output_strategy>
build_categorical_accuracy_strategy_from_pbuf(google::protobuf::Message const& msg) {
  using callback_type = lbann_data::Callback::CallbackSummarizeImages;
  using strategy_type = callback_type::SelectionStrategy::CategoricalAccuracyStrategy;
  using proto_match_type = strategy_type::MatchType;

  auto ConvertToLbannType = [](proto_match_type a) {
    return static_cast<categorical_accuracy_strategy::MatchType>(a);
  };

  const auto& strategy_msg = dynamic_cast<const strategy_type&>(msg);
  return make_unique<categorical_accuracy_strategy>(
    strategy_msg.accuracy_layer_name(),
    ConvertToLbannType(strategy_msg.match_type()),
    strategy_msg.num_images_per_epoch());
}

std::vector<El::Int> autoencoder_strategy::get_image_indices(model const& m) {

  auto input_layer = get_layer_by_name(m, m_input_layer_name);

  std::vector<El::Int> img_indices;
  auto* sample_indices =
    const_cast<Layer&>(*input_layer).get_sample_indices_per_mb();
  if (sample_indices == nullptr)
    LBANN_ERROR("NULL SAMPLE INDICES");

  for(El::Int ii = 0; ii < sample_indices->Height(); ii++){
    if (ii >= sample_indices->Height())
      LBANN_ERROR(
          "col_index: ", ii, " is greater than Matrix height: ",
          sample_indices->Height());

    if (m_tracked_images.find(sample_indices->Get(ii,0)) != m_tracked_images.end()){
      std::cout << "I found a tracked index! Idx = " << sample_indices->Get(ii,0)
                << "\n";
      img_indices.push_back(ii);
    }
    else if(m_tracked_images.size() < m_num_images){
      m_tracked_images.insert(sample_indices->Get(ii,0));
      std::cout << "Adding to tracked indices Idx = " << sample_indices->Get(ii,0)
                << "\n";
      img_indices.push_back(ii);
    }

    flush(std::cout);
  }
  return img_indices;

}

// Builder function
std::unique_ptr<image_output_strategy>
build_autoencoder_strategy_from_pbuf(google::protobuf::Message const& msg) {
  using callback_type = lbann_data::Callback::CallbackSummarizeImages;
  using strategy_type = callback_type::SelectionStrategy::AutoencoderStrategy;

  const auto& strategy_msg = dynamic_cast<const strategy_type&>(msg);
  return make_unique<autoencoder_strategy>(
    strategy_msg.input_layer_name(),
    strategy_msg.num_tracked_images());
}

summarize_images::summarize_images(std::shared_ptr<lbann_summary> const& summarizer,
                                   std::shared_ptr<image_output_strategy> const& strategy,
                                   std::string const& img_layer_name,
                                   std::string const& input_layer_name,
                                   uint64_t interval,
                                   uint64_t num_images,
                                   std::string const& img_format)
  : callback_base(1),
    m_summarizer(summarizer),
    m_strategy(strategy),
    m_img_source_layer_name(img_layer_name),
    m_input_layer_name(input_layer_name),
    m_interval(interval),
    m_num_images(num_images),
    m_img_format(img_format)

{
#ifndef LBANN_HAS_OPENCV
  LBANN_ERROR("OpenCV not detected");
#endif // LBANN_HAS_OPENCV
}

void summarize_images::on_batch_evaluate_end(model* m) {


  if (m->get_epoch() % m_interval != 0)
    return;

  if (m_img_source_layer == nullptr) {
    setup(m);
  }

  dump_images_to_summary(*m_img_source_layer, m->get_step(), m->get_epoch(), *m);
// FIXME: Dump original image for Autoencoder Strategy
//  if(m->get_epoch() > 1)
//    dump_images_to_summary(*m_img_layer, m->get_step());
}

void summarize_images::setup(model* m)
{
  /* find layers in model based on string */
  m_img_source_layer = get_layer_by_name(*m, m_img_source_layer_name);
  if (m_img_source_layer == nullptr)
    LBANN_ERROR("get_layer_by_name() failed on layer ",
                                  m_img_source_layer_name);

  m_input_layer = get_layer_by_name(*m, m_input_layer_name);
  if (m_input_layer == nullptr)
    LBANN_ERROR("get_layer_by_name() failed on layer ", m_input_layer_name);

//FIXME: Does this make sense? Error is supposed to catch reconstruction/img layer.
//       Should this be moved somewhere else for autoencoder strategy?
  // Check widths of img_layer.activations and reconstruction_layer are equal
  const AbsDistMat& img_source_activations = m_img_source_layer->get_activations();
  const AbsDistMat& input_layer_activations = m_input_layer->get_activations();
  if( img_source_activations.Width() != input_layer_activations.Width() )
    LBANN_ERROR(
        "Invalid data. Reconstruction layer activations and image activations widths "
        "do not match.");

  if (auto gil = dynamic_cast<generic_input_layer const*>(m_input_layer))
    m_num_images = std::min(static_cast<long>(m_num_images),
                            gil->get_dataset(execution_mode::validation).get_total_samples());

}

void summarize_images::dump_images_to_summary(
  Layer const& layer, uint64_t const& step, El::Int const& epoch, model const& m) {

  if(!m_input_layer)
    m_input_layer = get_layer_by_name(m, m_input_layer_name);

  auto sample_indices = const_cast<Layer&>(*m_input_layer).get_sample_indices_per_mb();
  if (sample_indices == nullptr)
    LBANN_ERROR("NULL SAMPLE INDICES");


//FIXME: Is this right?
  std::vector<El::Int> img_indices = m_strategy->get_image_indices(m);

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
            "col_index: ", col_index, " is greater than Matrix height: ",
            local_images.Height());
      auto sample_index = sample_indices->Get(col_index, 0);
      auto image_tag =  get_tag(sample_index, epoch);
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

Layer const* get_layer_by_name(model const& m,
                               std::string const& layer_name)
{

  auto layers = m.get_layers();

  for(auto const* l : layers)
    if( l->get_name() == layer_name)
      return l;

  LBANN_ERROR("Layer named ", layer_name, " not found.");
  return nullptr;
}

std::unique_ptr<callback_base>
build_summarize_images_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>& summarizer) {

  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackSummarizeImages&>(proto_msg);

//FIXME: Correct params?
  return make_unique<summarize_images>(
    summarizer,
    construct_strategy(params.selection_strategy()),
    params.image_source_layer_name(),
    params.input_layer_name(),
    params.epoch_interval());
}

}// interval callback
}// namespace lbann
