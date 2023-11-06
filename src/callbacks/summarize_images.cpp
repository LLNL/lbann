////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/callbacks/summarize_images.hpp"
#include "lbann/data_ingestion/coordinator/data_coordinator.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/layers/io/input_layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/trainers/trainer.hpp"
#include <lbann_config.hpp>

#include "lbann/utils/factory.hpp"
#include "lbann/utils/protobuf.hpp"
#include "lbann/utils/summary_impl.hpp"

#include "lbann/proto/callbacks.pb.h"

#include <iostream>

namespace lbann {
namespace callback {

// Strategy construction
namespace {

// Define the factory type.
using factory_type = lbann::generic_factory<
  image_output_strategy,
  std::string,
  generate_builder_type<image_output_strategy,
                        google::protobuf::Message const&>,
  default_key_error_policy>;

void register_default_builders(factory_type& factory)
{
  factory.register_builder("CategoricalAccuracyStrategy",
                           build_categorical_accuracy_strategy_from_pbuf);
  factory.register_builder("TrackSampleIDsStrategy",
                           build_track_sample_ids_strategy_from_pbuf);
}

// Manage a global factory
struct factory_manager
{
  factory_type factory_;

  factory_manager() { register_default_builders(factory_); }
};

factory_manager factory_mgr_;
factory_type const& get_strategy_factory() noexcept
{
  return factory_mgr_.factory_;
}

std::unique_ptr<image_output_strategy>
construct_strategy(google::protobuf::Message const& proto_msg)
{
  auto const& factory = get_strategy_factory();
  auto const& msg = protobuf::get_oneof_message(proto_msg, "strategy_type");
  return factory.create_object(msg.GetDescriptor()->name(), msg);
}

} // namespace

// categorical_accuracy_strategy
std::vector<std::pair<size_t, El::Int>>
categorical_accuracy_strategy::get_image_indices(model const& m) const
{
  static size_t img_counter = 0;
  static size_t epoch_counter = 0;
  auto const& exe_ctx =
    dynamic_cast<SGDExecutionContext const&>(m.get_execution_context());
  if (exe_ctx.get_epoch() > epoch_counter) {
    epoch_counter++;
    img_counter = 0;
  }
  std::vector<std::pair<size_t, El::Int>> img_indices;

  auto const& cat_accuracy_layer =
    get_layer_by_name(m, m_cat_accuracy_layer_name);

  const BaseDistMat& categorized_correctly_dist =
    cat_accuracy_layer.get_activations(
      *(cat_accuracy_layer.get_child_layers().front()));
  auto const& distdata = categorized_correctly_dist.DistData();
  CircMat<El::Device::CPU> categorized_correctly(*(distdata.grid),
                                                 distdata.root);
  El::Copy(categorized_correctly_dist, categorized_correctly);

  if (categorized_correctly.Height() != El::Int(1))
    LBANN_ERROR(
      "categorical_accuracy_strategry expected to find a tensor of size 1, ",
      "but found a tensor of size ",
      categorized_correctly.Height());

  // Fill return value if root process
  if (categorized_correctly.CrossRank() == categorized_correctly.Root()) {
    // Loop over all samples -- samples are the *width* of the matrix
    auto const num_samples = categorized_correctly.LocalWidth();
    for (auto sample = decltype(num_samples){0}; sample < num_samples;
         ++sample) {
      auto const& correctness_value =
        categorized_correctly.LockedMatrix()(0, sample);

      if ((correctness_value != DataType(0)) &&
          (correctness_value != DataType(1))) {
        LBANN_ERROR("Invalid data from ",
                    cat_accuracy_layer.get_name(),
                    ". Received ",
                    correctness_value,
                    ", expected 0 or 1.");
      }

      if (img_indices.size() > static_cast<size_t>(num_samples) ||
          img_counter >= m_num_images) {
        break;
      }

      if (meets_criteria(correctness_value)) {
        img_indices.push_back(std::make_pair(sample, El::Int(++img_counter)));
      }
    }
  }

  return img_indices;
}

void categorical_accuracy_strategy::write_strategy_proto(
  lbann_data::Callback_CallbackSummarizeImages& msg) const
{
  using ProtoMatchType = lbann_data::Callback::CallbackSummarizeImages::
    SelectionStrategy::CategoricalAccuracyStrategy;

  auto* strategy =
    msg.mutable_selection_strategy()->mutable_categorical_accuracy();
  strategy->set_accuracy_layer_name(m_cat_accuracy_layer_name);
  switch (m_match_type) {
  case MatchType::MATCH:
    strategy->set_match_type(ProtoMatchType::MATCH);
    break;
  case MatchType::NOMATCH:
    strategy->set_match_type(ProtoMatchType::NOMATCH);
    break;
  case MatchType::ALL:
    strategy->set_match_type(ProtoMatchType::ALL);
    break;
  default:
    LBANN_ERROR("Invalid MatchType");
  }
  strategy->set_num_images_per_epoch(m_num_images);
}

bool categorical_accuracy_strategy::meets_criteria(
  const DataType& match) const noexcept
{
  switch (m_match_type) {
  case MatchType::MATCH:
    return (match == 1);
  case MatchType::NOMATCH:
    return (match == 0);
  case MatchType::ALL:
    return true;
  }
  return false;
}

std::string
categorical_accuracy_strategy::get_tag(std::string const& layer_name,
                                       El::Int index,
                                       El::Int epoch) const
{
  // Sort by epoch
  return build_string("epoch ",
                      epoch,
                      "/layer: ",
                      layer_name,
                      "/sample_index-",
                      index);
}

// Builder function
std::unique_ptr<image_output_strategy>
build_categorical_accuracy_strategy_from_pbuf(
  google::protobuf::Message const& msg)
{
  using callback_type = lbann_data::Callback::CallbackSummarizeImages;
  using strategy_type =
    callback_type::SelectionStrategy::CategoricalAccuracyStrategy;
  using proto_match_type = strategy_type::MatchType;

  auto ConvertToLbannType = [](proto_match_type a) {
    return static_cast<categorical_accuracy_strategy::MatchType>(a);
  };

  const auto& strategy_msg = dynamic_cast<const strategy_type&>(msg);
  return std::make_unique<categorical_accuracy_strategy>(
    strategy_msg.accuracy_layer_name(),
    ConvertToLbannType(strategy_msg.match_type()),
    strategy_msg.num_images_per_epoch());
}
// End categorical_accuracy_strategy

std::vector<std::pair<size_t, El::Int>>
autoencoder_strategy::get_image_indices(model const& m) const
{
  std::vector<std::pair<size_t, El::Int>> img_indices;
  LBANN_ERROR("This feature is no longer supported");
  return img_indices;
}

std::string autoencoder_strategy::get_tag(std::string const& layer_name,
                                          El::Int index,
                                          El::Int epoch) const
{
  // Sort by index
  return build_string("image id ",
                      index,
                      "/layer: ",
                      layer_name,
                      "/epoch ",
                      epoch);
}

void autoencoder_strategy::write_strategy_proto(
  lbann_data::Callback_CallbackSummarizeImages& msg) const
{
  auto* strategy = msg.mutable_selection_strategy()->mutable_track_sample_ids();
  strategy->set_input_layer_name(m_input_layer_name);
  strategy->set_num_tracked_images(m_num_images);
}

// Builder function
std::unique_ptr<image_output_strategy>
build_track_sample_ids_strategy_from_pbuf(google::protobuf::Message const& msg)
{
  using callback_type = lbann_data::Callback::CallbackSummarizeImages;
  using strategy_type =
    callback_type::SelectionStrategy::TrackSampleIDsStrategy;

  const auto& strategy_msg = dynamic_cast<const strategy_type&>(msg);
  return std::make_unique<autoencoder_strategy>(
    strategy_msg.input_layer_name(),
    strategy_msg.num_tracked_images());
}
// End autoencoder strategy

summarize_images::summarize_images(
  std::shared_ptr<lbann_summary> const& summarizer,
  std::unique_ptr<image_output_strategy> strategy,
  std::string const& img_layer_name,
  uint64_t epoch_interval,
  std::string const& img_format)
  : callback_base(/*batch interval=*/1),
    m_summarizer(summarizer),
    m_strategy(std::move(strategy)),
    m_img_source_layer_name(img_layer_name),
    m_epoch_interval(std::max(epoch_interval, uint64_t{1})),
    m_img_format(img_format)
{
#ifndef LBANN_HAS_OPENCV
  LBANN_ERROR("OpenCV not detected");
#endif // LBANN_HAS_OPENCV
}

callback_base* summarize_images::copy() const
{
  LBANN_ERROR("This callback is not copyable.");
  return nullptr;
}

void summarize_images::on_batch_evaluate_end(model* m)
{

  auto const& exe_ctx =
    dynamic_cast<SGDExecutionContext const&>(m->get_execution_context());
  if (exe_ctx.get_epoch() % m_epoch_interval != 0)
    return;

  if (m->get_execution_context().get_execution_mode() ==
      execution_mode::validation)
    dump_images_to_summary(*m);
}

void summarize_images::write_specific_proto(lbann_data::Callback& proto) const
{
  auto* msg = proto.mutable_summarize_images();
  m_strategy->write_strategy_proto(*msg);
  msg->set_image_source_layer_name(m_img_source_layer_name);
  msg->set_epoch_interval(m_epoch_interval);
}

void summarize_images::dump_images_to_summary(model const& m) const
{

  auto img_indices = m_strategy->get_image_indices(m);

  const auto& layer = get_layer_by_name(m, m_img_source_layer_name);
  const auto& layer_activations =
    layer.get_activations(*(layer.get_child_layers().front()));
  const auto& layer_distdata = layer_activations.DistData();
  CircMat<El::Device::CPU> all_images(*(layer_distdata.grid),
                                      layer_distdata.root);
  El::Copy(layer_activations, all_images);

  if (all_images.CrossRank() == all_images.Root()) {
    auto const& local_images = all_images.LockedMatrix();
    auto dims = layer.get_output_dims();

    for (const auto& img_id : img_indices) {
      auto const& col_index = img_id.first;
      auto const& sample_index = img_id.second;
      if (col_index >= size_t(local_images.Width())) {
        LBANN_ERROR("Column index ",
                    col_index,
                    " is greater than Matrix width ",
                    local_images.Width());
      }
      auto const& exe_ctx =
        dynamic_cast<SGDExecutionContext const&>(m.get_execution_context());
      auto image_tag = m_strategy->get_tag(m_img_source_layer_name,
                                           sample_index,
                                           exe_ctx.get_epoch());
      auto const local_image = local_images(El::ALL, El::IR(col_index));
      this->m_summarizer->report_image(image_tag,
                                       m_img_format,
                                       local_image,
                                       dims,
                                       m.get_execution_context().get_step());
    }
  }
}

Layer const& get_layer_by_name(model const& m, std::string const& layer_name)
{
  for (El::Int ii = 0; ii < m.get_num_layers(); ++ii) {
    auto const& l = m.get_layer(ii);
    if (l.get_name() == layer_name)
      return l;
  }
  LBANN_ERROR("Did not find a layer with name \"", layer_name, "\" in model.");
  return m.get_layer(0); // Silence compiler warning
}

std::unique_ptr<callback_base> build_summarize_images_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>& summarizer)
{

  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackSummarizeImages&>(
      proto_msg);

  return std::make_unique<summarize_images>(
    summarizer,
    construct_strategy(params.selection_strategy()),
    params.image_source_layer_name(),
    params.epoch_interval());
}

} // namespace callback
} // namespace lbann
