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

#include <lbann_config.hpp>
#include <lbann/utils/image.hpp>
#include <lbann/utils/summary.hpp>
#include "lbann/callbacks/summarize_autoencoder_images.hpp"
#include <lbann/layers/io/input/generic_input_layer.hpp>

#include <callbacks.pb.h>

#include <iostream>

namespace lbann {
namespace callback {

summarize_autoencoder_images::summarize_autoencoder_images(
  std::shared_ptr<lbann_summary> const& summarizer,
  std::string const& reconstruction_layer_name,
  std::string const& img_layer_name,
  std::string const& input_layer_name,
  uint64_t interval,
  std::string const& img_format,
  size_t const& num_images)
  : callback_base(1),
    m_summarizer(summarizer),
    m_reconstruction_layer_name(reconstruction_layer_name),
    m_img_layer_name(img_layer_name),
    m_input_layer_name(input_layer_name),
    m_interval(interval),
    m_img_format(img_format),
    m_num_images(num_images)
{
#ifndef LBANN_HAS_OPENCV
  LBANN_ERROR("OpenCV not detected");
#endif // LBANN_HAS_OPENCV
}

namespace
{
template <typename... Ts>
std::string BuildErrorMessage(Ts... args)
{
  std::ostringstream oss;
  int dummy[] = { (oss << args, 0)... };
  (void) dummy;
  LBANN_ERROR(oss.str());
}
}

void summarize_autoencoder_images::on_batch_evaluate_end(model* m) {


  if (m->get_epoch() % m_interval != 0)
    return;

  if (m_reconstruction_layer == nullptr) {
    setup(m);
  }

  dump_images_to_summary(*m_reconstruction_layer, m->get_step(), m->get_epoch());

  if(m->get_epoch() > 1)
    dump_images_to_summary(*m_img_layer, m->get_step());
}

void summarize_autoencoder_images::setup(model* m)
{
  auto layers = m->get_layers();
  /* find layers in model based on string */
  m_reconstruction_layer = get_layer_by_name(layers, m_reconstruction_layer_name);
  if (m_reconstruction_layer == nullptr)
    LBANN_ERROR(BuildErrorMessage("get_layer_by_name() failed on layer ",
                                  m_reconstruction_layer_name));

  m_img_layer = get_layer_by_name(layers, m_img_layer_name);
  if (m_img_layer == nullptr)
    LBANN_ERROR(BuildErrorMessage("get_layer_by_name() failed on layer ", m_img_layer_name));

  m_input_layer = get_layer_by_name(layers, m_input_layer_name);
  if (m_input_layer == nullptr)
    LBANN_ERROR(BuildErrorMessage("get_layer_by_name() failed on layer ", m_input_layer_name));

  // Check widths of img_layer.activations and reconstruction_layer are equal
  const AbsDistMat& reconstruction_activations = m_reconstruction_layer->get_activations();
  const AbsDistMat& img_layer_activations = m_img_layer->get_activations();
  if( reconstruction_activations.Width() != img_layer_activations.Width() )
    LBANN_ERROR(
      BuildErrorMessage(
        "Invalid data. Reconstruction layer activations and image activations widths "
        "do not match."));

  if (auto gil = dynamic_cast<generic_input_layer const*>(m_input_layer)){
    m_num_images = std::min(static_cast<long>(m_num_images),
                            gil->get_dataset(execution_mode::validation).get_total_samples());
    m_mini_batch_size = gil->get_current_mini_batch_size();
  }
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

std::vector<El::Int> summarize_autoencoder_images::get_image_indices() {
  std::vector<El::Int> img_indices;

  auto* sample_indices =
    const_cast<Layer&>(*m_input_layer).get_sample_indices_per_mb();
  if (sample_indices == nullptr)
    LBANN_ERROR(BuildErrorMessage("NULL SAMPLE INDICES"));

  for(El::Int ii = 0; ii < sample_indices->Height(); ii++){
    if (ii >= sample_indices->Height())
      LBANN_ERROR(
        BuildErrorMessage(
          "col_index: ", ii, " is greater than Matrix height: ",
          sample_indices->Height()));

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

void summarize_autoencoder_images::dump_images_to_summary(
  const Layer& layer, const uint64_t& step, const El::Int& epoch) {

  std::vector<El::Int> img_indices = get_image_indices();

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

      auto sample_indices = const_cast<Layer&>(*m_input_layer).get_sample_indices_per_mb();
      auto sample_index = sample_indices->Get(col_index, 0);
      auto image_tag =  get_tag(sample_index,epoch);
      auto const local_image = local_images(El::ALL, El::IR(col_index));

      this->m_summarizer->report_image(image_tag, m_img_format, local_image, dims, step);
    }
  }
}

std::string summarize_autoencoder_images::get_tag(El::Int index, El::Int epoch){
  std::string image_tag;

  if(epoch == -1)
    image_tag = "sample_index-" + std::to_string(index) + "/ (original_image)";
  else
    image_tag = "sample_index-" + std::to_string(index) + "/ epoch-" + std::to_string(epoch);

  return image_tag;
}

std::unique_ptr<callback_base>
build_summarize_autoencoder_images_callback_from_pbuf(
  const google::protobuf::Message& proto_msg,
  const std::shared_ptr<lbann_summary>& summarizer) {

  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackSummarizeAutoencoderImages&>(proto_msg);

  return make_unique<summarize_autoencoder_images>(
    summarizer,
    params.reconstruction_layer(),
    params.image_layer(),
    params.input_layer(),
    params.interval());
}

} // namespace callback
} // namespace lbann
