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
// lbann_callback_dump_image_results .hpp .cpp - Callback hooks to dump
// results of image testing to event files
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_dump_image_results.hpp"
#include <iostream>

namespace lbann {

//FIXME: Should any of these params be const?
lbann_callback_dump_image_results::lbann_callback_dump_image_results(lbann_summary *summarizer,
                                                                     std::string const& cat_accuracy_layer_name,
                                                                     std::string const& img_layer_name,
                                                                     MatchType match_type,
                                                                     uint64_t interval,
                                                                     std::string img_format)
  : lbann_callback(1, summarizer),
    m_cat_accuracy_layer_name(cat_accuracy_layer_name),
    m_img_layer_name(img_layer_name),
    m_match_type(match_type),
    m_interval(interval),
    m_img_format(img_format) {

#ifndef LBANN_HAS_OPENCV
  LBANN_ERROR("OpenCV not detected");
#endif // LBANN_HAS_OPENCV
}
/** FIXME: Destructor error ?
lbann_callback_dump_image_results::~lbann_callback_dump_image_results(){
  delete m_summarizer;
}
*/

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

Layer const* lbann_callback_dump_image_results::get_layer_by_name(const std::vector<Layer*>& layers,
                                                                  const std::string& layer_name){

  for(auto const* l : layers)
      if( l->get_name() == layer_name)
        return l;

  ThrowLBANNError("Layer ", layer_name, " not found.");
  return nullptr;
}

std::vector<El::Int> lbann_callback_dump_image_results::get_image_indices() {
  const AbsDistMat& categorized_correctly = m_cat_accuracy_layer->get_activations();
  auto const& local_cat_correctly = categorized_correctly.LockedMatrix();
  El::AbstractMatrixReadDeviceProxy<DataType,El::Device::CPU> cpu_proxy(local_cat_correctly);
  CPUMat const& local_cpu_mat = cpu_proxy.GetLocked();

  std::vector<El::Int> img_indices;
  for(El::Int ii = 0; ii < local_cpu_mat.Width(); ++ii) {
    // Validate data
    if( local_cpu_mat(0,ii) != 0 && local_cpu_mat(0,ii) != 1 )
      ThrowLBANNError( "Invalid data from ", m_cat_accuracy_layer, ". Received ",
                       local_cpu_mat(0,ii), ", expected 0 or 1.");
    if( meets_criteria(local_cpu_mat(0,ii)) )
      img_indices.push_back(ii);
    if(img_indices.size() > 10)
      break;
  }
  return img_indices;
}

bool lbann_callback_dump_image_results::meets_criteria( const DataType& match ) {
  if( (match && (m_match_type == MatchType::MATCH)) ||
      (!match && (m_match_type == MatchType::NOMATCH)) ||
      (m_match_type == MatchType::ALL))
    return true;

  return false;

}

void lbann_callback_dump_image_results::dump_image_to_summary(const std::vector<El::Int>& img_indices) {

  const AbsDistMat& img_layer_activations = m_img_layer->get_activations();
  auto sample_view = std::unique_ptr<AbsDistMat>{img_layer_activations.Construct(
      img_layer_activations.Grid(), img_layer_activations.Root())};
  CircMat<El::Device::CPU> circ_image(
        img_layer_activations.Grid(), img_layer_activations.Root());
  for( const El::Int& col_index : img_indices ){
    El::LockedView(*sample_view, img_layer_activations, El::ALL, El::IR(col_index));
    circ_image = *sample_view;

    if(img_layer_activations.CrossRank() == img_layer_activations.Root())
    {

      auto const& local_image = circ_image.LockedMatrix();
      auto dims = m_img_layer->get_input_dims(col_index);
      m_summarizer->report_image("an image", local_image, dims, 0);

    }
  }
}

void lbann_callback_dump_image_results::on_batch_evaluate_end(model* m) {
  if (m_cat_accuracy_layer == nullptr) {
    auto layers = m->get_layers();
    /* find layers in model based on string */
    m_cat_accuracy_layer = get_layer_by_name(layers, m_cat_accuracy_layer_name);
    m_img_layer = get_layer_by_name(layers, m_img_layer_name);

    // Check widths of img_layer.activations and cat_accuracy_layer are equal
    const AbsDistMat& cat_accuracy_activations = m_cat_accuracy_layer->get_activations();
    const AbsDistMat& img_layer_activations = m_img_layer->get_activations();
    if( cat_accuracy_activations.Width() != img_layer_activations.Width() )
      ThrowLBANNError("Invalid data. Categorical accuracy activations and image activations widths do not match.");
    std::vector<El::Int> img_indices = get_image_indices();

    dump_image_to_summary(img_indices);

  }
}


} // namespace lbann
