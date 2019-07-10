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

#include <lbann_config.hpp>
//FIXME: Not turned on?
#ifdef LBANN_HAS_OPENCV
#include <opencv2/imgcodecs.hpp>
#endif // LBANN_HAS_OPENCV
#include <lbann/utils/summary.hpp>
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
//FIXME: pass in img limit?
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
  static size_t img_number = 0;

  const AbsDistMat& img_layer_activations = m_img_layer->get_activations();
  auto sample_view = std::unique_ptr<AbsDistMat>{img_layer_activations.Construct(
      img_layer_activations.Grid(), img_layer_activations.Root())};
  CircMat<El::Device::CPU> circ_image(
        img_layer_activations.Grid(), img_layer_activations.Root());

  //std::cout << "\n\nDumping " << img_indices.size() << " images!\n" << std::endl;
    for( const El::Int& col_index : img_indices ){
    El::LockedView(*sample_view, img_layer_activations, El::ALL, El::IR(col_index));
    circ_image = *sample_view;

    if(img_layer_activations.CrossRank() == img_layer_activations.Root())
    {
      auto const& local_image = circ_image.LockedMatrix();
      auto dims = m_img_layer->get_output_dims();

      this->m_summarizer->report_image("image-"+std::to_string(img_number++), local_image, dims, 0);
    }
  }
}

void lbann_callback_dump_image_results::on_batch_evaluate_end(model* m) {
  if (m_cat_accuracy_layer == nullptr) {
    auto layers = m->get_layers();
    /* find layers in model based on string */
    m_cat_accuracy_layer = get_layer_by_name(layers, m_cat_accuracy_layer_name);
    m_img_layer = get_layer_by_name(layers, m_img_layer_name);
  }

  // Check widths of img_layer.activations and cat_accuracy_layer are equal
  const AbsDistMat& cat_accuracy_activations = m_cat_accuracy_layer->get_activations();
  const AbsDistMat& img_layer_activations = m_img_layer->get_activations();
  if( cat_accuracy_activations.Width() != img_layer_activations.Width() )
    ThrowLBANNError("Invalid data. Categorical accuracy activations and image activations widths do not match.");
  std::vector<El::Int> img_indices = get_image_indices();

  //dump_image_to_summary(img_indices);
  save_image("saved_image", "jpg", img_indices);
  dump_image("dumped", "jpg", img_indices);
}

//FIXME: delete after cb success
void lbann_callback_dump_image_results::save_image(std::string prefix,
                                                   std::string format,
                                                   const std::vector<El::Int>& img_indices) {
#ifdef LBANN_HAS_OPENCV


    // Check that tensor dimensions are valid for images
    const auto& dims = m_img_layer->get_output_dims();
    El::Int num_channels(0), height(0), width(0);
    if (dims.size() == 2) {
      num_channels = 1;
      height = dims[0];
      width = dims[1];
    } else if (dims.size() == 3) {
      num_channels = dims[0];
      height = dims[1];
      width = dims[2];
    }
    if (!(num_channels == 1 || num_channels == 3)
        || height < 1 || width < 1) {
      std::stringstream err;
      err << "images are assumed to either be "
          << "2D tensors in HW format or 3D tensors in CHW format, "
          << "but the output of layer \"" << m_img_layer->get_name() << "\" "
          << "has dimensions ";
        for (size_t i = 0; i < dims.size(); ++i) {
          err << (i > 0 ? "" : " x ") << dims[i];
        }
      LBANN_ERROR(err.str());
    }

    // Get tensor data
    const auto& raw_data = m_img_layer->get_activations();
    std::unique_ptr<AbsDistMat> raw_data_v(raw_data.Construct(raw_data.Grid(), raw_data.Root()));
    for( const El::Int& col_index : img_indices ){
      El::LockedView(*raw_data_v, raw_data, El::ALL, El::IR(col_index));
      CircMat<El::Device::CPU> circ_data(raw_data_v->Grid(), raw_data_v->Root());
      circ_data = *raw_data_v;

      // Export tensor as image
      if (circ_data.CrossRank() == circ_data.Root()) {
        const auto& data = circ_data.LockedMatrix();

        // Data will be scaled to be in [0,256]
        DataType lower = data(0, 0);
        DataType upper = data(0, 0);
        for (El::Int i = 1; i < data.Height(); ++i) {
          lower = std::min(lower, data(i, 0));
          upper = std::max(upper, data(i, 0));
        }
        const auto& scale = ((upper > lower) ?
                             256 / (upper - lower) :
                             DataType(1));

        // Copy data into OpenCV matrix
        int type = -1;
        if (num_channels == 1) { type = CV_8UC1; }
        if (num_channels == 3) { type = CV_8UC3; }
        cv::Mat img(height, width, type);
        for (El::Int row = 0; row < height; ++row) {
          for (El::Int col = 0; col < width; ++col) {
            const auto& offset = row * width + col;
            if (num_channels == 1) {
              img.at<uchar>(row, col)
                = cv::saturate_cast<uchar>(scale * (data(offset, 0) - lower));
            } else if (num_channels == 3) {
              cv::Vec3b pixel;
              pixel[0] = cv::saturate_cast<uchar>(scale * (data(offset, 0) - lower));
              pixel[1] = cv::saturate_cast<uchar>(scale * (data(height*width + offset, 0) - lower));
              pixel[2] = cv::saturate_cast<uchar>(scale * (data(2*height*width + offset, 0) - lower));
              img.at<cv::Vec3b>(row, col) = pixel;
            }
          }
        }
        static size_t img_number = 0;
      // Write image to file
        cv::imwrite(prefix + "-" + std::to_string(img_number++) + "." + format, img);
      }
  }
#endif // LBANN_HAS_OPENCV
}

} // namespace lbann
