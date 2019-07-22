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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_save_images.hpp"
#include "lbann/proto/factories.hpp"

#include <callbacks.pb.h>

#ifdef LBANN_HAS_OPENCV
#include <opencv2/imgcodecs.hpp>
#endif // LBANN_HAS_OPENCV

namespace lbann {

namespace {

void save_image(std::string prefix,
                std::string format,
                const std::vector<Layer*>& layers,
                const std::vector<std::string>& layer_names) {
#ifdef LBANN_HAS_OPENCV
  for (const auto* l : layers) {

    // Only save outputs of layers in list
    const auto& name = l->get_name();
    if (std::find(layer_names.begin(), layer_names.end(), name)
        == layer_names.end()) {
      continue;
    }

    // Check that tensor dimensions are valid for images
    const auto& dims = l->get_output_dims();
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
          << "but the output of layer \"" << l->get_name() << "\" "
          << "has dimensions ";
        for (size_t i = 0; i < dims.size(); ++i) {
          err << (i > 0 ? "" : " x ") << dims[i];
        }
      LBANN_ERROR(err.str());
    }

    // Get tensor data
    const auto& raw_data = l->get_activations();
    std::unique_ptr<AbsDistMat> raw_data_v(raw_data.Construct(raw_data.Grid(), raw_data.Root()));
    El::LockedView(*raw_data_v, raw_data, El::ALL, El::IR(0));
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

      // Write image to file
      cv::imwrite(prefix + "-" + name + "." + format, img);

    }

  }
#endif // LBANN_HAS_OPENCV
}

} // namespace

lbann_callback_save_images::lbann_callback_save_images(std::vector<std::string> layer_names,
                                                       std::string image_format,
                                                       std::string image_prefix)
  : lbann_callback(),
    m_layer_names(std::move(layer_names)),
    m_image_format(image_format.empty() ? "jpg" : image_format),
    m_image_prefix(std::move(image_prefix)) {
#ifndef LBANN_HAS_OPENCV
  LBANN_ERROR("OpenCV not detected");
#endif // LBANN_HAS_OPENCV
}

void lbann_callback_save_images::on_epoch_end(model *m) {
  save_image(m_image_prefix + "epoch" + std::to_string(m->get_epoch()),
             m_image_format,
             m->get_layers(),
             m_layer_names);
}

void lbann_callback_save_images::on_test_end(model *m) {
  save_image(m_image_prefix + "test",
             m_image_format,
             m->get_layers(),
             m_layer_names);
}

std::unique_ptr<lbann_callback>
build_callback_save_images_from_pbuf(
  const google::protobuf::Message& proto_msg, lbann_summary*) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackSaveImages&>(proto_msg);
  return make_unique<lbann_callback_save_images>(
    parse_list<>(params.layers()),
    params.image_format(),
    params.image_prefix());
}

} // namespace lbann
