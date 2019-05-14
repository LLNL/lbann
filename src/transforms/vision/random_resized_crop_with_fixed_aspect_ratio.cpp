////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include <opencv2/imgproc.hpp>
#include "lbann/transforms/vision/random_resized_crop_with_fixed_aspect_ratio.hpp"
#include "lbann/utils/opencv.hpp"

namespace lbann {
namespace transform {

void random_resized_crop_with_fixed_aspect_ratio::apply(
  utils::type_erased_matrix& data, std::vector<size_t>& dims) {
  cv::Mat src = utils::get_opencv_mat(data, dims);
  std::vector<size_t> new_dims = {dims[0], m_crop_h, m_crop_w};
  auto dst_real = El::Matrix<uint8_t>(utils::get_linearized_size(new_dims), 1);
  cv::Mat dst = utils::get_opencv_mat(dst_real, new_dims);
  // Compute the projected crop area in the original image, crop it, and resize.
  const float zoom = std::min(float(src.rows) / float(m_h),
                              float(src.cols) / float(m_w));
  const size_t zoom_h = m_h*zoom;
  const size_t zoom_w = m_w*zoom;
  const size_t zoom_crop_h = m_crop_h*zoom;
  const size_t zoom_crop_w = m_crop_w*zoom;
  const size_t dx = transform::get_uniform_random_int(
    0, 2*(zoom*m_w - zoom_crop_w) + 1);
  const size_t dy = transform::get_uniform_random_int(
    0, 2*(zoom*m_h - zoom_crop_h) + 1);
  const size_t x = (dims[2] - zoom_w + dx + 1) / 2;
  const size_t y = (dims[1] - zoom_h + dy + 1) / 2;
  // Sanity check.
  if (x >= static_cast<size_t>(src.cols) ||
      y >= static_cast<size_t>(src.rows) ||
      (x + zoom_crop_w) > static_cast<size_t>(src.cols) ||
      (y + zoom_crop_h) > static_cast<size_t>(src.rows)) {
    std::stringstream ss;
    ss << "Bad crop dimensions for " << src.rows << "x" << src.cols << ": "
       << zoom_crop_h << "x" << zoom_crop_w << " at (" << x << "," << y << ")";
    LBANN_ERROR(ss.str());
  }
  // The crop is just a view.
  cv::Mat tmp = src(cv::Rect(x, y, zoom_crop_h, zoom_crop_w));
  cv::resize(tmp, dst, dst.size(), 0, 0, cv::INTER_LINEAR);
  data.emplace<uint8_t>(std::move(dst_real));
  dims = new_dims;
}

}  // namespace transform
}  // namespace lbann
