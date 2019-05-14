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
#include "lbann/transforms/vision/resized_center_crop.hpp"
#include "lbann/utils/opencv.hpp"

namespace lbann {
namespace transform {

void resized_center_crop::apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) {
  cv::Mat src = utils::get_opencv_mat(data, dims);
  std::vector<size_t> new_dims = {dims[0], m_crop_h, m_crop_w};
  auto dst_real = El::Matrix<uint8_t>(utils::get_linearized_size(new_dims), 1);
  cv::Mat dst = utils::get_opencv_mat(dst_real, new_dims);
  // This computes the projected crop area in the original image, crops it,
  // then resizes it.
  // Thus, we resize a smaller image, which is faster.
  // Method due to @JaeseungYeom.
  const float zoom = std::min(float(src.rows) / float(m_h),
                              float(src.cols) / float(m_w));
  const size_t zoom_h = m_crop_h*zoom;
  const size_t zoom_w = m_crop_w*zoom;
  const size_t x = std::round(float(src.cols - zoom_w) / 2.0f);
  const size_t y = std::round(float(src.rows - zoom_h) / 2.0f);
  // Sanity check.
  if (x >= static_cast<size_t>(src.cols) ||
      y >= static_cast<size_t>(src.rows) ||
      (x + zoom_w) > static_cast<size_t>(src.cols) ||
      (y + zoom_h) > static_cast<size_t>(src.rows)) {
    std::stringstream ss;
    ss << "Bad crop dimensions for " << src.rows << "x" << src.cols << ": "
       << zoom_h << "x" << zoom_w << " at (" << x << "," << y << ")";
    LBANN_ERROR(ss.str());
  }
  // The crop is just a view.
  cv::Mat tmp = src(cv::Rect(x, y, zoom_h, zoom_w));
  cv::resize(tmp, dst, dst.size(), 0, 0, cv::INTER_LINEAR);
  data.emplace<uint8_t>(std::move(dst_real));
  dims = new_dims;
}

}  // namespace transform
}  // namespace lbann
