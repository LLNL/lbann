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
#include "lbann/transforms/vision/random_resized_crop.hpp"
#include "lbann/utils/opencv.hpp"

namespace lbann {
namespace transform {

void random_resized_crop::apply(utils::type_erased_matrix& data,
                                std::vector<size_t>& dims) {
  cv::Mat src = utils::get_opencv_mat(data, dims);
  std::vector<size_t> new_dims = {dims[0], m_h, m_w};
  auto dst_real = El::Matrix<uint8_t>(utils::get_linearized_size(new_dims), 1);
  cv::Mat dst = utils::get_opencv_mat(dst_real, new_dims);
  size_t x = 0, y = 0, h = 0, w = 0;
  const size_t area = dims[1]*dims[2];
  // There's a chance this can fail, so we only make ten attempts.
  for (int attempt = 0; attempt < 10; ++attempt) {
    const float target_area = area*transform::get_uniform_random(m_scale_min,
                                                                 m_scale_max);
    const float target_ar = transform::get_uniform_random(m_ar_min, m_ar_max);
    w = std::sqrt(target_area * target_ar);
    h = std::sqrt(target_area / target_ar);
    // Swap these with 50% probability.
    if (transform::get_bool_random(0.5)) {
      std::swap(w, h);
    }
    if (w <= dims[2] && h <= dims[1]) {
      x = transform::get_uniform_random_int(0, dims[2] - w + 1);
      y = transform::get_uniform_random_int(0, dims[1] - h + 1);
      break;
    }
    // Reset.
    h = 0;
    w = 0;
  }
  bool fallback = false;
  // Fallback.
  if (h == 0) {
    fallback = true;
    w = std::min(dims[1], dims[2]);
    h = w;
    x = (dims[2] - w) / 2;
    y = (dims[1] - h) / 2;
  }
  // Sanity check.
  if (x >= static_cast<size_t>(src.cols) ||
      y >= static_cast<size_t>(src.rows) ||
      (x + w) > static_cast<size_t>(src.cols) ||
      (y + h) > static_cast<size_t>(src.rows)) {
    std::stringstream ss;
    ss << "Bad crop dimensions for " << src.rows << "x" << src.cols << ": "
       << h << "x" << w << " at (" << x << "," << y << ") fallback=" << fallback;
    LBANN_ERROR(ss.str());
  }
  // This is just a view.
  cv::Mat tmp = src(cv::Rect(x, y, w, h));
  cv::resize(tmp, dst, dst.size(), 0, 0, cv::INTER_LINEAR);
  // Sanity check.
  if (dst.ptr() != dst_real.Buffer()) {
    LBANN_ERROR("Did not resize into dst_real.");
  }
  data.emplace<uint8_t>(std::move(dst_real));
  dims = new_dims;
}

}  // namespace transform
}  // namespace lbann
