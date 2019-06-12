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

#include "lbann/transforms/vision/random_crop.hpp"
#include "lbann/utils/opencv.hpp"

namespace lbann {
namespace transform {

void random_crop::apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) {
  cv::Mat src = utils::get_opencv_mat(data, dims);
  if (dims[1] <= m_h || dims[2] <= m_w) {
    std::stringstream ss;
    ss << "Random crop to " << m_h << "x" << m_w
       << " applied to input " << dims[1] << "x" << dims[2];
    LBANN_ERROR(ss.str());
  }
  std::vector<size_t> new_dims = {dims[0], m_h, m_w};
  auto dst_real = El::Matrix<uint8_t>(utils::get_linearized_size(new_dims), 1);
  cv::Mat dst = utils::get_opencv_mat(dst_real, new_dims);
  // Select the upper-left corner of the crop.
  const size_t x = transform::get_uniform_random_int(0, dims[2] - m_w + 1);
  const size_t y = transform::get_uniform_random_int(0, dims[1] - m_h + 1);
  // Sanity check.
  if (x >= static_cast<size_t>(src.cols) ||
      y >= static_cast<size_t>(src.rows) ||
      (x + m_w) > static_cast<size_t>(src.cols) ||
      (y + m_h) > static_cast<size_t>(src.rows)) {
    std::stringstream ss;
    ss << "Bad crop dimensions for " << src.rows << "x" << src.cols << ": "
       << m_h << "x" << m_w << " at (" << x << "," << y << ")";
    LBANN_ERROR(ss.str());
  }
  // Copy is needed to ensure this is continuous.
  src(cv::Rect(x, y, m_h, m_w)).copyTo(dst);
  data.emplace<uint8_t>(std::move(dst_real));
  dims = new_dims;
}

}  // namespace transform
}  // namespace lbann
