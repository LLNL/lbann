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

#include "lbann/transforms/vision/cutout.hpp"
#include "lbann/utils/opencv.hpp"

namespace lbann {
namespace transform {

void cutout::apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) {
  cv::Mat src = utils::get_opencv_mat(data, dims);
  for (size_t i = 0; i < m_num_holes; ++i) {
    // Select the center of the hole.
    const ssize_t center_x = transform::get_uniform_random_int(0, dims[2]);
    const ssize_t center_y = transform::get_uniform_random_int(0, dims[1]);
    // Compute top-left corner and bottom-right corners of the hole.
    const ssize_t length = static_cast<ssize_t>(m_length);
    const size_t x1 = std::max(center_x - length / 2, 0l);
    const size_t x2 = std::min(x1 + length, dims[2] - 1);
    const size_t y1 = std::max(center_y - length / 2, 0l);
    const size_t y2 = std::min(y1 + length, dims[1] - 1);
    // Convert to height/width.
    const size_t h = y2 - y1;
    const size_t w = x2 - x1;
    // Sanity check.
    if (x1 >= static_cast<size_t>(src.cols) ||
        y1 >= static_cast<size_t>(src.rows) ||
        (x1 + w) > static_cast<size_t>(src.cols) ||
        (y1 + h) > static_cast<size_t>(src.rows)) {
      std::stringstream ss;
      ss << "Bad hole dimensions for " << src.rows << "x" << src.cols << ": "
         << h << "x" << w << " at (" << x1 << "," << y1 << ")";
      LBANN_ERROR(ss.str());
    }
    // This will be just a view into the original.
    cv::Mat hole = src(cv::Rect(x1, y1, w, h));
    hole = 0;
  }
}

}  // namespace transform
}  // namespace lbann
