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
#include "lbann/transforms/vision/adjust_saturation.hpp"
#include "lbann/utils/opencv.hpp"

namespace lbann {
namespace transform {

void adjust_saturation::apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) {
  // To adjust contrast, we essentially blend between the grayscale and
  // original image based on the given factor.
  cv::Mat src = utils::get_opencv_mat(data, dims);
  if (!src.isContinuous()) {
    // This should not occur, but just in case.
    LBANN_ERROR("Do not support non-contiguous OpenCV matrices.");
  }
  if (dims[0] == 1) {
    // Already grayscale, nothing to do.
  } else {
    // Handle RGB.
    // Get the grayscaled image.
    // If need be, we could do this computation in-place by manually computing
    // the grayscale value of each pixel.
    std::vector<size_t> gray_dims = {1, dims[1], dims[2]};
    const size_t gray_size = utils::get_linearized_size(gray_dims);
    auto gray_real = El::Matrix<uint8_t>(gray_size, 1);
    cv::Mat gray = utils::get_opencv_mat(gray_real, gray_dims);
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    const uint8_t* __restrict__ gray_buf = gray.ptr();
    // Mix the grayscale image with the original.
    uint8_t* __restrict__ src_buf = src.ptr();
    const float one_minus_factor = 1.0f - m_factor;
    for (size_t i = 0; i < gray_size; ++i) {
      // Handle the three channels, in OpenCV format.
      const size_t src_base = 3*i;
      src_buf[src_base] = cv::saturate_cast<uint8_t>(
        src_buf[src_base]*m_factor + gray_buf[i]*one_minus_factor);
      src_buf[src_base+1] = cv::saturate_cast<uint8_t>(
        src_buf[src_base+1]*m_factor + gray_buf[i]*one_minus_factor);
      src_buf[src_base+2] = cv::saturate_cast<uint8_t>(
        src_buf[src_base+2]*m_factor + gray_buf[i]*one_minus_factor);
    }
  }
}

}  // namespace transform
}  // namespace lbann
