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
#include "lbann/transforms/vision/colorize.hpp"
#include "lbann/utils/opencv.hpp"

namespace lbann {
namespace transform {

void colorize::apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) {
  cv::Mat src = utils::get_opencv_mat(data, dims);
  if (dims[0] != 1) {
    return;  // Already color.
  }
  std::vector<size_t> new_dims = {3, dims[1], dims[2]};
  auto dst_real = El::Matrix<uint8_t>(utils::get_linearized_size(new_dims), 1);
  cv::Mat dst = utils::get_opencv_mat(dst_real, new_dims);
  cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
  data.emplace<uint8_t>(std::move(dst_real));
  dims = new_dims;
}

}  // namespace transform
}  // namespace lbann
