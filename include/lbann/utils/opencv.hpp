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

#ifndef LBANN_UTILS_OPENCV_HPP_INCLUDED
#define LBANN_UTILS_OPENCV_HPP_INCLUDED

#include "lbann/utils/exception.hpp"
#include "lbann/utils/type_erased_matrix.hpp"
#include <opencv2/core.hpp>

namespace lbann {
namespace utils {

/**
 * Check whether data is an image.
 * Currently requires data to be a uint8_t CPUMat, with 3 dimensions, the first
 * (channel) being 1 or 3.
 *
 * @param data The data to check.
 * @param dims The dimensions associated with data.
 */
inline bool check_is_image(const utils::type_erased_matrix& data,
                           const std::vector<size_t>& dims) {
  try {
    // Check if we can do the conversion.
    const auto& unused = data.template get<uint8_t>();
    (void) unused;
  } catch (const utils::bad_any_cast&) {
    return false;
  }
  if (dims.size() != 3 || (dims[0] != 1 && dims[0] != 3)) {
    return false;
  }
  return true;
}

/**
 * Throw an error if data is not an image.
 * Currently requires data to be a uint8_t CPUMat, with 3 dimensions, the first
 * (channel) being 1 or 3.
 * Also throws an error if OpenCV is not supported.
 *
 * @param data The data to check.
 * @param dims The dimensions associated with data.
 */
inline void assert_is_image(const utils::type_erased_matrix& data,
                            const std::vector<size_t>& dims) {
  try {
    // Check if we can do the conversion.
    const auto& unused = data.template get<uint8_t>();
    (void) unused;
  } catch (const utils::bad_any_cast&) {
    LBANN_ERROR("Data is not an image: not uint8_t.");
  }
  if (dims.size() != 3 || (dims[0] != 1 && dims[0] != 3)) {
    LBANN_ERROR("Data is not an image: bad dims.");
  }
}

/**
 * Construct an OpenCV Mat that refers to data.
 * No data is copied, this just sets up a cv::Mat header.
 * @param data The matrix with data to use.
 * @param dims Dimensions of the data.
 */
inline cv::Mat get_opencv_mat(utils::type_erased_matrix& data, const std::vector<size_t>& dims) {
  assert_is_image(data, dims);
  auto& mat = data.template get<uint8_t>();
  return cv::Mat(dims[1], dims[2], dims[0] == 1 ? CV_8UC1 : CV_8UC3,
                 mat.Buffer());
}

/**
 * Construct an OpenCV Mat that refers to data.
 * No data is copied, this just sets up a cv::Mat header.
 * @param data The matrix with data to use.
 * @param dims Dimensions of the data.
 */
inline cv::Mat get_opencv_mat(El::Matrix<uint8_t>& data, const std::vector<size_t>& dims) {
  if (dims.size() != 3 || (dims[0] != 1 && dims[0] != 3)) {
    LBANN_ERROR("Data is not an image: bad dims.");
  }
  return cv::Mat(dims[1], dims[2], dims[0] == 1 ? CV_8UC1 : CV_8UC3,
                 data.Buffer());
}

/** Get the linearized size of dims. */
inline size_t get_linearized_size(const std::vector<size_t>& dims) {
  return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
}

}  // namespace utils
}  // namespace lbann

#endif  // LBANN_UTILS_OPENCV_HPP_INCLUDED
