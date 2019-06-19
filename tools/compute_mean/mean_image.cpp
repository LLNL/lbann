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
////////////////////////////////////////////////////////////////////////////////

#include <mpi.h>
#include <cmath>
#include <iostream>
#include <set>
#include "lbann/data_readers/opencv_extensions.hpp"
#include "file_utils.hpp"
#include "mean_image.hpp"

namespace tools_compute_mean {

/**
 *  Compute the pixel-wise sum of the local mean images over MPI ranks.
 *  If the numeric format of the pixel value is not floating point (SP/DP),
 *  convert it into the double-precision floating point between 0.0 and 1.0
 *  inclusively before reducing the values to avoid numeric overflow.
 */
cv::Mat reduce_mean_images(const cv::Mat mean_image, const mpi_states& ms) {
  cv::Mat local_data;

  if (!lbann::check_if_cv_Mat_is_float_type(mean_image)) {
    // convert non-floating point values to floating points
    const double alpha = lbann::get_depth_normalizing_factor(mean_image.depth());
    mean_image.convertTo(local_data, CV_64F, alpha, 0.0);
  } else if (!mean_image.isContinuous()) {
    // copy non-contiguous data into a contiguous memory block for easier
    // handling with MPI
    local_data = mean_image.clone();
  } else { // use the data block as is
    local_data = mean_image;
  }

  const unsigned int num_values = local_data.rows * local_data.cols * local_data.channels();
  if (num_values == 0u) {
    return cv::Mat();
  }

  cv::Mat global_sum(local_data.rows, local_data.cols, local_data.type());

  int mc = MPI_SUCCESS;
  mc = MPI_Allreduce(const_cast<void*>(_ConstVoidP(local_data.data)), _VoidP(global_sum.data), num_values,
                     ((local_data.depth() == CV_32F)? MPI_FLOAT : MPI_DOUBLE), MPI_SUM, ms.get_comm());
  ms.check_mpi(mc);
  return global_sum;
}


/**
 *  Divide the sum of mean images aggregated from every MPI ranks by the number
 *  of ranks and return the average.
 *  The average can be returned in the form of a visible image, in which the
 *  channel values are of an integral type..
 */
cv::Mat divide_sum_image(const cv::Mat& global_sum, const int num_ranks, const int type) {
  if (global_sum.empty() || (num_ranks <= 0)) {
    return cv::Mat();
  }

  static const std::set<int> depths = {CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F};

  if (depths.find(CV_MAT_DEPTH(type)) == depths.end()) {
    return cv::Mat();
  }

  cv::Mat global_mean;
  const double alpha = lbann::get_depth_denormalizing_factor(CV_MAT_DEPTH(type)) / num_ranks;
  global_sum.convertTo(global_mean, type, alpha, 0.0);

  return global_mean;
}


/**
 *  Reduce the mean of images, compute the average, and finally write the result
 *  into tow files. One is in a visible image format, and the other is in a
 *  binary file containing floating point values.
 */
bool write_mean_image(const lbann::cv_process& pp, const int mean_extractor_idx,
                      const mpi_states& ms, const std::string& out_dir) {

  if (dynamic_cast<const lbann::cv_mean_extractor *>(pp.get_transform(mean_extractor_idx)) == nullptr) {
    return false;
  }

  cv::Mat local_mean;
  local_mean = dynamic_cast<const lbann::cv_mean_extractor *>(pp.get_transform(mean_extractor_idx))->extract<void>();

  if (local_mean.empty()) {
    return false;
  }

  // reduce the sum of mean images across ranks
  cv::Mat global_sum = reduce_mean_images(local_mean, ms);

  if (ms.is_root()) { // write a visible image of mean data
    cv::Mat global_mean = divide_sum_image(global_sum, ms.get_effective_num_ranks(), CV_8UC3);
    cv::imwrite(out_dir + "mean.png", global_mean);
  }

  if (ms.is_root()) { // write binary image data (in the numeric format as is. e.g., floating point)
    cv::Mat global_mean  = divide_sum_image(global_sum, ms.get_effective_num_ranks(), local_mean.type());
    std::string image_name = out_dir + "mean-"
                             + std::to_string(global_mean.cols) + 'x'
                             + std::to_string(global_mean.rows) + 'x'
                             + std::to_string(global_mean.channels()) + '-'
                             + std::to_string(global_mean.depth()) + ".bin";

    const size_t sz =  global_mean.rows * global_mean.cols * global_mean.channels() * CV_ELEM_SIZE(global_mean.depth());

    // Either write into a pfm format image or a binary file (no handling of endianess)
    write_file(image_name, global_mean.data, sz);
  }

  return true;
}

} // end of namespace tools_compute_mean
