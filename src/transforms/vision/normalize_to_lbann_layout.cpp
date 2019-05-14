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

#include "lbann/transforms/vision/normalize_to_lbann_layout.hpp"
#include "lbann/utils/opencv.hpp"

namespace lbann {
namespace transform {

void normalize_to_lbann_layout::apply(utils::type_erased_matrix& data,
                                      std::vector<size_t>& dims) {
  auto dst = CPUMat(utils::get_linearized_size(dims), 1);
  apply(data, dst, dims);
  data.emplace<DataType>(std::move(dst));
}

void normalize_to_lbann_layout::apply(utils::type_erased_matrix& data,
                                      CPUMat& out,
                                      std::vector<size_t>& dims) {
  cv::Mat src = utils::get_opencv_mat(data, dims);
  if (!src.isContinuous()) {
    // This should not occur, but just in case.
    LBANN_ERROR("Do not support non-contiguous OpenCV matrices.");
  }
  // Ensure we have the right number of channels.
  if (dims.size() == 3 && m_means.size() != dims[0]) {
    LBANN_ERROR("Normalize channels does not match data");
  } else if (dims.size() != 3 && m_means.size() != 1) {
    LBANN_ERROR("Transform data has no channels, cannot normalize with multiple channels");
  }
  if (!out.Contiguous()) {
    LBANN_ERROR("NormalizeToLBANNLayout does not support non-contiguous destination.");
  }
  const uint8_t* __restrict__ src_buf = src.ptr();
  const size_t out_size = utils::get_linearized_size(dims);
  if (static_cast<size_t>(out.Height() * out.Width()) != out_size) {
    LBANN_ERROR("Transform output does not have sufficient space.");
  }
  DataType* __restrict__ dst_buf = out.Buffer();
  const float scale = 1.0f / 255.0f;
  if (dims[0] == 1) {
    // Greyscale.
    const DataType mean = m_means[0];
    const DataType std = m_stds[0];
    for (size_t row = 0; row < dims[1]; ++row) {
      for (size_t col = 0; col < dims[2]; ++col) {
        dst_buf[row + col*dims[1]] =
          (src_buf[row*dims[2] + col] * scale - mean) / std;
      }
    }
  } else {
    // RGB/three-channel.
    const size_t size = dims[1] * dims[2];
    for (size_t row = 0; row < dims[1]; ++row) {
      for (size_t col = 0; col < dims[2]; ++col) {
        // Multiply by 3 because there are three channels.
        const size_t src_base = 3*(row*dims[2] + col);
        const size_t dst_base = row + col*dims[1];
        dst_buf[dst_base] =
          (src_buf[src_base] * scale - m_means[0]) / m_stds[0];
        dst_buf[dst_base + size] =
          (src_buf[src_base + 1] * scale - m_means[1]) / m_stds[1];
        dst_buf[dst_base + 2*size] =
          (src_buf[src_base + 2] * scale - m_means[2]) / m_stds[2];
      }
    }
  }
}

}  // namespace transform
}  // namespace lbann
