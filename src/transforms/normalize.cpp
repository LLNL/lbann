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

#include "lbann/transforms/normalize.hpp"

namespace lbann {
namespace transform {

void normalize::apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) {
  // Ensure we have the right number of channels.
  if (dims.size() == 3 && m_means.size() != dims[0]) {
    LBANN_ERROR("Normalize channels does not match data");
  } else if (dims.size() != 3 && m_means.size() != 1) {
    LBANN_ERROR("Transform data has no channels, cannot normalize with multiple channels");
  }
  // Only work with DataTypes to avoid rounding/floating point issues.
  auto& mat = data.template get<DataType>();
  if (mat.Height() != mat.LDim()) {
    LBANN_ERROR("Normalizing non-contiguous matrix not supported");
  }
  DataType* __restrict__ buf = mat.Buffer();
  if (m_means.size() == 1) {
    const DataType mean = m_means[0];
    const DataType std = m_stds[0];
    const El::Int size = mat.Height() * mat.Width();
    for (El::Int i = 0; i < size; ++i) {
      buf[i] = (buf[i] - mean) / std;
    }
  } else {
    for (size_t channel = 0; channel < dims[0]; ++channel) {
      const DataType mean = m_means[channel];
      const DataType std = m_stds[channel];
      const size_t size = dims[1] * dims[2];
      const size_t channel_start = channel*size;
      const size_t channel_end = channel_start + size;
      for (size_t i = channel_start; i < channel_end; ++i) {
        buf[i] = (buf[i] - mean) / std;
      }
    }
  }
}

void normalize::apply(utils::type_erased_matrix& data, CPUMat& out,
                      std::vector<size_t>& dims) {
  // Ensure we have the right number of channels.
  if (dims.size() == 3 && m_means.size() != dims[0]) {
    LBANN_ERROR("Normalize channels does not match data");
  } else if (dims.size() != 3 && m_means.size() != 1) {
    LBANN_ERROR("Transform data has no channels, cannot normalize with multiple channels");
  }
  if (out.Height() != out.LDim()) {
    LBANN_ERROR("Normalizing to non-contiguous matrix not supported.");
  }
  const auto& src = data.template get<DataType>();
  if (src.Height() != src.LDim()) {
    LBANN_ERROR("Normalizing from non-contiguous matrix not supported.");
  }
  const DataType* __restrict__ src_buf = src.LockedBuffer();
  DataType* __restrict__ dst_buf = out.Buffer();
  if (m_means.size() == 1) {
    const DataType mean = m_means[0];
    const DataType std = m_stds[0];
    const El::Int size = src.Height() * src.Width();
    for (El::Int i = 0; i < size; ++i) {
      dst_buf[i] = (src_buf[i] - mean) / std;
    }
  } else {
    for (size_t channel = 0; channel < dims[0]; ++channel) {
      const DataType mean = m_means[channel];
      const DataType std = m_stds[channel];
      const size_t size = dims[1] * dims[2];
      const size_t channel_start = channel*size;
      const size_t channel_end = channel_start + size;
      for (size_t i = channel_start; i < channel_end; ++i) {
        dst_buf[i] = (src_buf[i] - mean) / std;
      }
    }
  }
}

}  // namespace transform
}  // namespace lbann
