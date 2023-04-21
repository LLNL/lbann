////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/transforms/repack_HWC_to_CHW_layout.hpp"
#include "lbann/utils/dim_helpers.hpp"

namespace lbann {
namespace transform {

void repack_HWC_to_CHW_layout::apply(utils::type_erased_matrix& data,
                                     std::vector<size_t>& dims)
{
  auto dst = CPUMat(get_linear_size(dims), 1);
  apply(data, dst, dims);
  data.emplace<DataType>(std::move(dst));
}

void repack_HWC_to_CHW_layout::apply(utils::type_erased_matrix& data,
                                     CPUMat& out,
                                     std::vector<size_t>& dims)
{
  CPUMat& src = data.template get<DataType>();
  if (!src.Contiguous()) {
    LBANN_ERROR("RepackHWCtoCHWLayout does not support non-contiguous src.");
  }
  if (!out.Contiguous()) {
    LBANN_ERROR(
      "RepackHWCtoCHWLayout does not support non-contiguous destination.");
  }
  const DataType* __restrict__ src_buf = src.LockedBuffer();
  const size_t out_size = get_linear_size(dims);
  if (static_cast<size_t>(out.Height() * out.Width()) != out_size) {
    LBANN_ERROR("Transform output does not have sufficient space.");
  }
  DataType* __restrict__ dst_buf = out.Buffer();
  // Pack an interleave multi-channel data structure into a
  // channel-strided data structure
  repack_HWC_to_CHW(src_buf, dst_buf, dims);
}

} // namespace transform
} // namespace lbann
