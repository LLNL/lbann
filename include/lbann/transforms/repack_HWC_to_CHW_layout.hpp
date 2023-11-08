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

#ifndef LBANN_TRANSFORMS_REPACK_HWC_TO_CHW_LAYOUT_HPP_INCLUDED
#define LBANN_TRANSFORMS_REPACK_HWC_TO_CHW_LAYOUT_HPP_INCLUDED

#include "lbann/transforms/transform.hpp"

namespace lbann {
namespace transform {

/**
 * Convert data to LBANN's native data layout.
 * Currently only supports converting from and interleaved channel format.
 */
class repack_HWC_to_CHW_layout : public transform
{
public:
  transform* copy() const override
  {
    return new repack_HWC_to_CHW_layout(*this);
  }

  std::string get_type() const override { return "to_lbann_layout"; }

  bool supports_non_inplace() const override { return true; }

  void apply(utils::type_erased_matrix& data,
             std::vector<size_t>& dims) override;

  void apply(utils::type_erased_matrix& data,
             CPUMat& out,
             std::vector<size_t>& dims) override;
};

template <typename T>
void repack_HWC_to_CHW(T const* hwc_src,
                       T* chw_dest,
                       std::vector<size_t> const& chw_dims)
{
  auto const num_channels = chw_dims[0];
  auto const img_height = chw_dims[1];
  auto const img_width = chw_dims[2];
  auto const img_size = img_height * img_width;

  // The image is stored row-major, so the width is actually varying fastest.
  for (size_t row = 0; row < img_height; ++row) {
    for (size_t col = 0; col < img_width; ++col) {
      for (size_t chan = 0; chan < num_channels; ++chan) {
        auto const dst_offset = chan * img_size + row * img_width + col;
        auto const src_offset =
          row * img_width * num_channels + col * num_channels + chan;
        chw_dest[dst_offset] = hwc_src[src_offset];
      }
    }
  }
}

template <typename T>
void repack_DHWC_to_CDHW(T const* dhwc_src,
                         T* cdhw_dest,
                         std::vector<size_t> const& cdhw_dims)
{
  auto const num_channels = cdhw_dims[0];
  auto const img_depth = cdhw_dims[1];
  auto const img_height = cdhw_dims[2];
  auto const img_width = cdhw_dims[3];
  auto const plane_size = img_height * img_width;
  auto const volume_size = img_depth * img_height * img_width;
  for (size_t depth = 0; depth < img_depth; ++depth) {
    for (size_t height = 0; height < img_height; ++height) {
      for (size_t width = 0; width < img_width; ++width) {
        for (size_t chan = 0; chan < num_channels; ++chan) {
          auto const dst_offset = chan * volume_size + depth * plane_size +
                                  height * img_width + width;
          auto const src_offset = depth * plane_size * num_channels +
                                  height * img_width * num_channels +
                                  width * num_channels + chan;
          cdhw_dest[dst_offset] = dhwc_src[src_offset];
        }
      }
    }
  }
}

} // namespace transform
} // namespace lbann

#endif // LBANN_TRANSFORMS_REPACK_HWC_TO_CHW_LAYOUT_HPP_INCLUDED
