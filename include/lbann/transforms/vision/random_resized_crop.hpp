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

#ifndef LBANN_TRANSFORMS_RANDOM_RESIZED_CROP_HPP_INCLUDED
#define LBANN_TRANSFORMS_RANDOM_RESIZED_CROP_HPP_INCLUDED

#include "lbann/transforms/transform.hpp"

namespace lbann {
namespace transform {

/**
 * Extract a crop of random size and aspect ratio, then crop to a size.
 * This is commonly used for Inception-style networks and some other
 * image classification networks.
 */
class random_resized_crop : public transform {
public:
  /**
   * Crop to a random size and aspect ratio, then resize to h x w.
   * The random crop has area in [scale_min, scale_max] of the original image
   * area, and aspect ratio in [ar_min, ar_max] of the original. This random
   * crop is then resized to be h x w.
   * These default to (0.08, 1.0) and (3/4, 4/3), respectively, which are the
   * standard.
   */
  random_resized_crop(size_t h, size_t w,
                      float scale_min=0.08, float scale_max=1.0,
                      float ar_min=0.75, float ar_max=4.0f/3.0f) :
    transform(),
    m_h(h), m_w(w),
    m_scale_min(scale_min), m_scale_max(scale_max),
    m_ar_min(ar_min), m_ar_max(ar_max) {}

  transform* copy() const override { return new random_resized_crop(*this); }

  std::string get_type() const override { return "random_resized_crop"; }

  void apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) override;
private:
  /** Height and width of the final crop. */
  size_t m_h, m_w;
  /** Range for the area of the random crop. */
  float m_scale_min, m_scale_max;
  /** Range for the aspect ratio of the random crop. */
  float m_ar_min, m_ar_max;
};

}  // namespace transform
}  // namespace lbann

#endif  // LBANN_TRANSFORMS_RANDOM_RESIZED_CROP_HPP_INCLUDED
