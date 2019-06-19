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

#ifndef LBANN_TRANSFORMS_COLOR_JITTER_HPP_INCLUDED
#define LBANN_TRANSFORMS_COLOR_JITTER_HPP_INCLUDED

#include "lbann/transforms/transform.hpp"

namespace lbann {
namespace transform {

/**
 * Randomly change brightness, contrast, and saturation.
 * This randomly adjusts brightness, contrast, and saturation, in a random
 * order.
 */
class color_jitter : public transform {
public:
  /**
   * Randomly adjust brightness, contrast, and saturation within given ranges.
   * Set both min and max to 0 to disable that adjustment.
   * @param min_brightness_factor Minimum brightness adjustment (>= 0).
   * @param max_brightness_factor Maximum brightness adjustment.
   * @param min_contrast_factor Minimum contrast adjustment (>= 0).
   * @param max_contrast_factor Maximum contrast adjustment.
   * @param min_saturation_factor Minimum saturation adjustment (>= 0).
   * @param max_saturation_factor Maximum saturation adjustment.
   */
  color_jitter(float min_brightness_factor, float max_brightness_factor,
               float min_contrast_factor, float max_contrast_factor,
               float min_saturation_factor, float max_saturation_factor);
  
  transform* copy() const override { return new color_jitter(*this); }

  std::string get_type() const override { return "color_jitter"; }

  void apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) override;

private:
  /** Minimum brightness factor. */
  float m_min_brightness_factor;
  /** Maximum brightness factor. */
  float m_max_brightness_factor;
  /** Minimum contrast factor. */
  float m_min_contrast_factor;
  /** Maximum contrast factor. */
  float m_max_contrast_factor;
  /** Minimum saturation factor. */
  float m_min_saturation_factor;
  /** Maximum saturation factor. */
  float m_max_saturation_factor;
};

}  // namespace transform
}  // namespace lbann

#endif  // LBANN_TRANSFORMS_COLOR_JITTER_HPP_INCLUDED
