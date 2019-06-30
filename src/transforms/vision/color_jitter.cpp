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

#include <algorithm>
#include "lbann/transforms/vision/color_jitter.hpp"
#include "lbann/transforms/vision/adjust_brightness.hpp"
#include "lbann/transforms/vision/adjust_contrast.hpp"
#include "lbann/transforms/vision/adjust_saturation.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/opencv.hpp"

namespace lbann {
namespace transform {

color_jitter::color_jitter(float min_brightness_factor, float max_brightness_factor,
                           float min_contrast_factor, float max_contrast_factor,
                           float min_saturation_factor, float max_saturation_factor) :
  transform(),
  m_min_brightness_factor(min_brightness_factor),
  m_max_brightness_factor(max_brightness_factor),
  m_min_contrast_factor(min_contrast_factor),
  m_max_contrast_factor(max_contrast_factor),
  m_min_saturation_factor(min_saturation_factor),
  m_max_saturation_factor(max_saturation_factor) {
  if (min_brightness_factor < 0.0f ||
      max_brightness_factor < min_brightness_factor) {
    LBANN_ERROR("Min/max brightness factors out of range: "
                + std::to_string(min_brightness_factor) + " "
                + std::to_string(max_brightness_factor));
  }
  if (min_contrast_factor < 0.0f ||
      max_contrast_factor < min_contrast_factor) {
    LBANN_ERROR("Min/max contrast factors out of range: "
                + std::to_string(min_contrast_factor) + " "
                + std::to_string(max_contrast_factor));
  }
  if (min_saturation_factor < 0.0f ||
      max_saturation_factor < min_saturation_factor) {
    LBANN_ERROR("Min/max saturation factors out of range: "
                + std::to_string(min_saturation_factor) + " "
                + std::to_string(max_saturation_factor));
  }
}

void color_jitter::apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) {
  fast_rng_gen& gen = get_fast_generator();
  // Determine the order to apply transforms.
  // Unused transforms will be skipped.
  // 1 == brightness, 2 == contrast, 3 == saturation.
  std::vector<int> transform_order = {1, 2, 3};
  std::shuffle(transform_order.begin(), transform_order.end(), gen);
  // Now apply the random adjustments.
  for (const auto& t : transform_order) {
    switch (t) {
    case 1:
      // Brightness.
      if (!(m_min_brightness_factor == 0.0f &&
            m_min_brightness_factor == m_max_brightness_factor)) {
        std::uniform_real_distribution<float> dist(
          m_min_brightness_factor, m_max_brightness_factor);
        adjust_brightness trans = adjust_brightness(dist(gen));
        trans.apply(data, dims);
      }
      break;
    case 2:
      // Contrast.
      if (!(m_min_contrast_factor == 0.0f &&
            m_min_contrast_factor == m_max_contrast_factor)) {
        std::uniform_real_distribution<float> dist(
          m_min_contrast_factor, m_max_contrast_factor);
        adjust_contrast trans = adjust_contrast(dist(gen));
        trans.apply(data, dims);
      }
      break;
    case 3:
      // Saturation.
      if (!(m_min_saturation_factor == 0.0f &&
            m_min_saturation_factor == m_max_saturation_factor)) {
        std::uniform_real_distribution<float> dist(
          m_min_saturation_factor, m_max_saturation_factor);
        adjust_saturation trans = adjust_saturation(dist(gen));
        trans.apply(data, dims);
      }
      break;
    default:
      LBANN_ERROR("Unexpected transform number");
    }
  }
}

}  // namespace transform
}  // namespace lbann
