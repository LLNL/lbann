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

#ifndef LBANN_TRANSFORMS_RANDOM_AFFINE_HPP_INCLUDED
#define LBANN_TRANSFORMS_RANDOM_AFFINE_HPP_INCLUDED

#include "lbann/transforms/transform.hpp"

namespace lbann {
namespace transform {

/** Apply a random affine transform to an image. */
class random_affine : public transform {
public:
  /**
   * Set up the affine transform.
   * Rotate a random number of degrees selected in [rotate_min, rotate_max].
   * Translate the vertical dimension in a random amount in [-h*translate_h,
   * h*translate_h], and the horizontal dimension in [-w*translate_w,
   * w*translate_w].
   * Scale by a random amount in [scale_min, scale_max].
   * Shear by a random number of degrees in [shear_min, shear_max].
   * Set arguments to 0 to disable that transform.
   */
  random_affine(float rotate_min, float rotate_max,
                float translate_h, float translate_w,
                float scale_min, float scale_max,
                float shear_min, float shear_max) :
    transform(),
    m_rotate_min(rotate_min), m_rotate_max(rotate_max),
    m_translate_h(translate_h), m_translate_w(translate_w),
    m_scale_min(scale_min), m_scale_max(scale_max),
    m_shear_min(shear_min), m_shear_max(shear_max) {}

  transform* copy() const override { return new random_affine(*this); }

  std::string get_type() const override { return "random_affine"; }

  void apply(utils::type_erased_matrix& data, std::vector<size_t>& dims) override;
private:
  /** Range in degrees to rotate. */
  float m_rotate_min, m_rotate_max;
  /** Fraction of height/width to translate. */
  float m_translate_h, m_translate_w;
  /** Range for fraction to scale by. */
  float m_scale_min, m_scale_max;
  /** Range for degrees to shear. */
  float m_shear_min, m_shear_max;
};

}  // namespace transform
}  // namespace lbann

#endif  // LBANN_TRANSFORMS_RANDOM_AFFINED_CENTER_CROP_HPP_INCLUDED
