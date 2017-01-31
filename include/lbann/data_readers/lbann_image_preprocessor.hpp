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
//
// lbann_image_preprocessor.hpp - Preprocessing utilities for image inputs
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_IMAGE_PREPROCESSOR
#define LBANN_IMAGE_PREPROCESSOR

#include "lbann/lbann_base.hpp"

namespace lbann {

/**
 * Support class for preprocessing image inputs.
 * Supports the following transforms:
 * - Random horizontal and vertical flips
 * - Random rotations
 * - Random horizontal and vertical shifts
 * - Random shearing
 * - Standardize to 0 mean
 * - Standardize to unit variance
 * - Scale to the range [0, 1]
 * - Normalize via z-score
 */
class lbann_image_preprocessor {
public:
  lbann_image_preprocessor();
  virtual ~lbann_image_preprocessor() {}

  void horizontal_flip(bool b) { m_horizontal_flip = b; }
  void vertical_flip(bool b) { m_vertical_flip = b; }
  void rotation(float range) { m_rotation_range = range; }
  void horizontal_shift(float range) { m_horizontal_shift = range; }
  void vertical_shift(float range) { m_vertical_shift = range; }
  void shear_range(float range) { m_shear_range = range; }
  void subtract_mean(bool b) { m_mean_subtraction = b; }
  void unit_variance(bool b) { m_unit_variance = b; }
  void scale(bool b) { m_scale = b; }
  void z_score(bool b) { m_z_score = b; }

  /**
   * Preprocess pixels according to the currently-set transforms.
   * @param pixels The pixels to process as a column vector (num x 1 mat).
   * @param num_channels The number of channels pixels has.
   */
  void preprocess(Mat& pixels, unsigned num_channels);

protected:
  /** Whether to do horizontal flips. */
  bool m_horizontal_flip;
  /** Whether to do vertical flips. */
  bool m_vertical_flip;
  /** Range in degrees for rotations (0-180). */
  float m_rotation_range;
  /** Range (fraction of total width) for horizontal shifts. */
  float m_horizontal_shift;
  /** Range (fraction of total height) for vertical shifts. */
  float m_vertical_shift;
  /** Shear angle (radians). */
  float m_shear_range;
  /** Whether to normalize to 0 mean. */
  bool m_mean_subtraction;
  /** Whether to normalize to unit variance. */
  bool m_unit_variance;
  /** Whether to scale to [0, 1]. */
  bool m_scale;
  /** Whether to normalize via z-score. */
  bool m_z_score;

  void mean_subtraction(Mat& pixels, unsigned num_channels);
  void unit_variance(Mat& pixels, unsigned num_channels);
  void unit_scale(Mat& pixels, unsigned num_channels);
  void z_score(Mat& pixels, unsigned num_channels);
};

}  // namespace lbann

#endif  // LBANN_IMAGE_PREPROCESSOR
