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
// lbann_cv_augmenter .cpp .hpp - Augmenting functions for images
//                                in opencv format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_AUGMENTER_HPP
#define LBANN_CV_AUGMENTER_HPP

#include "patchworks/patchworks_opencv.hpp"
#include "cv_transform.hpp"
#include "lbann/utils/random.hpp"
#include <iostream>
#include <ostream>
#include <cstring>
#include <string>

#ifdef __LIB_OPENCV
namespace lbann {

/**
 * Supports the following transforms:
 * - Random horizontal and vertical flips
 * - Random rotations
 * - Random horizontal and vertical shifts
 * - Random shearing
 */
class cv_augmenter : public cv_transform {
 protected:
  /** Whether to do horizontal flips. */
  bool m_do_horizontal_flip;
  /** Whether to do vertical flips. */
  bool m_do_vertical_flip;

  /** Range in degrees for rotations (0-180). */
  float m_rotation_range;
  /** Range (fraction of total width) for horizontal shifts. */
  float m_horizontal_shift_range;
  /** Range (fraction of total height) for vertical shifts. */
  float m_vertical_shift_range;
  /** Shear angle (radians). */
  float m_shear_range;


  /// Flip decision made
  cv_flipping m_flip;
  /// The rest of the affine tranformations determined
  cv::Mat_<float> m_trans;

  /// Check if there is a reason to enable. (i.e., any option set)
  virtual bool check_to_enable() const;

 public:
  cv_augmenter();
  cv_augmenter(const cv_augmenter& rhs);
  cv_augmenter& operator=(const cv_augmenter& rhs);
  virtual cv_augmenter *clone() const;

  virtual ~cv_augmenter() {}

  /// Set the parameters all at once
  virtual void set(const bool hflip, const bool vflip, const float rot,
                   const float hshift, const float vshift, const float shear);

  /// Reset all the parameters to the default values
  virtual void reset();

  /**
   * Construct an affine transformation matrix based on the options and random
   * numbers. If successful, the tranform is enabled. If not, it is disabled.
   * @return false if not enabled or unsuccessful.
   */
  virtual bool determine_transform(const cv::Mat& image);

  /**
   * Apply the transformation determined.
   * As this method is executed, the transform becomes deactivated.
   * @return false if not successful.
   */
  virtual bool apply(cv::Mat& image);

  /// The augmentation is nonreversible.
  bool determine_inverse_transform() {
    return false;
  }

  virtual std::ostream& print(std::ostream& os) const;
};

} // end of namespace lbann
#endif // __LIB_OPENCV

#endif // LBANN_CV_AUGMENTER_HPP
