////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
// cv_augmenter .cpp .hpp - Augmenting functions for images in opencv format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_AUGMENTER_HPP
#define LBANN_CV_AUGMENTER_HPP

#include "cv_transform.hpp"
#include <iostream>
#include <ostream>
#include <cstring>
#include <string>

#ifdef LBANN_HAS_OPENCV
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
  // --- configuration variables ---
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

  // --- state variables ---
  /// Flip decision made
  cv_flipping m_flip; // currently more of a configuration variable but can easily become a state variable
  /// The rest of the affine tranformations determined
  cv::Mat_<float> m_trans;

  /// Check if there is a reason to enable. (i.e., any option set)
  bool check_to_enable() const override;

 public:
  cv_augmenter();
  cv_augmenter(const cv_augmenter& rhs);
  cv_augmenter& operator=(const cv_augmenter& rhs);
  cv_augmenter* clone() const override;

  ~cv_augmenter() override {}

  /// Set the parameters all at once
  void set(const bool hflip, const bool vflip, const float rot,
           const float hshift, const float vshift, const float shear);

  /// Clear the states of the previous transform applied
  void reset() override;

  /**
   * Construct an affine transformation matrix based on the options and random
   * numbers. If successful, the tranform is enabled. If not, it is disabled.
   * @return false if not enabled or unsuccessful.
   */
  bool determine_transform(const cv::Mat& image) override;

  /// Augmentation is irreversible. Thus, this has no effect.
  bool determine_inverse_transform() override { return false; }

  /**
   * Apply the transformation determined.
   * As this method is executed, the transform becomes deactivated.
   * @return false if not successful.
   */
  bool apply(cv::Mat& image) override;

  std::string get_type() const override { return "augmenter"; }
  std::string get_description() const override;
  std::ostream& print(std::ostream& os) const override;
};

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV

#endif // LBANN_CV_AUGMENTER_HPP
