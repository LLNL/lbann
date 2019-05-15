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
// cv_resizer .cpp .hpp - Functions to resize images
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_RESIZER_HPP
#define LBANN_CV_RESIZER_HPP

#include "lbann/data_readers/cv_transform.hpp"
#include <utility>
#include <ostream>

#ifdef LBANN_HAS_OPENCV
namespace lbann {

/**
 * Simple image resizing without maintaining the aspect ratio.
 */
class cv_resizer : public cv_transform {
 protected:
  // --- configuration variables ---
  unsigned int m_width; ///< desired width of an image
  unsigned int m_height; ///< desired height of an image

  // --- state variables ---
  /** Three modes of pixel interpolation: INTER_LINEAR, INTER_AREA, and INTER_LINEAR
   *  The first choice is the default when not adaptive. The other two are used when
   *  interpolatng  adaptively. The second is when shrinking, and the third is when enlarging
   */
  static const int m_interpolation_choices[3];
  int m_interpolation; ///< id of the channel value interpolation method used
  bool m_adaptive_interpolation; ///< whether to use adaptive interpolation

 public:
  cv_resizer();
  cv_resizer(const cv_resizer& rhs) = default;
  cv_resizer& operator=(const cv_resizer& rhs) = default;
  cv_resizer *clone() const override;
  ~cv_resizer() override {}

  /**
   * Set the parameters all at once
   * @param width  desired width
   * @param height desired height
   * @param adaptive_interpolation whether to apply a different interpolation method depending on how an image is resized
   */
  void set(const unsigned int width, const unsigned int height,
           const bool adaptive_interpolation = false);

  unsigned int get_width() const { return m_width; }
  unsigned int get_height() const { return m_height; }

  /// Clear the states of the previous transform applied
  void reset() override;

  /**
   * Determine whether to enable transformation.
   * @return false if not enabled.
   */
  bool determine_transform(const cv::Mat& image) override;

  /// Determine whether to enable inverse transformation.
  bool determine_inverse_transform() override { return false; }

  /**
   * Apply the transformation.
   * As this method is executed, the transform becomes deactivated.
   * @return false if not successful.
   */
  bool apply(cv::Mat& image) override;

  std::string get_type() const override { return "resizer"; }
  std::string get_description() const override;
  std::ostream& print(std::ostream& os) const override;
};

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV

#endif // LBANN_CV_RESIZER_HPP
