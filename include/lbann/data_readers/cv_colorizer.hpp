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
// cv_colorizer .cpp .hpp - transform a non-color (grayscale) image into a
//                          3-channel color image
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_COLORIZER_HPP
#define LBANN_CV_COLORIZER_HPP

#include "cv_transform.hpp"

#ifdef LBANN_HAS_OPENCV
namespace lbann {

class cv_colorizer : public cv_transform {
 protected:
  // --- state variables ---
  bool m_gray; ///< whether an image is monochrome or not

 public:
  cv_colorizer() : cv_transform(), m_gray(false) {}
  cv_colorizer(const cv_colorizer& rhs);
  cv_colorizer& operator=(const cv_colorizer& rhs);
  cv_colorizer *clone() const override;

  ~cv_colorizer() override {}

  void set() { reset(); }
  void reset() override {
    m_enabled = false;
    m_gray = false;
  }

  /**
   * If a given image is in grayscale, the tranform is enabled, and not otherwise.
   * @return false if not enabled or unsuccessful.
   */
  bool determine_transform(const cv::Mat& image) override;

  /// convert back to color image if it used to be a grayscale image
  bool determine_inverse_transform() override;

  /**
   * Apply color conversion if enabled.
   * As it is applied, the transform becomes deactivated.
   * @return false if not successful.
   */
  bool apply(cv::Mat& image) override;

  std::string get_type() const override { return "colorizer"; }
  std::string get_description() const override;
  std::ostream& print(std::ostream& os) const override;
};

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV

#endif // LBANN_CV_COLORIZER_HPP
