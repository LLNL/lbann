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
// lbann_cv_colorizer .cpp .hpp - transform a non-color (grayscale) image into a
//                               3-channel color image
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_COLORIZE_HPP
#define LBANN_CV_COLORIZE_HPP

#include "cv_transform.hpp"

#ifdef __LIB_OPENCV
namespace lbann {

class cv_colorizer : public cv_transform {
 protected:
  bool m_gray;

 public:
  cv_colorizer() : cv_transform(), m_gray(false) {}
  cv_colorizer(const cv_colorizer& rhs);
  cv_colorizer& operator=(const cv_colorizer& rhs);
  virtual cv_colorizer *clone() const;

  virtual ~cv_colorizer() {}

  /**
   * If a given image is in grayscale, the tranform is enabled, and not otherwise.
   * @return false if not enabled or unsuccessful.
   */
  virtual bool determine_transform(const cv::Mat& image);

  /// convert back to color image if it used to be a grayscale image
  virtual bool determine_inverse_transform();

  /**
   * Apply color conversion if enabled.
   * As it is applied, the transform becomes deactivated.
   * @return false if not successful.
   */
  virtual bool apply(cv::Mat& image);

  virtual void enable() {
    m_enabled = true;
  }
  virtual void disable() {
    m_enabled = false;
  }
  virtual void reset() {
    m_enabled = false;
    m_gray = false;
  }
  virtual bool is_enabled() const {
    return m_enabled;
  }

  virtual std::ostream& print(std::ostream& os) const;
};

} // end of namespace lbann
#endif // __LIB_OPENCV

#endif // LBANN_CV_COLORIZE_HPP
