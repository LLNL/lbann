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
// lbann_cv_transform .cpp .hpp - base class for the transformation
//                                on image data in opencv format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_TRANSFORM_HPP
#define LBANN_CV_TRANSFORM_HPP

#include "lbann_opencv.hpp"

#ifdef __LIB_OPENCV
namespace lbann {

class cv_transform {
 protected:
  bool m_enabled;

  virtual bool check_to_enable(void) const {
    return true;
  }

 public:
  enum cv_flipping {_both_axes_=-1, _vertical_=0, _horizontal_=1, _no_flip_=2};

  static const float pi;


  cv_transform(void) : m_enabled(false) {}
  cv_transform(const cv_transform& rhs);
  cv_transform& operator=(const cv_transform& rhs);
  virtual cv_transform *clone(void) const;

  virtual ~cv_transform(void) {}

  virtual bool determine_transform(const cv::Mat& image) {
    m_enabled = check_to_enable();
    return m_enabled;
  }

  virtual bool determine_inverse_transform(void) {
    m_enabled = false;
    return false;
  }

  virtual bool apply(cv::Mat& image) = 0;

  //virtual void set(void) {}
  virtual void enable(void) {
    m_enabled = true;
  }
  virtual void disable(void) {
    m_enabled = false;
  }
  virtual void reset(void) {
    m_enabled = false;
  }
  virtual bool is_enabled(void) const {
    return m_enabled;
  }

  virtual std::ostream& print(std::ostream& os) const {
    return os;
  }
};

inline cv_transform::cv_transform(const cv_transform& rhs)
  : m_enabled(rhs.m_enabled) {}

inline cv_transform& cv_transform::operator=(const cv_transform& rhs) {
  m_enabled = rhs.m_enabled;
  return *this;
}

inline bool cv_transform::apply(cv::Mat& image) {
  m_enabled = false;
  return true;
}

inline cv_transform *cv_transform::clone(void) const {
  return static_cast<cv_transform *>(NULL);
}

inline std::ostream& operator<<(std::ostream& os, const cv_transform& tr) {
  tr.print(os);
  return os;
}

} // end of namespace lbann
#endif // __LIB_OPENCV

#endif // LBANN_CV_TRANSFORM_HPP
