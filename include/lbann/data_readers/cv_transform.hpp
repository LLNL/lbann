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
// cv_transform .cpp .hpp - base class for the transformation
//                          on image data in opencv format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_TRANSFORM_HPP
#define LBANN_CV_TRANSFORM_HPP

#include "opencv.hpp"

#ifdef __LIB_OPENCV
namespace lbann {

class cv_transform {
 protected:
  /// per-image indicator of whether to apply or not
  bool m_enabled;

  // Allow to manually shut transform off without destroying it
  //bool m_manual_switch;

  // transform prepared given the configuration (and the image)
  // m_trans;

  /** Check if transform is configured to apply.
   * (e.g., if any of the augmentaion methods is enabled)
   */
  virtual bool check_to_enable() const {
    return true;
  }

 public:
  enum cv_flipping {_both_axes_=-1, _vertical_=0, _horizontal_=1, _no_flip_=2};

  static const float pi;


  cv_transform();
  cv_transform(const cv_transform& rhs);
  cv_transform& operator=(const cv_transform& rhs);
  virtual cv_transform *clone() const;

  virtual ~cv_transform() {}

  virtual bool determine_transform(const cv::Mat& image);
  virtual bool determine_inverse_transform();
  virtual bool apply(cv::Mat& image) = 0;

  // define a method to configure the transform
  //virtual void set(args) {}

  /// Turn transform on
  virtual void enable() {
    m_enabled = true;
  }
  /// Turn transform off
  virtual void disable() {
    m_enabled = false;
  }
  /// Reset the transform state
  virtual void reset() {
    m_enabled = false;
    // e.g., m_trans.clear();
  }
  /// Check if transform is on
  virtual bool is_enabled() const {
    return m_enabled;
  }

  //bool toggle_manual_switch() { return (m_manual_switch = !m_manual_switch); }

  virtual std::ostream& print(std::ostream& os) const {
    return os;
  }
};

/// Default constructor
inline cv_transform::cv_transform()
  : m_enabled(false)//, m_manual_switch(false)
{}

/// Deep-copying constructor
inline cv_transform::cv_transform(const cv_transform& rhs)
  : m_enabled(rhs.m_enabled) {}

/// Assignement operator. deep-copy everything
inline cv_transform& cv_transform::operator=(const cv_transform& rhs) {
  m_enabled = rhs.m_enabled;
  return *this;
}

/** Prepare transform for the given image as configured.
 *  Then, check if they are valid, and turn the transform on if so.
 *  The preparation includes as much precomputation as possible. For example,
 *  if the transformation consists of constructing four affine transform matrices
 *  and applying them to the given image in sequence, the transform matrices
 *  will be reduced to one. Then, the following function apply(image) will 
 *  finally apply it to the image.
 */
inline bool cv_transform::determine_transform(const cv::Mat& image) {
  // clear any transform state computed for previous image
  // reset()
  m_enabled = check_to_enable();
  // if (!m_enabled) return false;
  // compute m_trans for the image and the configuration of the transform
  // Here, some transform may not applicable to the given image.
  // In that case, set m_enabled = false (or fruther throw an exception).
  return m_enabled;
}

/** Prepare the inverse transform to undo preprocessing transforms if needed
 *  for postprocessing. Not all transforms can be or need to be inversed.
 *  Then, check if they are valid, and turn the transform on if so.
 *  By default, turn this off as we do not need to undo in most of the cases.
 *  In need of manual overriding to enable/disable inverse transform, implement
 *  such a logic in this fuction and interfaces to enable/disable.
 */
inline bool cv_transform::determine_inverse_transform() {
  // In case of manual overriding, if (!m_manual_switch) return false;
  // If this transform, by design, can not be or does not need to be inversed,
  //   return (m_enabled = false);
  // 
  // If the transform has not been applied (e.g., m_trans has not been set),
  //   return (m_enabled = false);
  // Note that this cannot be determined by m_enabled as the transform is turned
  // off once applied.
  //
  // Compute the inverse of m_trans and overwrite m_trans;
  // set m_enabled to true;
  // return true;
  return false;
}

/** Apply transform once and turn it off
 *  To conditionally apply the transform given an image,
 *  determine_transform(image) or determine_inverse_transform() must be called
 *  in advance. These will do as much precomputation as possible. For example,
 *  if the transformation consists of constructing four affine transform matrices
 *  and multiplying them to the given image in sequence, the transform matrices
 *  will be reduced to one. Then, this function will finally apply it to the image.
 *  There are three possible ways to implement condition checking as shown below,
 *  but here the third option is preferred for minimizing the number of calls
 *  1. checking m_enabled internally
 *  2. externally call is_enabled()
 *  3. rely on the return value of determine_transform()/determine_inverse_transform()
 */
inline bool cv_transform::apply(cv::Mat& image) {
  // As the transform is applied once, turn this off
  m_enabled = false;
  // Return the success of transform
  return true;
}

/// Return the pointer of a newly copy-constructed object
inline cv_transform *cv_transform::clone() const {
  return static_cast<cv_transform *>(NULL);
}

std::ostream& operator<<(std::ostream& os, const cv_transform& tr);

} // end of namespace lbann
#endif // __LIB_OPENCV

#endif // LBANN_CV_TRANSFORM_HPP
