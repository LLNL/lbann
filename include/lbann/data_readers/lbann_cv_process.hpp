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
// lbann_cv_process .cpp .hpp - Image prerpocessing functions
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_PROCESS_HPP
#define LBANN_CV_PROCESS_HPP

#include "lbann/data_readers/patchworks/patchworks_opencv.hpp"
#include "lbann/data_readers/lbann_cv_preprocessor.hpp"

#ifdef __LIB_OPENCV
namespace lbann
{

class cv_custom_transform
{ // TODO: separate out to a new file
 protected:
  bool m_is_set;

 public:
  cv_custom_transform(void) : m_is_set(false) {}
  virtual ~cv_custom_transform(void) {}
  virtual bool apply(cv::Mat& image) const {
    if (!m_is_set) return false;
    return true;
  }
  virtual void set(void) { m_is_set = true; }
  virtual bool is_set(void) const { return m_is_set; }
};

/** A structure packs the parameters for image pre-/post-processing that takes
 *  advantage of the OpenCV framework.
 */
class cv_process {
 public:
  /// OpenCV flip codes: c<0 for top_left <-> bottom_right, c=0 for top<->down, and c>0 for left<->right
  enum flipping {_both_axes_=-1, _vertical_=0, _horizontal_=1, _none_=2};

 protected:
  /// Whether to flip an image
  flipping m_flip;
  /// Whether to split channels
  bool m_split;

  // These will be used to compute newvalue[ch] = cv::saturate_cast<T>(alpha[ch]*value[ch] + beta[ch])
  std::vector<double> m_alpha; ///< scale parameter for linear transform
  std::vector<double> m_beta;  ///< shift parameter for linear transform

  /// preprocessor
  cv_preprocessor m_preprocessor;
  
  /// custom transformation place holder (before augmentation)
  cv_custom_transform m_transform1;

  /// custom transformation place holder (after augmentation and before normalization)
  cv_custom_transform m_transform2;

  /// custom transformation place holder (after normalization)
  cv_custom_transform m_transform3;

 public:


 public:
  cv_process(void)
  : m_flip(_none_), m_split(true) {}

  cv_process(const flipping flip_code, const bool tosplit)
  : m_flip(flip_code), m_split(tosplit) {}

  /// Check whether to flip
  bool to_flip(void) const { return (m_flip != _none_); }
  /// Tell how to flip
  int how_to_flip(void) const { return static_cast<int>(m_flip); }
  /// Set the flipping behavior
  void set_to_flip(flipping f) { m_flip = f; }
  /// Set to split channels
  bool to_split(void) const { return m_split; }

  const std::vector<double>& alpha(void) const { return m_alpha; }
  const std::vector<double>& beta(void)  const { return m_beta; }
  bool set_normalization_params(const std::vector<double>& a, const std::vector<double>& b);
  void init_normalization_params(void) { m_alpha.clear(); m_beta.clear(); }

  void set_preprocessor(const cv_preprocessor& pp) { m_preprocessor = pp; }
  const cv_preprocessor& preprocessor(void) const { return m_preprocessor; }
  cv_preprocessor& preprocessor(void) { return m_preprocessor; }

  bool compute_normalization_params(const cv::Mat& image);

  bool compute_normalization_params(const cv::Mat& image,
      std::vector<double>& alpha, std::vector<double>& beta) const;

  bool augment(cv::Mat& image) { return m_preprocessor.augment(image); }

  bool normalize(cv::Mat& image) const
  { return m_preprocessor.normalize(image, m_alpha, m_beta); }

  void set_custom_transform1(const cv_custom_transform& tr1) { m_transform1 = tr1; }
  void set_custom_transform2(const cv_custom_transform& tr2) { m_transform2 = tr2; }
  void set_custom_transform3(const cv_custom_transform& tr3) { m_transform3 = tr3; }

  cv_custom_transform& custom_transform1(void) { return m_transform1; }
  const cv_custom_transform& custom_transform1(void) const { return m_transform1; }

  cv_custom_transform& custom_transform2(void) { return m_transform2; }
  const cv_custom_transform& custom_transform2(void) const { return m_transform2; }

  cv_custom_transform& custom_transform3(void) { return m_transform3; }
  const cv_custom_transform& custom_transform3(void) const { return m_transform3; }
};

} // end of namespace lbann
#endif // __LIB_OPENCV

#endif // LBANN_CV_PROCESS_HPP
