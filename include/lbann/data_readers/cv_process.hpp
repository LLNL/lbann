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
// lbann_cv_process .cpp .hpp - structure that defines the operations
//                              on image data in opencv format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_PROCESS_HPP
#define LBANN_CV_PROCESS_HPP

#include "patchworks/patchworks_opencv.hpp"
#include "cv_normalizer.hpp"
#include "cv_augmenter.hpp"
#include "cv_colorizer.hpp"
#include <memory>

#ifdef __LIB_OPENCV
namespace lbann {

/** A structure packs the parameters for image pre-/post-processing that takes
 *  advantage of the OpenCV framework.
 */
class cv_process {
  /// OpenCV flip codes: c<0 for top_left <-> bottom_right, c=0 for top<->down, and c>0 for left<->right

 protected:
  /// Whether to flip an image
  cv_transform::cv_flipping m_flip;
  /// Whether to split channels
  bool m_split;

  // preprocessors: normalizer and augmenter
  std::unique_ptr<cv_normalizer> m_normalizer;
  std::unique_ptr<cv_augmenter> m_augmenter;

  /// custom transformation place holder (before augmentation)
  std::unique_ptr<cv_transform> m_transform1;

  /// custom transformation place holder (after augmentation and before normalization)
  std::unique_ptr<cv_transform> m_transform2;

  /// custom transformation place holder (after normalization)
  std::unique_ptr<cv_transform> m_transform3;


 public:
  cv_process()
    : m_flip(cv_transform::_no_flip_), m_split(true) {}

  cv_process(const cv_process& rhs);
  cv_process& operator=(const cv_process& rhs);

  cv_process(const cv_transform::cv_flipping flip_code, const bool tosplit)
    : m_flip(flip_code), m_split(tosplit) {}

  virtual ~cv_process() {}

  void reset();

  /// Check whether to flip
  bool to_flip() const {
    return (m_flip != cv_transform::_no_flip_);
  }
  /// Tell how to flip
  int how_to_flip() const {
    return static_cast<int>(m_flip);
  }
  /**
   *  Set the flipping behavior. This is to deal with custom image format, and not
   *  to substitute for random flipping in augmentation
   */
  void set_to_flip(cv_transform::cv_flipping f) {
    m_flip = f;
  }
  /// Set to split channels
  bool to_split() const {
    return m_split;
  }


  std::vector<cv_normalizer::channel_trans_t> get_transform_normalize() const;
  std::vector<cv_normalizer::channel_trans_t> get_transform_normalize(const unsigned int ch) const;

  void disable_normalizer() {
    if (m_normalizer) {
      m_normalizer->disable();
    }
  }
  void determine_inverse_normalization();
  /**
   * Call this after preprocessing and image loading to deactivate all the transforms.
   * Then, selectively enable those which require inverse transforms by calling
   * determine_inverse_transform()
   */
  void disable_transforms();

  /// Set the normalization processor
  void set_normalizer(std::unique_ptr<cv_normalizer> np) {
    m_normalizer = std::move(np);
  }
  /// Set the augmentation processor
  void set_augmenter(std::unique_ptr<cv_augmenter> ap) {
    m_augmenter = std::move(ap);
  }
  /// Set the custom transform 1 (comes before the augmentation)
  void set_custom_transform1(std::unique_ptr<cv_transform> tr1) {
    m_transform1 = std::move(tr1);
  }
  /// Set the custom transform 2 (comes after the augmentation and before the normalization)
  void set_custom_transform2(std::unique_ptr<cv_transform> tr2) {
    m_transform2 = std::move(tr2);
  }
  /// Set the custom transform 3 (comes after the normalization)
  void set_custom_transform3(std::unique_ptr<cv_transform> tr3) {
    m_transform3 = std::move(tr3);
  }

  /// Check if the normalizer has been set
  bool is_set_normalizer() const {
    return !!m_normalizer;
  }
  /// Check if the augmenter has been set
  bool is_set_augmenter() const {
    return !!m_augmenter;
  }
  /// Check if the custom transform 1 has been set
  bool is_set_custom_transform1() const {
    return !!m_transform1;
  }
  /// Check if the custom transform 2 has been set
  bool is_set_custom_transform2() const {
    return !!m_transform2;
  }
  /// Check if the custom transform 3 has been set
  bool is_set_custom_transform3() const {
    return !!m_transform3;
  }

  /// Allow read-only access to the normalization processor
  const cv_normalizer *normalizer() const {
    return m_normalizer.get();
  }
  /// Allow read-only access to the augmentation processor
  const cv_augmenter *augmenter() const {
    return m_augmenter.get();
  }
  /// Allow read-only access to the first custom transform
  const cv_transform *custom_transform1() const {
    return m_transform1.get();
  }
  /// Allow read-only access to the second custom transform
  const cv_transform *custom_transform2() const {
    return m_transform2.get();
  }
  /// Allow read-only access to the third custom transform
  const cv_transform *custom_transform3() const {
    return m_transform3.get();
  }

  /// Allow read-write access to the normalization processor
  cv_normalizer *normalizer() {
    return m_normalizer.get();
  }
  /// Allow read-write access to the augmentation processor
  cv_augmenter *augmenter() {
    return m_augmenter.get();
  }
  /// Allow read-write access to the first custom transform
  cv_transform *custom_transform1() {
    return m_transform1.get();
  }
  /// Allow read-write access to the second custom transform
  cv_transform *custom_transform2() {
    return m_transform2.get();
  }
  /// Allow read-write access to the third custom transform
  cv_transform *custom_transform3() {
    return m_transform3.get();
  }

  bool preprocess(cv::Mat& image);
  bool postprocess(cv::Mat& image);
};

/**
 * Call this after preprocessing and image loading but before image saving and
 * postprocessing if inverse normalization is needed (e.g. to save image).
 * Unless transform3 exists, normalization is done while copying data from
 * El::Matrix<DataType> to cv::Mat format. Otherwise, it will be done during
 * postprocessing after potentially inversing transform3.
 */
inline void cv_process::determine_inverse_normalization() {
  if (!m_normalizer) {
    return;
  }

  if (m_transform3) {
    m_normalizer->disable();
  } else {
    m_normalizer->determine_inverse_transform();
  }
}

inline std::vector<cv_normalizer::channel_trans_t> cv_process::get_transform_normalize() const {
  return (m_normalizer? m_normalizer->transform() :
          std::vector<cv_normalizer::channel_trans_t>());
}

inline std::vector<cv_normalizer::channel_trans_t> cv_process::get_transform_normalize(const unsigned int ch) const {
  std::vector<cv_normalizer::channel_trans_t> trans;
  if (m_normalizer) {
    trans = m_normalizer->transform();
  }

  return ((trans.size() > ch) ?
          std::vector<cv_normalizer::channel_trans_t>(1, trans[ch]) :
          std::vector<cv_normalizer::channel_trans_t>(1, cv_normalizer::channel_trans_t(1.0, 0.0)));
}

} // end of namespace lbann
#endif // __LIB_OPENCV

#endif // LBANN_CV_PROCESS_HPP
