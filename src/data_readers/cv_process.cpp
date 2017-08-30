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


#include "lbann/data_readers/cv_process.hpp"

#ifdef __LIB_OPENCV
namespace lbann {

/**
 * Copy constructor.
 * Rather than transferring the ownership of the managed cv_transform objects
 * pointed by the pointers, or sharing them by simply copying the pointers,
 * copy-constructs the objects and owns the pointers to those newly created
 * objects.
 */
cv_process::cv_process(const cv_process& rhs)
  : m_flip(rhs.m_flip), m_split(rhs.m_split),
    m_normalizer((!!rhs.m_normalizer) ? (rhs.m_normalizer->clone()) : NULL),
    m_augmenter((!!rhs.m_augmenter)   ? (rhs.m_augmenter->clone()) : NULL),
    m_transform1((!!rhs.m_transform1) ? (rhs.m_transform1->clone()) : NULL),
    m_transform2((!!rhs.m_transform2) ? (rhs.m_transform2->clone()) : NULL),
    m_transform3((!!rhs.m_transform3) ? (rhs.m_transform3->clone()) : NULL)
{}

/**
 * Assignment operator.
 * Rather than transferring the ownership of the managed cv_transform objects
 * pointed by the pointers, or sharing them by simply copying the pointers,
 * copy-constructs the objects and owns the pointers to those newly created
 * objects.
 */
cv_process& cv_process::operator=(const cv_process& rhs) {
  if (this == &rhs) {
    return (*this);
  }

  m_flip = rhs.m_flip;
  m_split = rhs.m_split;
  m_normalizer.reset((!!rhs.m_normalizer)? (rhs.m_normalizer->clone()) : NULL);
  m_augmenter.reset((!!rhs.m_augmenter)? (rhs.m_augmenter->clone()) : NULL);
  m_transform1.reset((!!rhs.m_transform1)? (rhs.m_transform1->clone()) : NULL);
  m_transform2.reset((!!rhs.m_transform2) ? (rhs.m_transform2->clone()) : NULL);
  m_transform3.reset((!!rhs.m_transform3) ? (rhs.m_transform3->clone()) : NULL);

  return (*this);
}


void cv_process::reset() {
  if (m_normalizer) {
    m_normalizer->reset();
  }
  if (m_augmenter) {
    m_augmenter->reset();
  }
  if (m_transform1) {
    m_transform1->reset();
  }
  if (m_transform2) {
    m_transform2->reset();
  }
  if (m_transform3) {
    m_transform3->reset();
  }
}


void cv_process::disable_transforms() {
  if (m_normalizer) {
    m_normalizer->disable();
  }
  if (m_augmenter) {
    m_augmenter->disable();
  }
  if (m_transform1) {
    m_transform1->disable();
  }
  if (m_transform2) {
    m_transform2->disable();
  }
  if (m_transform3) {
    m_transform3->disable();
  }
}

/**
 * Preprocess an image.
 * @return true if successful
 */
bool cv_process::preprocess(cv::Mat& image) {
  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  bool ok = true;

  if (m_transform1 && m_transform1->determine_transform(image)) {
    ok = m_transform1->apply(image);
    _LBANN_MILD_EXCEPTION(!ok, "transform1 has failed!", false);
  }

  if (to_flip()) {
    cv::flip(image, image, how_to_flip());
  }

  if (m_augmenter && m_augmenter->determine_transform(image)) {
    ok = m_augmenter->apply(image);
    _LBANN_MILD_EXCEPTION(!ok, "augmentation has failed!", false);
  }

  if (m_transform2 && m_transform2->determine_transform(image)) {
    ok = m_transform2->apply(image);
    _LBANN_MILD_EXCEPTION(!ok, "transform2 has failed!", false);
  }

  // The place for early-normalization in case that there is something to be done
  // after normalization. Otherwise, normalization will implicitly be applied
  // during data copying. in which case it needs to be disabled manually after the
  // copying. If it is explicilty applied here, the normalizer will automatically be
  // disabled to avoid redundant transform during copying.

  if (m_normalizer) {
    m_normalizer->determine_transform(image);
  }

#if 0
  const std::vector<cv_normalizer::channel_trans_t> ntrans = get_transform_normalize();
  for (size_t i=0u; i < ntrans.size(); ++i) {
    std::cout << "preprocess scaling: " << ntrans[i].first << ' ' << ntrans[i].second << std::endl;
  }
#endif

  if (m_transform3) {
    if (m_normalizer) {
      ok = m_normalizer->apply(image);
      _LBANN_MILD_EXCEPTION(!ok, "normalization has failed!", false);
      //m_normalizer->disable();
    }

    std::cout << "custom_transform3 " << std::endl;
    if (m_transform3->determine_transform(image)) {
      ok = m_transform3->apply(image);
      _LBANN_MILD_EXCEPTION(!ok, "transform3 has failed!", false);
    }
  }

  return ok;
}

/**
 * Postprocess an image.
 * @return true if successful
 */
bool cv_process::postprocess(cv::Mat& image) {
  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  bool ok = true;

  if (m_transform3 && m_transform3->is_enabled()) {
    // must have been enabled by calling determine_inverse_transform()
    ok = m_transform3->apply(image);
    _LBANN_MILD_EXCEPTION(!ok, "inverse transform3 has failed!", false);
  }

  // Inverse the early-normalization in case that there was something
  // (transform3) had to be done after normalization during preprocessing.
  // Otherwise, the transform will be done during copying via scaling.

  if (m_transform3) {
    if (m_normalizer && m_normalizer->determine_inverse_transform()) {
      ok = m_normalizer->apply(image);
      _LBANN_MILD_EXCEPTION(!ok, "inverse normalization has failed!", false);
    }
  }

#if 0
  const std::vector<cv_normalizer::channel_trans_t> ntrans = get_transform_normalize();
  for (size_t i=0u; i < ntrans.size(); ++i) {
    std::cout << "postprocess scaling: " << ntrans[i].first << ' ' << ntrans[i].second << std::endl;
  }
#endif

  if (m_transform2 && m_transform2->is_enabled()) {
    ok = m_transform2->apply(image);
    _LBANN_MILD_EXCEPTION(!ok, "inverse transform2 has failed!", false);
  }

  if (to_flip()) {
    cv::flip(image, image, how_to_flip());
  }

  if (m_transform1 && m_transform1->is_enabled()) {
    ok = m_transform1->apply(image);
    _LBANN_MILD_EXCEPTION(!ok, "inverse transform1 has failed!", false);
  }

  return ok;
}

} // end of namespace lbann
#endif // __LIB_OPENCV
