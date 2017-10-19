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
#include "lbann/utils/exception.hpp"

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
    m_is_normalizer_set(rhs.m_is_normalizer_set),
    m_normalizer_idx(rhs.m_normalizer_idx)
{
  for (size_t i = 0u; i < rhs.m_transforms.size(); ++i) {
    std::unique_ptr<cv_transform> p(rhs.m_transforms[i]->clone());
    if (!p) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: cv_process: undefined transform " << i;
      throw lbann_exception(err.str());
    }
    m_transforms.push_back(std::move(p)); // avoid using emplace
  }
}

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
  m_is_normalizer_set = rhs.m_is_normalizer_set;
  m_normalizer_idx = rhs.m_normalizer_idx;

  m_transforms.clear();

  for (size_t i = 0u; i < rhs.m_transforms.size(); ++i) {
    std::unique_ptr<cv_transform> p(rhs.m_transforms[i]->clone());
    if (!p)  {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: cv_process: undefined transform " << i;
      throw lbann_exception(err.str());
    }
    m_transforms.push_back(std::move(p));
  }

  return (*this);
}


void cv_process::reset() {
  for (size_t i = 0u; i < m_transforms.size(); ++i)
    m_transforms[i]->reset();
}

void cv_process::disable_normalizer() {
  if (m_is_normalizer_set) {
    m_transforms[m_normalizer_idx]->disable();
  }
}

void cv_process::disable_transforms() {
  for (size_t i = 0u; i < m_transforms.size(); ++i) {
    m_transforms[i]->disable();
  }
}

bool cv_process::add_transform(std::unique_ptr<cv_transform> tr) {
  if (!tr) return false;
  m_transforms.push_back(std::move(tr));
  return true;
}

bool cv_process::add_normalizer(std::unique_ptr<cv_normalizer> tr) {
  if (!tr || m_is_normalizer_set) return false;
  m_is_normalizer_set = true;
  m_normalizer_idx = m_transforms.size();
  m_transforms.push_back(std::move(tr));
  return true;
}

/// Allow read-only access to a particular transform indexed by idx
const cv_transform* cv_process::get_transform(const unsigned int idx) const {
  if (idx >= m_transforms.size()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: cv_process: invalid index " << idx << " >= " << m_transforms.size();
    throw lbann_exception(err.str());
  }
  return m_transforms[idx].get();
}

/// Allow read-write access to a particular transform indexed by idx
cv_transform* cv_process::get_transform(const unsigned int idx) {
  if (idx >= m_transforms.size()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: cv_process: invalid index " << idx << " >= " << m_transforms.size();
    throw lbann_exception(err.str());
  }
  return m_transforms[idx].get();
}

/**
 * Call this before image saving/exporting in postprocessing if inverse normalization
 * is needed to save image.  Unless normalization is followed by a transform, inverse
 * normalization is done while copying data from El::Matrix<DataType> to cv::Mat format.
 * Otherwise, it will be done during postprocessing as the rest of transforms in order.
 */
void cv_process::determine_inverse_normalization() {
  if (!m_is_normalizer_set || !is_normalizer_last()) {
    return;
  }

  m_transforms[m_normalizer_idx]->determine_inverse_transform();
}

/**
 * Preprocess an image.
 * @return true if successful
 */
bool cv_process::preprocess(cv::Mat& image) {
  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  bool ok = true;

  if (to_flip()) {
    cv::flip(image, image, how_to_flip());
  }

  // While many transforms can update pixel values in place, some require new
  // memory locations to write new values. In addition, at the end of a pre-
  // processing pipeline, the values in an OpenCV matrix is copied into an
  // Elemental matrix. Normalization typically is the last transform in a
  // preprocessing pipeline. It is also simple enough (e.g., applying a linear
  // function to existing values) that we can merge it with copying from one memory
  // to another. Therefore, unless there is another preprocessing operation to be
  // done after normalization, in which case we prefer in-place updating,
  // we implicitly apply it during copying between memory locations to avoid
  // redundant memory access overheads. For this reason, we treat normalization
  // differently from other transforms.

  const bool is_normalizer_the_last = is_normalizer_last();
  const unsigned int n_immediate_transforms 
      = (is_normalizer_the_last? m_normalizer_idx : m_transforms.size());

  for (size_t i = 0u; i < n_immediate_transforms; ++i) {
    if (m_transforms[i]->determine_transform(image)) {
      ok = m_transforms[i]->apply(image);
      _LBANN_MILD_EXCEPTION(!ok, "transform " << i << " has failed!", false);
    }
  }

  if (is_normalizer_the_last) {
    m_transforms[m_normalizer_idx]->determine_transform(image);
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

  const bool is_normalizer_the_last = is_normalizer_last();
  const unsigned int n_immediate_transforms 
      = (is_normalizer_the_last? m_normalizer_idx : m_transforms.size());

  // If normalizer is the last transform in the preprocessing pipeline, it will
  // be the first in the postprocessing. In addition, it has implicitly been
  // inversed during copying from El::Mat to cv::Mat before calling postprocess(image)

  for (size_t i = n_immediate_transforms; i > 0; --i) {
    if (m_transforms[i-1]->determine_inverse_transform()) {
      ok = m_transforms[i-1]->apply(image);
      _LBANN_MILD_EXCEPTION(!ok, "inverse transform " << i-1 << " has failed!", false);
    }
  }

  if (to_flip()) {
    cv::flip(image, image, how_to_flip());
  }

  return ok;
}

std::vector<cv_normalizer::channel_trans_t> cv_process::get_transform_normalize() const {
  return (m_is_normalizer_set?
          dynamic_cast<const cv_normalizer*>(m_transforms[m_normalizer_idx].get())->transform() :
          std::vector<cv_normalizer::channel_trans_t>());
}

std::vector<cv_normalizer::channel_trans_t> cv_process::get_transform_normalize(const unsigned int ch) const {
  std::vector<cv_normalizer::channel_trans_t> trans;
  if (m_is_normalizer_set) {
    trans = dynamic_cast<const cv_normalizer*>(m_transforms[m_normalizer_idx].get())->transform();
  }

  return ((trans.size() > ch) ?
          std::vector<cv_normalizer::channel_trans_t>(1, trans[ch]) :
          std::vector<cv_normalizer::channel_trans_t>(1, cv_normalizer::channel_trans_t(1.0, 0.0)));
}

} // end of namespace lbann
#endif // __LIB_OPENCV
