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
// cv_mean_extractor .cpp .hpp - accumulate mean over the image set
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/cv_mean_extractor.hpp"
#include "lbann/utils/mild_exception.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_OPENCV
namespace lbann {

cv_mean_extractor::cv_mean_extractor()
: cv_transform(), m_batch_size(m_default_batch_size), m_batch_cnt(0u), m_partial_cnt(0u), m_type_code(0)
{}

cv_mean_extractor::cv_mean_extractor(const cv_mean_extractor& rhs)
  : cv_transform(rhs), m_batch_size(rhs.m_batch_size),
    m_batch_cnt(rhs.m_batch_cnt), m_partial_cnt(rhs.m_partial_cnt),
    m_type_code(rhs.m_type_code), m_sum(rhs.m_sum.clone()), m_avg(rhs.m_avg.clone())
{}

cv_mean_extractor& cv_mean_extractor::operator=(const cv_mean_extractor& rhs) {
  cv_transform::operator=(rhs);
  m_batch_size = rhs.m_batch_size;
  m_batch_cnt = rhs.m_batch_cnt;
  m_partial_cnt = rhs.m_partial_cnt;
  m_type_code = rhs.m_type_code;
  m_sum = rhs.m_sum.clone();
  m_avg = rhs.m_avg.clone();
  return *this;
}

cv_mean_extractor *cv_mean_extractor::clone() const {
  return (new cv_mean_extractor(*this));
}

/** Set up the internal matrices used to accumulate image statistics,
 *  and initialize the batch size.
 */
void cv_mean_extractor::set(const unsigned int width, const unsigned int height,
                            const unsigned int n_ch, const unsigned int batch_sz) {
  if (!m_sum.empty() || (width == 0u) || (height == 0u) || (n_ch == 0u) || (batch_sz == 0u)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: cv_mean_extractor: either using an invalid "
        << "parameter or attempting to reconfigure";
    throw lbann_exception(err.str());
  }

  m_batch_size = batch_sz;

  create_matrices(width, height, n_ch);
  reset();
}

/**
 * This can be used to set the batch size only, and defer the creation of
 * matrices for accumulating statistics until the first image is seen.
 */
void cv_mean_extractor::set(const unsigned int batch_sz) {
  if (!m_sum.empty() || (batch_sz == 0u)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: cv_mean_extractor: " <<
          "cannot reset the batch size once started and it must be greater than 0";
    throw lbann_exception(err.str());
  }
  m_batch_size = batch_sz;
}

void cv_mean_extractor::create_matrices(const unsigned int width, const unsigned int height, const unsigned int n_ch) {
  // OpenCV image type code
  m_type_code = cv_image_type<Float_T>::T(n_ch);
  m_sum = cv::Mat(height, width, m_type_code);
  m_avg = cv::Mat(height, width, m_type_code);
}

void cv_mean_extractor::reset() {
  // convert to a single change image before resetting the values as the
  // dimension of Scalar is limited to 4 (4 channels)
  cv::Mat m_sum_1ch = m_sum.reshape(1);
  m_sum_1ch.setTo(static_cast<Float_T>(0));
  cv::Mat m_avg_1ch = m_avg.reshape(1);
  m_avg_1ch.setTo(static_cast<Float_T>(0));

  m_batch_cnt = 0u;
  m_partial_cnt = 0u;
  m_enabled = false;
}

/**
 * If the size or the number of channels of the given image is different
 * from what is expected, fails.
 */
bool cv_mean_extractor::determine_transform(const cv::Mat& image) {
  m_enabled = false;
  _LBANN_SILENT_EXCEPTION(image.empty(), "", false);
  // If it has not been configured (other than batch size), do it here
  if (m_sum.empty()) {
    create_matrices(image.cols, image.rows, image.channels());
    reset();

    m_enabled = true;
  } else {
    m_enabled = check_if_cv_Mat_has_same_shape(image, m_avg);
  }
  return m_enabled;
}

bool cv_mean_extractor::determine_inverse_transform() {
  // inversing is irrelevant
  return (m_enabled = false);
}

bool cv_mean_extractor::apply(cv::Mat& image) {
  m_enabled = false; // turn off as the transform is applied once
  const double f = get_depth_normalizing_factor(image.depth());
  cv::addWeighted(m_sum, 1.0, image, f, 0.0, m_sum, m_type_code);
  if (++m_partial_cnt == m_batch_size) {
    m_partial_cnt = 0u;
    ++m_batch_cnt;
    cv::addWeighted(m_avg, static_cast<double>(m_batch_cnt-1)/m_batch_cnt,
                    m_sum, 1/static_cast<double>(m_batch_cnt*m_batch_size),
                    0.0, m_avg, m_type_code);
    cv::Mat m_sum_1ch = m_sum.reshape(1);
    m_sum_1ch.setTo(static_cast<Float_T>(0));
  }
  return true;
}

std::string cv_mean_extractor::get_description() const {
  std::stringstream os;
  os << get_type() + ":" << std::endl
     << " - batch size " << m_batch_size << std::endl;
  return os.str();
}

std::ostream& cv_mean_extractor::print(std::ostream& os) const {
  os << get_description()
     << " - partial cnt " << m_partial_cnt << std::endl
     << " - batch cnt " << m_batch_cnt << std::endl;
  return os;
}

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV
