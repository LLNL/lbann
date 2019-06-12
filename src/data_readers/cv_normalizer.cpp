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
// lbann_cv_normalizer .cpp .hpp - Normalizing functions for images
//                                 in opencv format
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/cv_normalizer.hpp"
#include "lbann/utils/mild_exception.hpp"
#include <cmath> //fabs

#ifdef LBANN_HAS_OPENCV
namespace lbann {

cv_normalizer::cv_normalizer()
  : cv_transform(), m_mean_subtraction(false), m_unit_variance(false),
    m_unit_scale(true), m_z_score(false)
{}


cv_normalizer::cv_normalizer(const cv_normalizer& rhs)
  : cv_transform(rhs), m_mean_subtraction(rhs.m_mean_subtraction), m_unit_variance(rhs.m_unit_variance),
    m_unit_scale(rhs.m_unit_scale), m_z_score(rhs.m_z_score), m_trans(rhs.m_trans) {
}


cv_normalizer& cv_normalizer::operator=(const cv_normalizer& rhs) {
  if (this == &rhs) {
    return (*this);
  }
  cv_transform::operator=(rhs);
  m_mean_subtraction = rhs.m_mean_subtraction;
  m_unit_variance = rhs.m_unit_variance;
  m_unit_scale = rhs.m_unit_scale;
  m_z_score = rhs.m_z_score;
  m_trans = rhs.m_trans;

  return (*this);
}


cv_normalizer *cv_normalizer::clone() const {
  return new cv_normalizer(*this);
}


cv_normalizer::normalization_type& cv_normalizer::set_normalization_type(
  normalization_type& ntype, const normalization_type flag) const {
  return (ntype = set_normalization_bits(ntype, flag));
}


bool cv_normalizer::check_to_enable() const {
  return (m_mean_subtraction || m_unit_variance || m_unit_scale || m_z_score);
}


void cv_normalizer::set(const bool meansub, const bool unitvar, const bool unitscale, const bool zscore) {
  reset();
  m_mean_subtraction = meansub;
  m_unit_variance = unitvar;
  m_unit_scale = unitscale;
  m_z_score = zscore;
}


void cv_normalizer::reset() {
  m_enabled = false;
  m_trans.clear();
}


bool cv_normalizer::determine_transform(const cv::Mat& image) {
  reset();

  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  if (!check_to_enable()) {
    return false;
  }

  normalization_type ntype = _none;
  if (m_unit_scale) {
    set_normalization_type(ntype, _u_scale);
  }
  if (m_mean_subtraction) {
    set_normalization_type(ntype, _mean_sub);
  }
  if (m_unit_variance) {
    set_normalization_type(ntype, _unit_var);
  }
  if (m_z_score) {
    set_normalization_type(ntype, _z_score);
  }

  ComputeType u_scale = 1.0;
  ComputeType largest = 1.0;

  //if (!m_z_score && m_unit_scale) {
  if (ntype < _z_score) { // !(m_z_score || (m_mean_subtraction && m_unit_variance))
    switch(image.depth()) {
    case CV_8U:
      largest = std::numeric_limits<uint8_t>::max();
      break;
    case CV_8S:
      largest = std::numeric_limits<int8_t>::max();
      break;
    case CV_16U:
      largest = std::numeric_limits<uint16_t>::max();
      break;
    case CV_16S:
      largest = std::numeric_limits<int16_t>::max();
      break;
    case CV_32S:
      largest = std::numeric_limits<int32_t>::max();
      break;
    default:
      return false;
      // Currently, do nothing for non-integral types. However, a set of scaling
      // paramters can be added to the argument list of this function.
    }
    u_scale = static_cast<ComputeType>(1.0)/largest;
  }

  std::vector<ComputeType> mean;
  std::vector<ComputeType> stddev;
  const normalization_type code_wo_uscale = mask_normalization_bits(ntype, _z_score);
  const auto NCh = static_cast<size_t>(image.channels());

  if (code_wo_uscale != _none) {
    if (!compute_mean_stddev(image, mean, stddev) || (NCh != mean.size())) {
      return false;
    }
  #if 0
    for (int ch = 0; ch < image.channels(); ++ch) {
      std::cout << "channel " << ch << "\tmean " << mean[ch] << "\tstddev " << stddev[ch] << std::endl;
    }
  #endif
  }

  m_trans.resize(NCh);

  switch (code_wo_uscale) {
  case _none: // Note that mean.size() is zero in this case
    for (size_t ch=0u; ch < NCh; ++ch) {
      m_trans[ch] = channel_trans_t(u_scale, 0.0);
    }
    break;
  case _mean_sub:
    for (size_t ch=0u; ch < NCh; ++ch) {
      m_trans[ch] = channel_trans_t(u_scale,
                                    - u_scale * mean[ch]);
    }
    break;
  case _unit_var:
    for (size_t ch=0u; ch < NCh; ++ch) {
      if (stddev[ch] > fabs(mean[ch])*(1e-7)) {
        m_trans[ch] =
          channel_trans_t(static_cast<ComputeType>(1.0)/stddev[ch],
                          u_scale * mean[ch] - mean[ch]/stddev[ch]);
      } else {
        m_trans[ch] = channel_trans_t(u_scale, 0.0);
      }
    }
    break;
  case _z_score:
    for (size_t ch=0u; ch < NCh; ++ch) {
      if (stddev[ch] > fabs(mean[ch])*(1e-7)) {
        m_trans[ch] = channel_trans_t(static_cast<ComputeType>(1.0)/stddev[ch],
                                      - mean[ch]/stddev[ch]);
      } else {
        m_trans[ch] = channel_trans_t(0.0, 0.0);
      }
    }
    break;
  default:
    return false;
  }

  m_enabled = true;
  return true;
}


/**
 * Manually invoke normalization before copying image from cv::Mat into
 * El::Matrix<DataType> format. Then, the transform must be disabled to
 * prevent it from being automatically applied again during copying.
 * After the copying is complete, either of the following two is required
 * depending on whether the inverse transform is needed afterwards or not.
 * If no inverse transform is needed , disabling or resetting is ok.
 * As the normalization could have been implicitly applied during copying
 * via scaling, the transform must be disabled after copying.
 * On the other hand, resetting the structure is ok if no inverse transform
 * is needed. Alternatively, the inverse transform can be set.
 */
bool cv_normalizer::apply(cv::Mat& image) {
  m_enabled = false; // turn off as it is applied
  return scale(image, m_trans);
}


/**
 * The actual transform can either be manually invoked, or automatically during
 * copying from a cv::Mat image to El::Matrix<DataType> data to avoid reading
 * the image twice.
 * @param _trans The channel-wise parameters for linear transform
 */
void cv_normalizer::set_transform(const std::vector<channel_trans_t>& _trans) {
  m_trans = _trans;
  m_enabled = true;
}


/**
 * In case that undoing normalization is required, this call arranges it to
 * occur during copying from El::Matrix<DataType> data to a cv::Mat image
 * while avoiding reading the image twice.
 */
bool cv_normalizer::determine_inverse_transform() {
  m_enabled = false; // unless this method is successful, stays disabled
  const size_t NCh = m_trans.size();
  if (NCh == 0u) {
    m_trans.clear();
    return false;
  }

  std::vector<channel_trans_t> trans_reverse(NCh, channel_trans_t(1.0, 0.0));

  for (size_t ch=0u; ch < NCh; ++ch) {
    if (m_trans[ch].first == 0.0) {
      m_trans.clear();
      return false;
    }
    trans_reverse[ch] =
      channel_trans_t(static_cast<ComputeType>(1.0)/m_trans[ch].first,
                      - m_trans[ch].second/m_trans[ch].first);
  }
  trans_reverse.swap(m_trans);

  return (m_enabled = true);
}



bool cv_normalizer::scale(cv::Mat& image, const std::vector<channel_trans_t>& trans) {
  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  switch(image.depth()) {
  case CV_8U:
    return scale_with_known_type<_depth_type(CV_8U),  DataType>(image, trans);
  case CV_8S:
    return scale_with_known_type<_depth_type(CV_8S),  DataType>(image, trans);
  case CV_16U:
    return scale_with_known_type<_depth_type(CV_16U), DataType>(image, trans);
  case CV_16S:
    return scale_with_known_type<_depth_type(CV_16S), DataType>(image, trans);
  case CV_32S:
    return scale_with_known_type<_depth_type(CV_32S), DataType>(image, trans);
  case CV_32F:
    return scale_with_known_type<_depth_type(CV_32F), DataType>(image, trans);
  case CV_64F:
    return scale_with_known_type<_depth_type(CV_64F), DataType>(image, trans);
  }
  return false;
}


bool cv_normalizer::compute_mean_stddev(const cv::Mat& image,
                                        std::vector<ComputeType>& mean, std::vector<ComputeType>& stddev,
                                        cv::InputArray mask) {
  if (image.empty()) {
    return false;
  }
  if (image.channels() > 4) {
    _SWITCH_CV_FUNC_4PARAMS(image.depth(), \
                            compute_mean_stddev_with_known_type, image, mean, stddev, mask)
  } else {
    // cv::meanStdDev() currently only works with double type for mean and stddev and images of 1-4 channels
    using Ch_T = double;
    //using Ch_T = ComputeType;
    using Output_T = cv_image_type<Ch_T>;
    cv::Mat _mean(1, 4, Output_T::T());
    cv::Mat _stddev(1, 4, Output_T::T());
    cv::meanStdDev(image, _mean, _stddev, mask);
    mean.resize(image.channels());
    stddev.resize(image.channels());
    for (int c=0; c < image.channels(); ++c) {
      mean[c] = static_cast<ComputeType>(_mean.at<Ch_T>(0,c));
      stddev[c] = static_cast<ComputeType>(_stddev.at<Ch_T>(0,c));
    }
    return true;
  }
  return false;
}

std::string cv_normalizer::get_description() const {
  std::stringstream os;
  os << get_type() + ":" << std::endl
     << " - mean subtraction: " << (m_mean_subtraction? "true" : "false") << std::endl
     << " - unit variance: " << (m_unit_variance? "true" : "false") << std::endl
     << " - unit scale: " << (m_unit_scale? "true" : "false") << std::endl
     << " - z-score: " << (m_z_score? "true" : "false") << std::endl;
  return os.str();
}

std::ostream& cv_normalizer::print(std::ostream& os) const {
  os << get_description()
     << " - transform:";
  for (const channel_trans_t& tr: m_trans) {
    os << " [" << tr.first << ' ' << tr.second << "]\n             ";
  }
  os << std::endl;

  return os;
}


} // end of namespace lbann
#endif // LBANN_HAS_OPENCV
