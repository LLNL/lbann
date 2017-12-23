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
// cv_subtractor .cpp .hpp - subtract channel values of an image (possibly the
// pixel-wise mean of dataset) from the corresponding values of another (input)
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/cv_subtractor.hpp"
#include "lbann/data_readers/cv_utils.hpp"
#include "lbann/utils/mild_exception.hpp"

#ifdef __LIB_OPENCV
namespace lbann {

cv_subtractor::cv_subtractor(const cv_subtractor& rhs)
  : cv_transform(rhs),
    m_img_to_sub(rhs.m_img_to_sub),
    m_subtracted(rhs.m_subtracted) {}

cv_subtractor& cv_subtractor::operator=(const cv_subtractor& rhs) {
  cv_transform::operator=(rhs);
  m_img_to_sub = rhs.m_img_to_sub;
  m_subtracted = rhs.m_subtracted;
  return *this;
}

cv_subtractor *cv_subtractor::clone() const {
  return (new cv_subtractor(*this));
}

void cv_subtractor::set(const std::string name_of_img_to_sub, const int depth_code) {
  // TODO: check extention and load appropriately
  cv::Mat img_to_sub = cv::imread(name_of_img_to_sub);
  set(img_to_sub, depth_code);
}

void cv_subtractor::set(const cv::Mat& image, const int depth_code) {
  reset();

  const double f = cv_utils::get_depth_normalizing_factor(image.depth());

  // Make sure that the image is set as a floating point type image
  // Note that this is the only way to set m_img_to_sub. This means that
  // m_img_to_sub will be of a floating point type unless it is empty.

  if ((depth_code != CV_32F) && (depth_code != CV_64F)) {
    // If the depth_code does not indicate a floating point type, see if the
    // image is already of a floating point type. If so, use the same type.
    // Otherwise, use the type of LBANN's DataType.
    if (check_if_cv_Mat_is_float_type(image)) {
      image.convertTo(m_img_to_sub, image.depth(), f, 0.0);
    } else {
      image.convertTo(m_img_to_sub, cv_image_type<::DataType>::T(), f, 0.0);
    }
  } else {
    image.convertTo(m_img_to_sub, depth_code, f, 0.0);
  }
}

bool cv_subtractor::determine_transform(const cv::Mat& image) {
  //reset(); // redundant here
  // enable if the given image has the same size and the same number of
  // channels as the image to subtract.
  m_subtracted = false;
  m_enabled = check_if_cv_Mat_has_same_shape(image, m_img_to_sub);
  return m_enabled;
}

bool cv_subtractor::determine_inverse_transform() {
  return (m_enabled = m_subtracted);
}

bool cv_subtractor::apply(cv::Mat& image) {
  m_enabled = false; // turn off as the transform is applied once
  if (m_subtracted) { // inverse if applied already
    const double f = cv_utils::get_depth_denormalizing_factor(CV_8U);
    cv::Mat image_sub;
    cv::addWeighted(m_img_to_sub, f, image, 1.0, 0.0, image_sub, CV_8U);
    image = image_sub;
    m_subtracted = false;
  } else {
    const double f = cv_utils::get_depth_normalizing_factor(image.depth());
    cv::Mat image_sub;
    cv::addWeighted(m_img_to_sub, -1.0, image, f, 0.0, image_sub, m_img_to_sub.depth());
    image = image_sub;
    m_subtracted = true;
  }

  return true;
}

std::string cv_subtractor::get_description() const {
  std::stringstream os;
  os << get_type() + ":" << std::endl;
  return os.str();
}

std::ostream& cv_subtractor::print(std::ostream& os) const {
  os << get_description()
     << " - configuration of the image to subtract: "
     << m_img_to_sub.cols << 'x' << m_img_to_sub.rows << 'x' << m_img_to_sub.channels() << std::endl;
  return os;
}

} // end of namespace lbann
#endif // __LIB_OPENCV
