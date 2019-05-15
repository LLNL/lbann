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
// cv_subtractor .cpp .hpp - subtract channel values of an image (possibly the
// pixel-wise mean of dataset) from the corresponding values of another (input)
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/cv_subtractor.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/mild_exception.hpp"
#include "lbann/utils/file_utils.hpp"
#include <iostream>
#include <fstream>
#include <iterator>

#ifdef LBANN_HAS_OPENCV
namespace lbann {

cv_subtractor::cv_subtractor(const cv_subtractor& rhs)
  : cv_transform(rhs),
    m_img_to_sub(rhs.m_img_to_sub),
    m_img_to_div(rhs.m_img_to_div),
    m_channel_mean(rhs.m_channel_mean),
    m_channel_stddev(rhs.m_channel_stddev),
    m_applied(rhs.m_applied)
{}

cv_subtractor& cv_subtractor::operator=(const cv_subtractor& rhs) {
  cv_transform::operator=(rhs);
  m_img_to_sub = rhs.m_img_to_sub;
  m_img_to_div = rhs.m_img_to_div;
  m_channel_mean = rhs.m_channel_mean;
  m_channel_stddev = rhs.m_channel_stddev;
  m_applied = rhs.m_applied;
  return *this;
}

cv_subtractor *cv_subtractor::clone() const {
  return (new cv_subtractor(*this));
}

/**
 * Load an image in the file of the proprietary format.
 * The file name describes the image configuration as:
 *   *-(width)x(height)x(num_channels)-(opencv_depth_code).bin
 * There is no header in the file. The file is a binary dump of an OpenCV cv::Mat data.
 * For the better portability, an existing format can be used to carry image data.
 */
cv::Mat cv_subtractor::read_binary_image_file(const std::string filename) {
  std::vector<int> tokens;
  { // Extract the information on the image from the file name
    const std::vector<char> delims = {'-', 'x','x','-','.'};
    std::string dir;
    std::string basename;

    parse_path(filename, dir, basename);
    tokens = get_tokens(basename, delims);
    if (tokens.size() != delims.size()) {
      return cv::Mat();
    }
  }

  std::ifstream file(filename, std::ios::binary);
  if (!file.good()) {
    return cv::Mat();
  }
  file.unsetf(std::ios::skipws);

  { // Check file size
    const size_t image_byte_size
      = tokens[1] * tokens[2] * tokens[3] * CV_ELEM_SIZE(tokens[4]);

    file.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(file.tellg());
    if (image_byte_size != file_size) {
      return cv::Mat();
    }
  }

  // Construct an image data structure
  cv::Mat image(tokens[1], tokens[2], CV_MAKETYPE(tokens[4], tokens[3]));

  // Reset the file pointer
  file.seekg(0, std::ios::beg);

  // Load the image from the file
  std::copy(std::istream_iterator<unsigned char>(file),
            std::istream_iterator<unsigned char>(),
            reinterpret_cast<unsigned char*>(image.data));

  return image;
}

void cv_subtractor::set_mean(const std::string name_of_img_to_sub, const int depth_code) {
  cv::Mat img_to_sub;
  std::string ext = get_ext_name(name_of_img_to_sub);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  if (ext == "bin") {
    img_to_sub = read_binary_image_file(name_of_img_to_sub);
  } else { // let OpenCV handle
    img_to_sub = cv::imread(name_of_img_to_sub);
  }
  if (img_to_sub.empty()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: cv_subtractor: cannot load the image "
        << name_of_img_to_sub << " to subtract.";
    throw lbann_exception(err.str());
  }
  set_mean(img_to_sub, depth_code);
  m_channel_mean.clear();

  if (m_channel_stddev.empty() && !m_img_to_div.empty() &&
      !check_if_cv_Mat_has_same_shape(m_img_to_div, m_img_to_sub)) {
    throw lbann_exception("cv_subtractor::set_mean() : mean and variance images have different shapes");
  }
}

void cv_subtractor::set_mean(const std::vector<lbann::DataType> ch_mean) {
  if (ch_mean.size() > cv::Scalar::channels) {
    throw lbann_exception(std::string("cv_subtractor::set_mean() : ") +
      "provide the mean image if the number of channels are larger than " +
      std::to_string(cv::Scalar::channels) + '.');
  }
  m_channel_mean = ch_mean;
}

bool cv_subtractor::create_img_to_sub(int width, int height, int n_channels) {
  if ((n_channels == 0) || (static_cast<size_t>(n_channels) != m_channel_mean.size()) ||
      (width == 0) || (height == 0)) {
    return false;
  }
  const std::vector<lbann::DataType>& ch_mean = m_channel_mean;
  cv::Scalar px = cv::Scalar::all(0.0);
  for (size_t i = 0u; i < ch_mean.size(); ++i) {
    px[static_cast<int>(i)] = ch_mean[i];
  }
  cv::Mat img_to_sub(height, width, cv_image_type<lbann::DataType>::T(n_channels), px);
  set_mean(img_to_sub);
  return true;
}

void cv_subtractor::set_mean(const cv::Mat& image, const int depth_code) {
  reset();

  const double f = get_depth_normalizing_factor(image.depth());

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
      image.convertTo(m_img_to_sub, cv_image_type<lbann::DataType>::T(), f, 0.0);
    }
  } else {
    image.convertTo(m_img_to_sub, depth_code, f, 0.0);
  }
}

void cv_subtractor::set_stddev(const std::string name_of_img_to_div, const int depth_code) {
  cv::Mat img_to_div;
  std::string ext = get_ext_name(name_of_img_to_div);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  if (ext == "bin") {
    img_to_div = read_binary_image_file(name_of_img_to_div);
  } else { // let OpenCV handle
    img_to_div = cv::imread(name_of_img_to_div);
  }
  if (img_to_div.empty()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: cv_subtractor: cannot load the image "
        << name_of_img_to_div << " to normalize.";
    throw lbann_exception(err.str());
  }
  set_stddev(img_to_div, depth_code);
  m_channel_stddev.clear();

  if (m_channel_mean.empty() && !m_img_to_sub.empty() &&
      !check_if_cv_Mat_has_same_shape(m_img_to_sub, m_img_to_div)) {
    throw lbann_exception("cv_subtractor::set_stddev() : mean and variance images have different shapes.");
  }
}

void cv_subtractor::set_stddev(const std::vector<lbann::DataType> ch_stddev) {
  if (ch_stddev.size() > cv::Scalar::channels) {
    throw lbann_exception(std::string("cv_subtractor::set_stddev() : ") +
      "provide the stddev image if the number of channels are larger than " +
      std::to_string(cv::Scalar::channels) + '.');
  }
  m_channel_stddev = ch_stddev;
}

bool cv_subtractor::create_img_to_div(int width, int height, int n_channels) {
  if ((n_channels == 0) || (static_cast<size_t>(n_channels) != m_channel_stddev.size()) ||
      (width == 0) || (height == 0)) {
    return false;
  }
  const std::vector<lbann::DataType>& ch_stddev = m_channel_stddev;
  cv::Scalar px = cv::Scalar::all(0.0);
  for (size_t i = 0u; i < ch_stddev.size(); ++i) {
    px[static_cast<int>(i)] = ch_stddev[i];
  }
  cv::Mat img_to_div(height, width, cv_image_type<lbann::DataType>::T(n_channels), px);
  set_stddev(img_to_div);
  return true;
}

void cv_subtractor::set_stddev(const cv::Mat& image, const int depth_code) {
  reset();

  const double f = get_depth_normalizing_factor(image.depth());

  if ((depth_code != CV_32F) && (depth_code != CV_64F)) {
    if (check_if_cv_Mat_is_float_type(image)) {
      image.convertTo(m_img_to_div, image.depth(), f, 0.0);
    } else {
      image.convertTo(m_img_to_div, cv_image_type<lbann::DataType>::T(), f, 0.0);
    }
  } else {
    image.convertTo(m_img_to_div, depth_code, f, 0.0);
  }
}

bool cv_subtractor::determine_transform(const cv::Mat& image) {
  reset();
  if (m_channel_mean.empty()) {
    if (!m_img_to_sub.empty()) { // pixel-wise
      if (!check_if_cv_Mat_has_same_shape(image, m_img_to_sub)) {
        throw lbann_exception(std::string("cv_subtactor::determine_transform(): ") +
                              "input and mean images have different sizes.");
      }
      m_enabled = true;
    }
  } else { // channel-wise
    if (!check_if_cv_Mat_has_same_shape(image, m_img_to_sub) &&
        !create_img_to_sub(image.cols, image.rows, image.channels())) {
      throw lbann_exception(std::string("cv_subtactor::determine_transform(): ") +
                            "failed to create mean image.");
    }
    m_enabled = true;
  }
  if (m_channel_stddev.empty()) {
    if (!m_img_to_div.empty()) { // pixel-wise
      if (!check_if_cv_Mat_has_same_shape(image, m_img_to_div)) {
        throw lbann_exception(std::string("cv_subtactor::determine_transform(): ") +
                              "input and stddev images have different sizes.");
      }
      m_enabled = true;
    }
  } else { // channel-wise
    if (!check_if_cv_Mat_has_same_shape(image, m_img_to_div) &&
        !create_img_to_div(image.cols, image.rows, image.channels())) {
      throw lbann_exception(std::string("cv_subtactor::determine_transform(): ") +
                            "failed to create stddev image.");
    }
    m_enabled = true;
  }
  return m_enabled;
}

bool cv_subtractor::determine_inverse_transform() {
  return (m_enabled = m_applied);
}

/**
 * Currently only supports mean-subtraction and z-score.
 * TODO: Unit variance is not supported. It can be implemented by adding
 * 'm_img_to_sub' to the result of z-score. Both z-score and unit variance
 * requires both mean and stddev. Thus, we would need an additional flag to
 * distinguish which method is being set up.
 */
bool cv_subtractor::apply(cv::Mat& image) {
  m_enabled = false; // turn off as the transform is applied once
  if (m_applied) { // inverse if applied already
    double f = get_depth_denormalizing_factor(CV_8U);

    cv::Mat image_new;

    if (!m_img_to_div.empty()) {
      double ff = 1.0;
      if (m_img_to_sub.empty()) {
        ff = f;
        f = 1.0;
      }
      cv::multiply(image, m_img_to_div, image_new, ff, m_img_to_div.depth());
      image = image_new;
    }

    if (!m_img_to_sub.empty()) {
      cv::addWeighted(m_img_to_sub, f, image, f, 0.0, image_new, CV_8U);
      image = image_new;
    }

    m_applied = false;
  } else {
    double f = get_depth_normalizing_factor(image.depth());

    cv::Mat image_new;
    if (!m_img_to_sub.empty()) {
      cv::addWeighted(m_img_to_sub, -1.0, image, f, 0.0, image_new, m_img_to_sub.depth());
      f = 1.0; // to avoid redundant depth normalization
      image = image_new;
    }

    if (!m_img_to_div.empty()) {
      cv::divide(image, m_img_to_div, image_new, f, m_img_to_div.depth());
      image = image_new;
    }

    m_applied = true;
  }

  return true;
}

bool cv_subtractor::check_if_channel_wise() const {
  return !(m_channel_mean.empty() || m_channel_stddev.empty());
}

std::string cv_subtractor::get_description() const {
  std::stringstream os;
  os << get_type() + ":" << std::endl;
  return os.str();
}

std::ostream& cv_subtractor::print(std::ostream& os) const {
  os << get_description()
     << " - image shape to subtract: "
     << m_img_to_sub.cols << 'x' << m_img_to_sub.rows
     << 'x' << m_img_to_sub.channels()
     << '-' << m_img_to_sub.depth() << std::endl
     << " - image shape to divide: "
     << m_img_to_div.cols << 'x' << m_img_to_div.rows
     << 'x' << m_img_to_div.channels()
     << '-' << m_img_to_div.depth() << std::endl;

  os << " - mean per channel to subtract:";
  for (const auto v: m_channel_mean) {
    os << ' ' << v;
  }
  os << std::endl;

  os << " - stddev per channel to divide:";
  for (const auto v: m_channel_stddev) {
    os << ' ' << v;
  }
  os << std::endl;

#if 0
  double f = get_depth_denormalizing_factor(CV_8U);
  if (!m_img_to_sub.empty()) {
    cv::Mat img_sub;
    m_img_to_sub.convertTo(img_sub, CV_8U, f, 0.0);
    cv::imwrite("img_sub.png", img_sub);
  }
  if (!m_img_to_div.empty()) {
    cv::Mat img_div;
    m_img_to_div.convertTo(img_div, CV_8U, f, 0.0);
    cv::imwrite("img_div.png", img_div);
  }
#endif
  return os;
}

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV
