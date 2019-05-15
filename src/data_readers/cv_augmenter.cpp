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
// cv_augmenter .cpp .hpp - Augmenting functions for images in opencv format
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/cv_augmenter.hpp"
#include "lbann/utils/mild_exception.hpp"
#include "lbann/utils/random.hpp"

#ifdef LBANN_HAS_OPENCV
namespace lbann {

cv_augmenter::cv_augmenter()
  : cv_transform(),
    m_do_horizontal_flip(false),
    m_do_vertical_flip(false),
    m_rotation_range(0.0f),
    m_horizontal_shift_range(0.0f),
    m_vertical_shift_range(0.0f),
    m_shear_range(0.0f),
    m_flip(_no_flip_),
    m_trans(cv::Mat_<float>::eye(3,3)) {
  //check_enabled(); // enable if default parameter changes
}


cv_augmenter::cv_augmenter(const cv_augmenter& rhs)
  : cv_transform(rhs),
    m_do_horizontal_flip(rhs.m_do_horizontal_flip),
    m_do_vertical_flip(rhs.m_do_vertical_flip),
    m_rotation_range(rhs.m_rotation_range),
    m_horizontal_shift_range(rhs.m_horizontal_shift_range),
    m_vertical_shift_range(rhs.m_vertical_shift_range),
    m_shear_range(rhs.m_shear_range),
    m_flip(rhs.m_flip),
    m_trans(rhs.m_trans) {
}

cv_augmenter *cv_augmenter::clone() const {
  return new cv_augmenter(*this);
}

cv_augmenter& cv_augmenter::operator=(const cv_augmenter& rhs) {
  if (this == &rhs) {
    return (*this);
  }

  cv_transform::operator=(rhs);
  m_do_horizontal_flip = rhs.m_do_horizontal_flip;
  m_do_vertical_flip = rhs.m_do_vertical_flip;
  m_rotation_range = rhs.m_rotation_range;
  m_horizontal_shift_range = rhs.m_horizontal_shift_range;
  m_vertical_shift_range = rhs.m_vertical_shift_range;
  m_shear_range = rhs.m_shear_range;
  m_flip = rhs.m_flip;
  m_trans = rhs.m_trans;

  return (*this);
}


bool cv_augmenter::check_to_enable() const {
  return ( m_do_horizontal_flip ||
           m_do_vertical_flip ||
          (m_horizontal_shift_range != 0.0f) ||
          (m_vertical_shift_range != 0.0f) ||
          (m_shear_range != 0.0f) ||
          (m_rotation_range != 0.0f));
}


void cv_augmenter::set(const bool hflip, const bool vflip, const float rot,
                       const float hshift, const float vshift, const float shear) {
  reset();
  m_do_horizontal_flip = hflip;
  m_do_vertical_flip = vflip;
  m_rotation_range = rot;
  m_horizontal_shift_range = hshift;
  m_vertical_shift_range = vshift;
  m_shear_range = shear;
}


void cv_augmenter::reset() {
  m_enabled = false; // will turns on when the transform is determined
  m_flip = _no_flip_;
  m_trans = cv::Mat_<float>::eye(3,3);
}


bool cv_augmenter::determine_transform(const cv::Mat& image) {
  reset();

  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  if (!check_to_enable()) {
    return false;
  }

  rng_gen& gen = get_io_generator();

  std::uniform_int_distribution<int> bool_dist(0, 1);

  // Flips
#ifdef _COMPAT_WITH_EL_AUGMENT_
  const bool horiz_flip = bool_dist(gen) && m_do_horizontal_flip;
  const bool vert_flip = bool_dist(gen) && m_do_vertical_flip;
#else
  const bool horiz_flip = m_do_horizontal_flip && bool_dist(gen);
  const bool vert_flip = m_do_vertical_flip && bool_dist(gen);
#endif

  if (horiz_flip && vert_flip) {
    m_flip = _both_axes_;
  } else if (horiz_flip) {
    m_flip = _horizontal_;
  } else if (vert_flip) {
    m_flip = _vertical_;
  } else {
    m_flip = _no_flip_;
  }

  // Shift (Translate)
  float x_shift = 0.0f;
  float y_shift = 0.0f;
  if (m_horizontal_shift_range != 0.0f) {
    std::uniform_real_distribution<float> dist(-m_horizontal_shift_range,
        m_horizontal_shift_range);
    x_shift = dist(gen) * image.cols;
  }
  if (m_vertical_shift_range != 0.0f) {
    std::uniform_real_distribution<float> dist(-m_vertical_shift_range,
        m_vertical_shift_range);
    y_shift = dist(gen) * image.rows;
  }
  cv::Mat_<float> shift_mat = cv::Mat_<float>::eye(3,3);
  shift_mat(0, 2) = x_shift;
  shift_mat(1, 2) = y_shift;
  //std::cout << "x_shift " << x_shift << ",    y_shift " << y_shift << std::endl;

  // Shearing
  float shear = 0.0f;
  if (m_shear_range != 0.0f) {
    std::uniform_real_distribution<float> dist(-m_shear_range,
        m_shear_range);
    shear = dist(gen);
  }
  cv::Mat_<float> shear_mat = cv::Mat_<float>::zeros(3,3);
  shear_mat(0, 0) = 1.0f;
  shear_mat(2, 2) = 1.0f;
  shear_mat(0, 1) = -std::sin(shear);
  shear_mat(1, 1) = std::cos(shear);
  //std::cout << "shear " << shear << std::endl;

  // Rotation
  float rotate = 0.0f;
  if (m_rotation_range != 0.0f) {
    std::uniform_real_distribution<float> dist(-m_rotation_range,
        m_rotation_range);
    rotate = pi / 180.0f * dist(gen);
  }
  cv::Mat_<float> rot_mat = cv::Mat_<float>::zeros(3,3);
  rot_mat(2, 2) = 1.0f;
  rot_mat(0, 0) = std::cos(rotate);
  rot_mat(0, 1) = -std::sin(rotate);
  rot_mat(1, 0) = std::sin(rotate);
  rot_mat(1, 1) = std::cos(rotate);
  //std::cout << "rotate " << rotate << std::endl;

  // Compute the final transformation.
#if 0
  cv::Mat_<float> tmp_mat = cv::Mat_<float>::zeros(3, 3);
  cv::gemm(shift_mat, shear_mat, 1.0f, tmp_mat, 0.0f, tmp_mat, 0);
  cv::gemm(tmp_mat, rot_mat, 1.0f, m_trans, 0.0f, m_trans, 0);
#else
  //m_trans = (shift_mat * shear_mat) * rot_mat;
  m_trans = shear_mat * rot_mat;
  m_trans(0,2) = x_shift;
  m_trans(1,2) = y_shift;
#endif

  return (m_enabled = true);
}


bool cv_augmenter::apply(cv::Mat& image) {
  m_enabled = false; // turn off as it is applied

  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  cv::Mat image_copy;

  if (m_flip == _no_flip_) {
    image_copy = image.clone();
  } else {
    cv::flip(image, image_copy, static_cast<int>(m_flip));
  }

  cv::Mat_<float> _trans(m_trans, cv::Rect_<float>(0,0,3,2));

  cv::warpAffine(image_copy, image, _trans, image.size(),
                 cv::INTER_LINEAR, cv::BORDER_REPLICATE);

  return true;
}

std::string cv_augmenter::get_description() const {
  std::stringstream os;
  os << get_type() + ":" << std::endl
     << " - horizontal flip: " << (m_do_horizontal_flip? "true" : "false") << std::endl
     << " - vertical flip: " << (m_do_vertical_flip? "true" : "false") << std::endl
     << " - rotation range: " << m_rotation_range << std::endl
     << " - horizontal shift range: " << m_horizontal_shift_range << std::endl
     << " - vertical shift range: " << m_vertical_shift_range << std::endl
     << " - shear range: " << m_shear_range << std::endl;
  return os.str();
}

std::ostream& cv_augmenter::print(std::ostream& os) const {
  os << get_description()
     << " - flipping: " << cv_transform::flip_desc(m_flip) << std::endl << std::fixed
     << " - transfrom: " << m_trans(0,0) << '\t' << m_trans(0,1) << '\t' << m_trans(0,2)  << std::endl
     << "              " << m_trans(1,0) << '\t' << m_trans(1,1) << '\t' << m_trans(1,2)  << std::endl
     << "              " << m_trans(2,0) << '\t' << m_trans(2,1) << '\t' << m_trans(2,2)  << std::endl; //<< std::defaultfloat;

  return os;
}

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV
