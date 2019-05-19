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
// cv_resizer .cpp .hpp - Functions to resize images
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/cv_resizer.hpp"
#include "lbann/utils/exception.hpp"
#include <algorithm>
#include <ostream>

#ifdef LBANN_HAS_OPENCV
namespace lbann {

const int cv_resizer::m_interpolation_choices[3] = {cv::INTER_LINEAR, cv::INTER_AREA, cv::INTER_LINEAR};

cv_resizer::cv_resizer()
  : cv_transform(), m_width(0u), m_height(0u),
    m_interpolation(m_interpolation_choices[0]),
    m_adaptive_interpolation(false) {}


cv_resizer *cv_resizer::clone() const {
  return new cv_resizer(*this);
}

void cv_resizer::set(const unsigned int width, const unsigned int height,
                     const bool adaptive_interpolation) {
  reset();
  m_width = width;
  m_height = height;
  m_adaptive_interpolation = adaptive_interpolation;

  if ((m_width == 0u) || (m_height == 0u)) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: cv_resizer: invalid size of the resized image";
    throw lbann_exception(err.str());
  }
}

void cv_resizer::reset() {
  m_enabled = false;
  m_interpolation = m_interpolation_choices[0];
}

bool cv_resizer::determine_transform(const cv::Mat& image) {
  m_enabled = false; //sufficient for now in place of reset();

  if (image.empty()) {
    throw lbann_exception("cv_resizer::determine_transform : empty image.");
  }

  const double zoom = image.cols * image.rows / static_cast<double>(m_width * m_height);

  if (zoom <= 1.0) { // shirinking
    m_interpolation =  m_interpolation_choices[static_cast<int>(m_adaptive_interpolation)];
  } else { // enlarging
    m_interpolation =  m_interpolation_choices[static_cast<int>(m_adaptive_interpolation) << 1];
  }

  return (m_enabled = true);
}

bool cv_resizer::apply(cv::Mat& image) {
  m_enabled = false; // turn off as it is applied

  cv::Mat image_new;
  cv::resize(image, image_new, cv::Size(m_width, m_height), 0, 0, m_interpolation);
  image = image_new;

  return true;
}

std::string cv_resizer::get_description() const {
  std::stringstream os;
  os << get_type() + ":" << std::endl
     << " - desired size: " << m_width  << "x" << m_height << std::endl
     << " - adaptive interpolation: " << m_adaptive_interpolation << std::endl;
  return os.str();
}

std::ostream& cv_resizer::print(std::ostream& os) const {
  os << get_description()
     << " - interpolation: ";
  switch(m_interpolation) {
    case cv::INTER_LINEAR: os << "INTER_LINEAR" << std::endl; break;
    case cv::INTER_CUBIC:  os << "INTER_CUBIC" << std::endl; break;
    case cv::INTER_AREA:   os << "INTER_AREA" << std::endl; break;
    default: os << "unrecognized" << std::endl; break;
  }
  return os;
}

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV
