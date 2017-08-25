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
// lbann_cv_resizer .cpp .hpp - functions to resize images
//                              in opencv format
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/cv_resizer.hpp"
#include "lbann/utils/mild_exception.hpp"
#include <algorithm>
#include <ostream>
//#include <iostream>

#ifdef __LIB_OPENCV
namespace lbann {

cv_resizer::cv_resizer()
  : cv_transform(), m_width(0u), m_height(0u), m_uniform_zoom(true),
    m_center(std::pair<float,float>(0.0f, 0.0f)),
    m_roi_sz(std::pair<int,int>(0,0)),
    //m_roi(cv::Rect()), m_center_roi(0.0,0.0),
    m_hzoom(0.0f), m_vzoom(0.0f), m_interpolation(cv::INTER_LINEAR) {
  //check_enabled(); // enable if default parameter changes
}


cv_resizer *cv_resizer::clone() const {
  return new cv_resizer(*this);
}


bool cv_resizer::check_center(const cv::Mat& image) {
  const int width = image.cols;
  const int height = image.rows;

  if ((m_center.first >= 1.0f) || (m_center.second >= 1.0f) ||
      (m_center.first <= 0.0f) || (m_center.second <= 0.0f)) {
#if 0
    return false;
#else
    m_center = std::pair<float, float>(0.5f, 0.5f);
#endif
  }

  if ((static_cast<int>(m_center.first * width) == 0) ||
      (static_cast<int>(m_center.second * height) == 0)) {
    return false;
  }

  return true;
}


void cv_resizer::set(const unsigned int width, const unsigned int height, const bool uzoom,
                     const std::pair<float, float>& center, const std::pair<int, int>& roi_sz) {
  m_width = width;
  m_height = height;
  m_uniform_zoom = uzoom;
  m_center = center;
  m_roi_sz = roi_sz;

  m_roi = cv::Rect();
  m_center_roi = cv::Point_<float>(0.0,0.0);
  m_hzoom = 0.0f;
  m_vzoom = 0.0f;
  m_interpolation = cv::INTER_LINEAR;
  m_enabled = false; // will turns on when the transform is determined
}


void cv_resizer::reset() {
  m_width = 0u;
  m_height = 0u;
  m_uniform_zoom = true;
  m_center = std::pair<float, float>(0.5f, 0.5f);
  m_roi_sz = std::pair<int, int>(0, 0);
  m_roi = cv::Rect();
  m_center_roi = cv::Point_<float>(0.0,0.0);
  m_hzoom = 0.0f;
  m_vzoom = 0.0f;
  m_interpolation = cv::INTER_LINEAR;
  m_enabled = false;
}


bool cv_resizer::determine_initial_roi(const cv::Mat& image) {
  if ((m_roi_sz.first > 0) && (m_roi_sz.second > 0)) {
    // do initial crop
    const int cx_roi = static_cast<int>(image.cols * m_center.first);
    const int cy_roi = static_cast<int>(image.rows * m_center.second);

    int roi_width = m_roi_sz.first;
    int roi_height = m_roi_sz.second;

    int x0 = cx_roi - roi_width/2;
    int y0 = cy_roi - roi_height/2;

    if (x0 < 0) {
      roi_width += x0;
      x0 = 0;
    }

    if (y0 < 0) {
      roi_height += y0;
      y0 = 0;
    }

    if (x0+roi_width > image.cols) {
      roi_width = image.cols - x0;
    }

    if (y0+roi_height > image.rows) {
      roi_height = image.rows - y0;
    }

    m_roi = cv::Rect(x0, y0, roi_width, roi_height);

    m_center_roi.x = static_cast<float>(cx_roi - x0)/roi_width;
    m_center_roi.y = static_cast<float>(cy_roi - y0)/roi_height;
  } else if ((m_roi_sz.first == 0) && (m_roi_sz.second == 0)) {
    m_roi = cv::Rect(0, 0, image.cols, image.rows);
    m_center_roi.x = m_center.first;
    m_center_roi.y = m_center.second;
  }
  return true;
}


bool cv_resizer::determine_zoom_ratio(const cv::Mat& image) {
  const unsigned int x_center = static_cast<unsigned int>(m_roi.width * m_center_roi.x);
  const unsigned int half_width_short =
    std::min(m_roi.width - x_center, x_center);

  const unsigned int y_center = static_cast<unsigned int>(m_roi.height * m_center_roi.y);
  const unsigned int half_height_short =
    std::min(m_roi.height - y_center, y_center);

  if ((half_width_short == 0u) || (half_height_short == 0u)) {
    return false;
  }

  m_hzoom = m_width / (2.0f * half_width_short);
  m_vzoom = m_height / (2.0f * half_height_short);

  if (m_uniform_zoom) {
    m_hzoom = m_vzoom = std::max(m_hzoom, m_vzoom);
  }

  m_interpolation = cv::INTER_LINEAR;

  if ((m_hzoom < 1.0f) && (m_vzoom < 1.0f)) {
    m_interpolation = cv::INTER_AREA; // (better for shrinking)
  } else if ((m_hzoom > 1.0f) && (m_vzoom > 1.0f)) {
#if 0
    m_interpolation = cv::INTER_CUBIC; // (slow but better)
#else
    m_interpolation = cv::INTER_LINEAR; // (faster but ok)
#endif
  }

  return true;
}


bool cv_resizer::determine_transform(const cv::Mat& image) {
  m_enabled = false; // unless this method is successful, stays disabled

  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  bool ok = ((m_width > 0u) && (m_height > 0u));
  ok = ok && check_center(image);
  ok = ok && determine_initial_roi(image);
  ok = ok && determine_zoom_ratio(image);
  //_LBANN_MILD_EXCEPTION(!ok, "Failed to determine resizing method", false)
  if (!ok) {
    return false;
  }

  m_enabled = true;
  return true;
}


bool cv_resizer::apply(cv::Mat& image) {
  m_enabled = false; // turn off as it is applied

  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  //std::cout << "initial crop; at (" << m_roi.x << ", " << m_roi.y
  //          << ") for " << m_roi.width << " x " << m_roi.height << std::endl;

  cv::Mat image_roi = image(m_roi);

  const unsigned int zoomed_width = static_cast<unsigned int>(image_roi.cols * m_hzoom + 0.5f);
  const unsigned int zoomed_height = static_cast<unsigned int>(image_roi.rows * m_vzoom + 0.5f);

  _LBANN_MILD_EXCEPTION((zoomed_width < m_width) || (zoomed_height < m_height),
                        "Zoom ratio is not large enough.", false);

  //std::cout << "zoom to " << zoomed_width << " x " << zoomed_height << std::endl;

  cv::Mat zoomed_image;
  cv::resize(image_roi, zoomed_image, cv::Size(zoomed_width, zoomed_height), 0, 0, m_interpolation);

  int cx = static_cast<int>(zoomed_width * m_center_roi.x);
  int cy = static_cast<int>(zoomed_height * m_center_roi.y);

  int px0 = cx - m_width/2;
  int py0 = cy - m_height/2;
  if (px0 < 0) {
    px0 = 0;
  }
  if (py0 < 0) {
    py0 = 0;
  }

  if (static_cast<unsigned int>(px0 + m_width) > zoomed_width) {
    px0 = zoomed_width - m_width;
  }

  if (static_cast<unsigned int>(py0 + m_height) > zoomed_height) {
    py0 = zoomed_height - m_height;
  }

  _LBANN_MILD_EXCEPTION((px0 < 0) || (py0 < 0), "Failed to resize", false);

  //std::cout << "taking at (" << px0 << ", " << py0
  //          << ") for " << m_width << " x " << m_height << std::endl;

  image = zoomed_image(cv::Rect(px0, py0, m_width, m_height));

  return true;
}


std::ostream& cv_resizer::print(std::ostream& os) const {
  os << "m_width: " << m_width << std::endl
     << "m_height: " << m_height << std::endl
     << "m_uniform_zoom: " << (m_uniform_zoom? "true" : "false") << std::endl
     << "m_center: " << m_center.first << " " << m_center.second << std::endl
     << "m_roi_sz: " << m_roi_sz.first << " " << m_roi_sz.second << std::endl
     << "m_roi: " << m_roi.width << " x " << m_roi.height << " at ("
     << m_roi.x << ", " << m_roi.y << ")" << std::endl
     << "m_center_roi: " << m_center_roi.x << " " << m_center_roi.y << std::endl
     << "m_hzoom: " << m_hzoom << std::endl
     << "m_vzoom: " << m_vzoom << std::endl;
  return os;
}

} // end of namespace lbann
#endif // __LIB_OPENCV

