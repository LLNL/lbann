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
// cv_cropper .cpp .hpp - functions to crop images
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/cv_cropper.hpp"
#include "lbann/utils/mild_exception.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/exception.hpp"
#include <algorithm>
#include <ostream>

#ifdef __LIB_OPENCV
namespace lbann {

cv_cropper::cv_cropper()
  : cv_transform(), m_width(0u), m_height(0u),
    m_rand_crop(false), m_is_roi_set(false),
    m_roi_size(std::pair<int,int>(0,0)),
    m_zoom(1.0), m_interpolation(cv::INTER_AREA) {
}


cv_cropper *cv_cropper::clone() const {
  return new cv_cropper(*this);
}

void cv_cropper::unset_roi(void) {
  m_is_roi_set = false;
  m_roi_size = std::pair<int, int>(0, 0);
}

void cv_cropper::set(const unsigned int width, const unsigned int height,
                     const bool random_crop,
                     const std::pair<int, int>& roi_sz) {
  m_width = width;
  m_height = height;
  m_rand_crop = random_crop;

  if ((roi_sz.first > 0) && (roi_sz.second > 0)) {
    if (((unsigned) roi_sz.first < width) || ((unsigned) roi_sz.second < height)) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: cv_cropper: ROI size is smaller than that of a patch";
      throw lbann_exception(err.str());
    } else {
      m_is_roi_set = true;
      m_roi_size = roi_sz;
    }
  } else if (!((roi_sz.first == 0) && (roi_sz.second == 0))) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: cv_cropper: invalid ROI size";
    throw lbann_exception(err.str());
  } else {
    unset_roi();
  }

  m_zoom = 1.0; // default
  m_interpolation = cv::INTER_AREA; // default
  m_enabled = false; // will turns on when the transform is determined
}


void cv_cropper::reset() {
  m_width = 0u;
  m_height = 0u;
  m_rand_crop = false;
  unset_roi();
  m_zoom = 1.0;
  m_interpolation = cv::INTER_AREA;
  m_enabled = false;
}


bool cv_cropper::determine_transform(const cv::Mat& image) {
  m_enabled = false; // unless this method is successful, stays disabled

  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  int roi_width = m_roi_size.first;
  int roi_height = m_roi_size.second;

  if (!m_is_roi_set) {
    roi_width = image.cols;
    roi_height = image.rows;
  }

  double zoom_h = static_cast<double>(roi_width) / image.cols;
  double zoom_v = static_cast<double>(roi_height) / image.rows;
  m_zoom = std::max(zoom_h, zoom_v);

  if (m_zoom > 1.0) {
   #if 0
    m_interpolation = cv::INTER_CUBIC; // (slow but better)
   #else
    m_interpolation = cv::INTER_LINEAR; // (faster but ok)
   #endif
  } else {
    m_interpolation = cv::INTER_AREA; // (better for shrinking)
  }

  return (m_enabled = true);
}


bool cv_cropper::apply(cv::Mat& image) {
  m_enabled = false; // turn off as it is applied

  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  int roi_width = 0;
  int roi_height = 0;
  cv::Mat roi;

  roi_width = m_roi_size.first;
  roi_height = m_roi_size.second;
  cv::Mat scaled_image;
  cv::resize(image, scaled_image, cv::Size(), m_zoom, m_zoom, m_interpolation);
  cv::Rect crop((scaled_image.cols - roi_width + 1) / 2,
                (scaled_image.rows - roi_height + 1) / 2,
                roi_width,
                roi_height);
  roi = scaled_image(crop);

  int crop_x_start=0, crop_y_start=0;

  // Get random crop of image
  if(m_rand_crop) {
    crop_x_start = fast_rand_int(get_fast_generator(), roi_width - m_width + 1);
    crop_y_start = fast_rand_int(get_fast_generator(), roi_height- m_height + 1);
  } else {
    crop_x_start = (roi_width - m_width + 1) / 2;
    crop_y_start = (roi_height - m_height + 1) / 2;
  }

  image = roi(cv::Rect(crop_x_start, crop_y_start, m_width, m_height));

  return true;
}


std::ostream& cv_cropper::print(std::ostream& os) const {
  os << "cv_cropper:" << std::endl
     << " - m_width: " << m_width << std::endl
     << " - m_height: " << m_height << std::endl
     << " - m_roi_size: " << m_roi_size.first << " " << m_roi_size.second << std::endl
     << " - m_zoom: " << m_zoom << std::endl;
  return os;
}

} // end of namespace lbann
#endif // __LIB_OPENCV

