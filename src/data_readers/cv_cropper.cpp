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
// cv_cropper .cpp .hpp - functions to crop images
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/cv_cropper.hpp"
#include "lbann/utils/mild_exception.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/exception.hpp"
#include <algorithm>
#include <ostream>

#ifdef LBANN_HAS_OPENCV
namespace lbann {

const int cv_cropper::m_interpolation_choices[3] = {cv::INTER_LINEAR, cv::INTER_AREA, cv::INTER_LINEAR};

cv_cropper::cv_cropper()
  : cv_transform(), m_width(0u), m_height(0u),
    m_rand_crop(false), m_is_roi_set(false),
    m_roi_size(std::pair<int,int>(0,0)),
    m_zoom(1.0), m_interpolation(m_interpolation_choices[0]),
    m_adaptive_interpolation(false) {}


cv_cropper *cv_cropper::clone() const {
  return new cv_cropper(*this);
}

/// Make sure to clear the roi flag as well when clearing roi size
void cv_cropper::unset_roi() {
  m_is_roi_set = false;
  m_roi_size = std::pair<int, int>(0, 0);
}

void cv_cropper::set(const unsigned int width, const unsigned int height,
                     const bool random_crop,
                     const std::pair<int, int>& roi_sz,
                     const bool adaptive_interpolation) {
  reset();
  m_width = width;
  m_height = height;
  m_rand_crop = random_crop;
  m_adaptive_interpolation = adaptive_interpolation;

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
}

void cv_cropper::reset() {
  m_enabled = false;
  m_zoom = 1.0;
  m_interpolation = m_interpolation_choices[0];
}

bool cv_cropper::determine_transform(const cv::Mat& image) {
  m_enabled = false; //sufficient for now in place of reset();

  _LBANN_SILENT_EXCEPTION(image.empty(), "", false)

  double zoom_h = 1.0;
  double zoom_v = 1.0;
  if (m_is_roi_set) {
    zoom_h = image.cols / static_cast<double>(m_roi_size.first);
    zoom_v = image.rows / static_cast<double>(m_roi_size.second);
  }

  m_zoom = std::min(zoom_h, zoom_v);

  if (m_zoom > 1.0) { // rescales the image by the factor of 1/m_zoom (shrink)
    m_interpolation =  m_interpolation_choices[static_cast<int>(m_adaptive_interpolation)];
  } else {
    m_interpolation =  m_interpolation_choices[static_cast<int>(m_adaptive_interpolation) << 1];
  }

  return (m_enabled = true);
}

/**
 * Method 1:
 *  a. Rescale the raw image, I, such that one dimension matches the corresponding
 *     dimension of the specified rectangular area, R, while trying to maintain the
 *     size as closely as possible to that of the raw image without altering the
 *     aspect ratio.
 *  b. Crop off the excess area of the resized image, which goes beyond the
 *     specified R aligned at the center of the image.
 *  c. Crop out an area of the specified size, C, at the center of R or at a random
 *     position within R.
 *
 * Method 2:
 *  Instead of rescaling-crop-crop as in method 1,
 *  a. Compute the projection of the final crop area, C', on the raw image I without
 *     actually rescaling the image. This still requires to compute the scaling factor
 *     for image resizing.
 *     However, instead of applying it to the raw image, apply the inverse to project
 *     the crop C onto the raw image I. This does not change any actual pixel.
 *  b. Crop the projected area C'
 *  c. Rescale C' to C. This deals with a smaller number of pixels than method 1 for
 *     resizing, only those that remain.
 *
 *  We rely on Method 2 here.
 */
bool cv_cropper::apply(cv::Mat& image) {
  m_enabled = false; // turn off as it is applied

  //_LBANN_SILENT_EXCEPTION(image.empty(), "", false); // redundant

  const double zoomed_roi_width = m_roi_size.first * m_zoom;
  const double zoomed_roi_height = m_roi_size.second * m_zoom;
  const double zoomed_width = m_width * m_zoom;
  const double zoomed_height = m_height * m_zoom;

  int crop_x_start = 0;
  int crop_y_start = 0;

  // Get random crop of image
  if(m_rand_crop) {
    const int rnd_dw = fast_rand_int(get_fast_io_generator(), static_cast<int>(2*(zoomed_roi_width - zoomed_width)) + 1);
    const int rnd_dh = fast_rand_int(get_fast_io_generator(), static_cast<int>(2*(zoomed_roi_height - zoomed_height)) + 1);
    crop_x_start = static_cast<int>(image.cols - zoomed_roi_width + rnd_dw + 1) / 2;
    crop_y_start = static_cast<int>(image.rows - zoomed_roi_height + rnd_dh + 1) / 2;
  } else {
    crop_x_start = static_cast<int>(image.cols - zoomed_width + 1) / 2;
    crop_y_start = static_cast<int>(image.rows - zoomed_height + 1) / 2;
  }

  cv::Mat zoomed_crop = image(cv::Rect(crop_x_start, crop_y_start, zoomed_width, zoomed_height));
  cv::Mat crop;
  cv::resize(zoomed_crop, crop, cv::Size(m_width,m_height), 0, 0, m_interpolation);
  image = crop;

  return true;
}

std::string cv_cropper::get_description() const {
  std::stringstream os;
  os << get_type() + ":" << std::endl
     << " - crop size: " << m_width  << "x" << m_height << std::endl
     << " - resized size: " << m_roi_size.first << "x" << m_roi_size.second << std::endl
     << " - random crop: " << m_rand_crop << std::endl
     << " - adaptive interpolation: " << m_adaptive_interpolation << std::endl;
  return os.str();
}

std::ostream& cv_cropper::print(std::ostream& os) const {
  os << get_description()
     << " - zoom: 1/" << m_zoom << std::endl
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
