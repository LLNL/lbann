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
// patchworks_ROI.cpp - LBANN PATCHWORKS ROI (region-of-interest) implementation
////////////////////////////////////////////////////////////////////////////////

/**
 * LBANN PATCHWORKS ROI implementation
 *  - Region of interest descriptor
 */

#include "lbann/data_readers/patchworks/patchworks_ROI.hpp"

#ifdef LBANN_HAS_OPENCV
#include <iostream>

namespace lbann {
namespace patchworks {

const int ROI::undefined_coordinate = -1;

/// Reset to the initial condition indicating to cover the whole image
void ROI::init() {
  m_left = undefined_coordinate;
  m_top = undefined_coordinate;
  m_right = undefined_coordinate;
  m_bottom = undefined_coordinate;
}

bool ROI::is_undefined() const {
  return ((m_left == undefined_coordinate)  ||
          (m_top == undefined_coordinate)   ||
          (m_right == undefined_coordinate) ||
          (m_bottom == undefined_coordinate)); // default

}

/// Sanity check on a set of two coordinates that defines a region of interest
bool ROI::is_valid() const {
  return (!is_undefined() && (m_left < m_right) && (m_top < m_bottom));
}

/**
 * Check how the region of interest overlaps with the image, and shrink it to
 * preceisely match the image boundary in case that it is out of boundary.
 */
bool ROI::set_overlapping_region(const cv::Mat& img) {
  if (!is_valid() || (img.data == nullptr)) {
    return false;
  }
  if (m_left < 0) {
    m_left = 0;
  }
  if (m_top < 0) {
    m_top = 0;
  }
  if (m_right > img.cols) {
    m_right = img.cols;
  }
  if (m_bottom > img.rows) {
    m_bottom = img.rows;
  }
  if (m_right == undefined_coordinate) {
    m_right = img.cols;
  }
  if (m_bottom == undefined_coordinate) {
    m_bottom = img.rows;
  }

  return true;
}

bool ROI::is_whole_image(const cv::Mat& img) {
  const bool ok = set_overlapping_region(img);
  return ok &&
         ((m_left == 0) &&
          (m_top == 0) &&
          (m_right == img.cols) &&
          (m_bottom == img.rows));
}

bool ROI::set_by_corners(const int p0_x, const int p0_y, const int p1_x, const int p1_y) {
  m_left = p0_x;
  m_top = p0_y;
  m_right = p1_x;
  m_bottom = p1_y;

  return is_valid();
}

bool ROI::set_by_center(const int px, const int py, const unsigned int _width, const unsigned int _height) {
  m_left = px - (_width + _width%2)/2;
  m_right = px + (_width + _width%2)/2;
  m_top = py - (_height + _height%2)/2;
  m_bottom = py + (_height + _height%2)/2;

  return is_valid();
}

void ROI::move(const std::pair<int, int>  displ) {
  m_left   += displ.first;
  m_right  += displ.first;
  m_top    += displ.second;
  m_bottom += displ.second;
}

bool ROI::operator==(const ROI& rarea) const {
  return ((rarea.m_left == m_left) && (rarea.m_top == m_top) &&
          (m_right == rarea.m_right) && (m_bottom == rarea.m_bottom));
}

bool ROI::operator!=(const ROI& rarea) const {
  return !(*this == rarea);
}

bool ROI::operator<(const ROI& rarea) const {
  return ((*this <= rarea) && !(*this == rarea));
}

bool ROI::operator>(const ROI& rarea) const {
  return ((*this >= rarea) && !(*this == rarea));
}

/// Stream out the content of the region of interest
std::ostream& operator<<(std::ostream& os, const ROI& roi) {
  return roi.Print(os);
}

} // end of namespace patchworks
} // end of namespace lbann
#endif // LBANN_HAS_OPENCV
