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
// patchworks_ROI.hpp - LBANN PATCHWORKS ROI (region-of-interest) header
////////////////////////////////////////////////////////////////////////////////

/**
 * LBANN PATCHWORKS ROI header
 *  - Region of interest descriptor
 */

#ifdef __LIB_OPENCV
#ifndef _PATCHWORKS_ROI_H_INCLUDED_
#define _PATCHWORKS_ROI_H_INCLUDED_

#include <ostream>
#include <sstream>
#include "patchworks_common.hpp"

namespace lbann {
namespace patchworks {

/**
 * Regions of interest descriptor.
 * Contains a pair of coordinates that defines a rectangular region of interest
 */
class ROI {
 public:
  /// An internal value to represent an uninitialized coordinate value
  static const int undefined_coordinate;

  int m_left; ///< The left-most pixel position of the region
  int m_top; ///< The top-most pixel position of the region
  int m_right; ///< The right-most pixel position of the region
  int m_bottom; ///< The bottom-most pixel position of the region

  ROI(void) ///< The default constructor
    : m_left(undefined_coordinate), m_top(undefined_coordinate),
      m_right(undefined_coordinate), m_bottom(undefined_coordinate) {}

  void init(void); ///< Reset the structure with undefined coordinate values
  bool is_undefined(void) const; ///< Tell if the structure has not been initialized
  bool is_valid(void) const; ///< Check if the region is valid
  bool set_overlapping_region(const cv::Mat& img);
  /// Check if the region of interest covers the whole image
  bool is_whole_image(const cv::Mat& img);

  /// Set a region by the coordinates
  bool set_by_corners(const int p0_x, const int p0_y,
                      const int p1_x, const int p1_y);
  /// Set a region by the center and its size
  bool set_by_center(const int px, const int py,
                     const unsigned int _width, const unsigned int _height);

  /// move the region horizontally by dx and vertically by dy
  void move(const std::pair<int, int> displacement);

  /// Returns the left position of the region
  int left(void) const {
    return m_left;
  }
  /// Returns the top poisition of the region
  int top(void) const {
    return m_top;
  }
  /// Returns the right position of the region
  int right(void) const {
    return m_right;
  }
  /// Returns the bottom position of the region
  int bottom(void) const {
    return m_bottom;
  }

  /// Returns a cv::Rect equivalent
  cv::Rect rect(void) const {
    return cv::Rect(m_left, m_top, m_right-m_left, m_bottom-m_top);
  }
  /// Returns the width of the rectangular region
  int width(void) const {
    return (m_right - m_left);
  }
  /// Returns the height of the rectangular region
  int height(void) const {
    return (m_bottom - m_top);
  }
  /// Returns the area of the rectangular region
  int area(void) const {
    return width()*height();
  }
  /// Returns the size of the area (width, hegiht)

  std::ostream& Print(std::ostream& os) const { ///< Print out the content
    return os << '(' << m_left << ", " << m_top << ") ("
           <<  m_right << ", " << m_bottom << ')';
  }

  /// Check if this ROI is exactly the same as the given rectangular area
  bool operator==(const ROI& rarea) const;
  /// Check if this ROI is not exactly the same as the given rectangular area
  bool operator!=(const ROI& rarea) const;
  /// Check if the given rectangular region contains this ROI but is not the same
  bool operator<(const ROI& rarea) const;
  /// Check if the given rectangular region contains this ROI
  bool operator<=(const ROI& rarea) const;
  /// Check if this ROI  contains the given rectangular region but is not the same
  bool operator>(const ROI& rarea) const;
  /// Check if this ROI  contains the given rectangular region
  bool operator>=(const ROI& rarea) const;
};

inline bool ROI::operator<=(const ROI& rarea) const {
  return (((rarea.m_left <= m_left) && (rarea.m_top <= m_top)) &&
          ((m_right <= rarea.m_right) && (m_bottom <= rarea.m_bottom)) &&
          is_valid());
}

inline bool ROI::operator>=(const ROI& rarea) const {
  return (((m_left <= rarea.m_left) && (m_top <= rarea.m_top)) &&
          ((rarea.m_right <= m_right) && (rarea.m_bottom <= m_bottom)) &&
          rarea.is_valid());
}

std::ostream& operator<<(std::ostream& os, const ROI& roi);

} // end of namespace patchworks
} // end of namespace lbann
#endif // _PATCHWORKS_ROI_H_INCLUDED_
#endif // __LIB_OPENCV
