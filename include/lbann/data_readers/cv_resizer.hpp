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
// lbann_cv_resizer .cpp .hpp - Functions to resize images
//                              in opencv format
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_RESIZER_HPP
#define LBANN_CV_RESIZER_HPP

#include "lbann/data_readers/patchworks/patchworks_opencv.hpp"
#include "lbann/data_readers/cv_transform.hpp"
#include <utility>
#include <ostream>

#ifdef __LIB_OPENCV
namespace lbann {

/**
 * If a region of interest is defined, use only that region centered at the given
 * coordinate in a given image. Otherwise, use the entire image.
 * Then, Zoom in/out the image to a minimal size that can cover the entire region
 * of a desired rectangular size and then crop to match the size.
 * During this one can choose the coordinate of the desired center of the image,
 * and whether to zoom equally to both the horizontal and the vertical directions
 * or not.
 */
class cv_resizer : public cv_transform {
 protected:
  unsigned int m_width; ///< desired width of an image
  unsigned int m_height; ///< desired height of an image
  /// whether to apply the same zoom ratio to width and height
  bool m_uniform_zoom;
  /// The factional coordinate of the desired center of an image
  std::pair<float, float> m_center;
  /// The size of the region of interest to crop initially
  std::pair<int, int> m_roi_sz;

  cv::Rect m_roi; ///< The region of interest set
  cv::Point_<float> m_center_roi;
  float m_hzoom; ///< horizontal zoom
  float m_vzoom; ///< vertical zoom
  int m_interpolation; ///< channel value interpolation method

  /**
   * Check if the center coordinate is valid.
   * If the coordinate has not been initialized, i.e., (0.0,0.0),
   * it is set to the center, i.e., (0.5, 0.5).
   */
  virtual bool check_center(const cv::Mat& image);
  /// Determine the initial region to crop to and set to use
  virtual bool determine_initial_roi(const cv::Mat& image);
  /// Determine the ratio to zoom by, to minimally cover the desired area
  virtual bool determine_zoom_ratio(const cv::Mat& image);

 public:
  cv_resizer();
  cv_resizer(const cv_resizer& rhs) = default;
  cv_resizer& operator=(const cv_resizer& rhs) = default;
  virtual cv_resizer *clone() const;
  virtual ~cv_resizer() {}

  /**
   * Set the parameters all at once
   * @param width  desired width of the image
   * @param height desired height of the image
   * @param uzoom  whether to use a uniform zoom ratio for both horizontal and vertical scalings
   * @param center desired center of the image in terms of the fractional position
   */
  virtual void set(const unsigned int width, const unsigned int height, const bool uzoom = true,
                   const std::pair<float, float>& center = std::make_pair(0.5f, 0.5f),
                   const std::pair<int, int>& roi = std::make_pair(0,0));

  /// Reset all the parameters to the default values
  virtual void reset();

  /**
   * Construct an affine transformation matrix based on the options and random
   * numbers. If successful, the tranform is enabled.If not, it is disabled.
   * @return false if not enabled or unsuccessful.
   */
  virtual bool determine_transform(const cv::Mat& image);

  /**
   * Apply the transformation determined.
   * As this method is executed, the transform becomes deactivated.
   * @return false if not successful.
   */
  virtual bool apply(cv::Mat& image);

  /// The augmentation is nonreversible.
  bool determine_inverse_transform() {
    return false;
  }

  virtual std::ostream& print(std::ostream& os) const;
};

} // end of namespace lbann
#endif // __LIB_OPENCV

#endif // LBANN_CV_RESIZER_HPP
