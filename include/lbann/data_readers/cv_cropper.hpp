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
// lbann_cv_cropper .cpp .hpp - Functions to crop images
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CV_CROPPER_HPP
#define LBANN_CV_CROPPER_HPP

#include "lbann/data_readers/patchworks/patchworks_opencv.hpp"
#include "lbann/data_readers/cv_transform.hpp"
#include <utility>
#include <ostream>

#ifdef __LIB_OPENCV
namespace lbann {

/**
 * If the size of a region of interest (ROI) is defined, use the area at the
 * center of a given image. Otherwise, use the entire image.
 * Zoom in/out the image if necessary to cover the ROI. Then, crop out an area
 * of the desired size from the region either randomly within the ROI or at the
 * center depending on the given specification.
 */
class cv_cropper : public cv_transform {
 protected:
  unsigned int m_width; ///< desired width of an image
  unsigned int m_height; ///< desired height of an image
  /// randomize the center position of the area of interest
  bool m_rand_crop;
  /// indicate if a specific ROI is set or supposed to use whole image
  bool m_is_roi_set;
  /// The size of the initial region of interest to crop from
  std::pair<int, int> m_roi_size;

  double m_zoom; ///< zoom factor to prepare the initial region for a given image
  int m_interpolation; ///< channel value interpolation method

  void unset_roi(void);

 public:
  cv_cropper();
  cv_cropper(const cv_cropper& rhs) = default;
  cv_cropper& operator=(const cv_cropper& rhs) = default;
  virtual cv_cropper *clone() const;
  virtual ~cv_cropper() {}

  /**
   * Set the parameters all at once
   * @param width  desired width of the crop
   * @param height desired height of the crop
   * @param random_crop whether to crop randomly from the initial region of interest or at the center
   * @param roi the size of the initial region of interest to crop from. Set (0,0) to use the full image.
   */
  virtual void set(const unsigned int width, const unsigned int height,
                   const bool random_crop = false,
                   const std::pair<int, int>& roi = std::make_pair(0,0));

  /// Reset all the parameters to the default values
  virtual void reset();

  /**
   * Construct transformation parameters based on the options and random
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

  /// Cropping is irreversible.
  bool determine_inverse_transform() {
    return false;
  }

  virtual std::ostream& print(std::ostream& os) const;
};

} // end of namespace lbann
#endif // __LIB_OPENCV

#endif // LBANN_CV_CROPPER_HPP
