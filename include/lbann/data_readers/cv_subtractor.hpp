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

#ifndef LBANN_CV_SUBTRACTOR_HPP
#define LBANN_CV_SUBTRACTOR_HPP

#include "cv_transform.hpp"
#include "lbann/base.hpp"

#ifdef LBANN_HAS_OPENCV
namespace lbann {

/**
 * Subtract channel values of an image from the corresponding values of another.
 * The former is likely to carry pre-computed mean data per pixel and per channel.
 * The latter is an input image. Both image needs to have the same size and the
 * same number of channels. The subtracted result is represented in the scale
 * between 0 and 1 (both inclusive).
 * In the common current use case, a colorizer comes before a subtractor which is
 * followed by a random cropper. In this scenario, the input images must be resized
 * in advance to match the size of the mean image.
 * In another scenario, where the random cropping is not used but resizing is done
 * on-line, the subtractor can come after cropper without requiring the input images
 * to be resized in advance.
 * Alternatively, even a simpler approach is to use a mean image with uniform pixels.
 * In this way, it does not need to know the size of input images, and is not impacted
 * by random cropping or flipping augmentation.
 */
class cv_subtractor : public cv_transform {
 protected:
  // --- configuration variables ---
  /**
   * The image to subtract from an input image in the pixel-wise fashion.
   * It has channel values of a floating point type, in the scale from 0 to 1.
   * An input image will be mapped into the scale before subtraction by linearly
   * mapping the smallest representative value to 0 and the largest representative
   * value to 1.
   */
  cv::Mat m_img_to_sub;

  /**
   * The image to divide an input image in the pixel-wise fashion.
   * It has channel values of a floating point type, in the scale from 0 to 1.
   * An input image will be mapped into the scale before division.
   */
  cv::Mat m_img_to_div;

  /** uniform mean per channel used for channel-wise mean-subtraction.
   *  This is used to construct the m_img_to_sub when the size of the image is known.
   */
  std::vector<lbann::DataType> m_channel_mean;

  /** uniform standard deviation per channel used for channel-wise z-score (division).
   *  This is used to construct the m_img_to_div when the size of the image is known.
   */
  std::vector<lbann::DataType> m_channel_stddev;

  // --- state variables ---
  bool m_applied; ///< has been subtracted

 public:
  cv_subtractor() : cv_transform(), m_applied(false) {}
  cv_subtractor(const cv_subtractor& rhs);
  cv_subtractor& operator=(const cv_subtractor& rhs);
  cv_subtractor *clone() const override;

  ~cv_subtractor() override {}

  static cv::Mat read_binary_image_file(const std::string filename);

  /// Load and set the image to subtract from every input image.
  void set_mean(const std::string name_of_img, const int depth_code = cv_image_type<lbann::DataType>::T());

  /**
   * Set the mean fixed per channel for mean-subtracting each input image.
   * This supports an alternative method for mean subtraction given that the
   * mean per channel is uniform.
   */
  void set_mean(const std::vector<lbann::DataType> channel_mean);

  /**
   * Set the dataset-wise mean image to subtract from each input image.
   * The image represents the pre-computed pixel-wise mean of the dataset.
   * In case that this image is not in a floating point type, it is converted to
   * one with the depth specified by depth_code.
   */
  void set_mean(const cv::Mat& img, const int depth_code = cv_image_type<lbann::DataType>::T());

  /// Load and set the image to normalize the pixels of every input image.
  void set_stddev(const std::string name_of_img, const int depth_code = cv_image_type<lbann::DataType>::T());

  /**
   * Set the dataset-wise standard deviation fixed per channel for normalizing
   * each input image.
   * This supports an alternative method for normalizing with stddev given that
   * it is uniform per channel.
   */
  void set_stddev(const std::vector<lbann::DataType> channel_stddev);

  /**
   * Set the dataset-wise standard deviation to normalize each input image.
   * In case that this image is not in a floating point type, it is converted to
   * one with the depth specified by depth_code.
   */
  void set_stddev(const cv::Mat& img, const int depth_code = cv_image_type<lbann::DataType>::T());

  void reset() override {
    m_enabled = false;
    m_applied = false;
  }

  /**
   * If a given image is in grayscale, the tranform is enabled, and not otherwise.
   * @return false if not enabled or unsuccessful.
   */
  bool determine_transform(const cv::Mat& image) override;

  /// convert back to color image if it used to be a grayscale image
  bool determine_inverse_transform() override;

  /**
   * Apply color conversion if enabled.
   * As it is applied, the transform becomes deactivated.
   * @return false if not successful.
   */
  bool apply(cv::Mat& image) override;

  /// true if both sub and div are channel-wise
  bool check_if_channel_wise() const;

  std::string get_type() const override { return "subtractor"; }
  std::string get_description() const override;
  std::ostream& print(std::ostream& os) const override;

 protected:
  /// Construct an image of the unform channel values using the channel-wise mean.
  bool create_img_to_sub(int width, int height, int n_channels);
  /// Construct an image of the unform channel values using the channel-wise stddev.
  bool create_img_to_div(int width, int height, int n_channels);
};

} // end of namespace lbann
#endif // LBANN_HAS_OPENCV

#endif // LBANN_CV_SUBTRACTOR_HPP
