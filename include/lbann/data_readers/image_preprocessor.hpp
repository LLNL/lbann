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
// image_preprocessor.hpp - Preprocessing utilities for image inputs
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_IMAGE_PREPROCESSOR
#define LBANN_IMAGE_PREPROCESSOR

#include "lbann_config.hpp"

#ifdef LBANN_HAS_OPENCV
#include "lbann/data_readers/opencv.hpp"
#else
#error OpenCV required
#endif
#include "lbann/base.hpp"

namespace lbann {

/**
 * Support class for preprocessing image inputs.
 * Supports the following transforms:
 * - Random horizontal and vertical flips
 * - Random rotations
 * - Random horizontal and vertical shifts
 * - Random shearing
 * - Standardize to 0 mean
 * - Standardize to unit variance
 * - Scale to the range [0, 1]
 * - Normalize via z-score
 */
class lbann_image_preprocessor {
 public:
  lbann_image_preprocessor();
  lbann_image_preprocessor(const lbann_image_preprocessor&) = default;
  lbann_image_preprocessor& operator=(
    const lbann_image_preprocessor&) = default;
  virtual ~lbann_image_preprocessor() {}

  /** Whether to do random horizontal flips. */
  void horizontal_flip(bool b) {
    m_horizontal_flip = b;
  }
  /** Whether to do random vertical flips. */
  void vertical_flip(bool b) {
    m_vertical_flip = b;
  }
  /** Do random rotations up to range degrees (0-180). */
  void rotation(float range) {
    m_rotation_range = range;
  }
  /** Do random horizontal shifts up to range (fraction of image width). */
  void horizontal_shift(float range) {
    m_horizontal_shift = range;
  }
  /** Do random vertical shifts up to range (fraction of image height). */
  void vertical_shift(float range) {
    m_vertical_shift = range;
  }
  /** Do random shears up to range (radians). */
  void shear_range(float range) {
    m_shear_range = range;
  }
  /** Whether to subtract the sample-wise mean. */
  void subtract_mean(bool b) {
    m_mean_subtraction = b;
  }
  /** Whether to normalize to unit variance, sample-wise. */
  void unit_variance(bool b) {
    m_unit_variance = b;
  }
  /** Whether to scale to [0, 1] (assumes max value is 255). */
  void scale(bool b) {
    m_scale = b;
  }
  /**
   * Whether to normalize by z-scores, sample-wise.
   * This and mean subtraction/unit variance are mutually exclusive.
   */
  void z_score(bool b) {
    m_z_score = b;
  }
  /** Disable all data augmentation. */
  void disable_augmentation() {
    horizontal_flip(false);
    vertical_flip(false);
    rotation(0.0f);
    horizontal_shift(0.0f);
    vertical_shift(0.0f);
    shear_range(0.0f);
  }

  /**
   * Add noise to data (disable by default)
   * noise_factor control the ammount of noise
   * to be set to a value above zero but less than 1 (say 0.5)
   * */
  void add_noise(float noise_factor=0.0f) {
    m_noise_factor = noise_factor;
   }

  /**
   * Preprocess pixels according to the currently-set augmentation transforms.
   * @param pixels The pixels to process as a column vector (num x 1 mat).
   * @param imheight Height of the image in pixels.
   * @param imwidth Width of the image in pixels.
   * @param num_channels The number of channels pixels has.
   */
  void augment(Mat& pixels, unsigned imheight, unsigned imwidth,
               unsigned num_channels);
  /**
   * Normalize poxels according to the currently-set transforms.
   * @param pixels The pixels to process as a column vector.
   * @param num_channels The number of channels pixels has.
   */
  void normalize(Mat& pixels, unsigned num_channels);

  /**
   * External interface to saving an image.
   * Classes that want to support this should use it to interface with
   * internal_save_image.
   * @param pixels The image to save (as a column vector).
   * @param filename The image filename (type inferred from extension).
   * @param do_scale Whether pixels has been scaled (default true).
   */
  virtual void save_image(Mat& pixels, const std::string filename,
                          bool do_scale = true) {}

 protected:
  /** Whether to do horizontal flips. */
  bool m_horizontal_flip;
  /** Whether to do vertical flips. */
  bool m_vertical_flip;
  /** Range in degrees for rotations (0-180). */
  float m_rotation_range;
  /** Range (fraction of total width) for horizontal shifts. */
  float m_horizontal_shift;
  /** Range (fraction of total height) for vertical shifts. */
  float m_vertical_shift;
  /** Shear angle (radians). */
  float m_shear_range;
  /** Whether to normalize to 0 mean. */
  bool m_mean_subtraction;
  /** Whether to normalize to unit variance. */
  bool m_unit_variance;
  /** Whether to scale to [0, 1]. */
  bool m_scale;
  /** Whether to normalize via z-score. */
  bool m_z_score;

  float m_noise_factor;

  void mean_subtraction(Mat& pixels, unsigned num_channels);
  void unit_variance(Mat& pixels, unsigned num_channels);
  void unit_scale(Mat& pixels, unsigned num_channels);
  void z_score(Mat& pixels, unsigned num_channels);

  void pixel_noise(Mat& pixels);

  /**
   * Convert a column vector of pixels to an OpenCV matrix.
   */
  cv::Mat cv_pixels(const Mat& pixels, unsigned imheight, unsigned imwidth,
                    unsigned num_channels);
  /** Undo cv_pixels. */
  void col_pixels(const cv::Mat& sqpixels, Mat& pixels, unsigned num_channels);

  /** @brief Flip sqpixels.
   *  @param sqpixels The image to flip
   *  @param flip_flag OpenCV flip flag: 0=vertical, 1=horizontal, -1=both.
   */
  void flip(cv::Mat& sqpixels, int flip_flag);
  /** Apply the affine transformation in 3x3 matrix trans. */
  void affine_trans(cv::Mat& sqpixels, const Mat& trans);

  /**
   * Save pixels to filename.
   */
  void internal_save_image(Mat& pixels, const std::string filename,
                           unsigned imheight, unsigned imwidth,
                           unsigned num_channels, bool do_scale);
};

}  // namespace lbann

#endif  // LBANN_IMAGE_PREPROCESSOR
