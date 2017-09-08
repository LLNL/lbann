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
// image_preprocessor.cpp - Preprocessing utilities for image inputs
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/image_preprocessor.hpp"
#include "lbann/data_readers/image_utils.hpp"
#include "lbann/data_readers/patchworks/patchworks.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/statistics.hpp"
#include "lbann/utils/exception.hpp"

namespace {
const float pi = std::acos(-1);
};

namespace lbann {

lbann_image_preprocessor::lbann_image_preprocessor() :
  m_horizontal_flip(false),
  m_vertical_flip(false),
  m_rotation_range(0.0f),
  m_horizontal_shift(0.0f),
  m_vertical_shift(0.0f),
  m_shear_range(0.0f),
  m_mean_subtraction(false),
  m_unit_variance(false),
  m_scale(true),  // We always did scaling by default.
  m_z_score(false),
  m_noise_factor(0.0f) {
}

void lbann_image_preprocessor::augment(Mat& pixels, unsigned imheight,
                                       unsigned imwidth,
                                       unsigned num_channels) {
  bool do_transform = m_horizontal_flip || m_vertical_flip ||
                      m_rotation_range || m_horizontal_shift || m_vertical_shift ||
                      m_shear_range;
  if (do_transform) {
    cv::Mat sqpixels = cv_pixels(pixels, imheight, imwidth, num_channels);
    rng_gen& gen = get_generator();
    std::uniform_int_distribution<int> bool_dist(0, 1);
    // Flips.
    bool horiz_flip = bool_dist(gen) && m_horizontal_flip;
    bool vert_flip = bool_dist(gen) && m_vertical_flip;
    if (horiz_flip || vert_flip) {
      if (horiz_flip && !vert_flip) {
        flip(sqpixels, 1);
      } else if (!horiz_flip && vert_flip) {
        flip(sqpixels, 0);
      } else {
        flip(sqpixels, -1);
      }
    }
    // Translations.
    float x_trans = 0.0f;
    float y_trans = 0.0f;
    if (m_horizontal_shift) {
      std::uniform_real_distribution<float> dist(-m_horizontal_shift,
          m_horizontal_shift);
      x_trans = dist(gen) * imwidth;
    }
    if (m_vertical_shift) {
      std::uniform_real_distribution<float> dist(-m_vertical_shift,
          m_vertical_shift);
      y_trans = dist(gen) * imheight;
    }
    Mat trans_mat;
    El::Diagonal(trans_mat, std::vector<DataType>({1.0f, 1.0f, 1.0f}));
    trans_mat(0, 2) = x_trans;
    trans_mat(1, 2) = y_trans;
    // Shearing.
    float shear = 0.0f;
    if (m_shear_range) {
      std::uniform_real_distribution<float> dist(-m_shear_range,
          m_shear_range);
      shear = dist(gen);
    }
    Mat shear_mat;
    El::Zeros(shear_mat, 3, 3);
    shear_mat(0, 0) = 1.0f;
    shear_mat(2, 2) = 1.0f;
    shear_mat(0, 1) = -std::sin(shear);
    shear_mat(1, 1) = std::cos(shear);
    // Rotation.
    float rotate = 0.0f;
    if (m_rotation_range) {
      std::uniform_real_distribution<float> dist(-m_rotation_range,
          m_rotation_range);
      rotate = ::pi / 180.0f * dist(gen);
    }
    Mat rot_mat;
    El::Zeros(rot_mat, 3, 3);
    rot_mat(2, 2) = 1.0f;
    rot_mat(0, 0) = std::cos(rotate);
    rot_mat(0, 1) = -std::sin(rotate);
    rot_mat(1, 0) = std::sin(rotate);
    rot_mat(1, 1) = std::cos(rotate);
    // Compute the final transformation.
    Mat affine_mat_tmp(3, 3);
    Mat affine_mat(3, 3);
    El::Gemm(NORMAL, NORMAL, (DataType) 1.0, trans_mat, shear_mat,
             (DataType) 0.0, affine_mat_tmp);
    El::Gemm(NORMAL, NORMAL, (DataType) 1.0, affine_mat_tmp, rot_mat,
             (DataType) 0.0, affine_mat);
    affine_trans(sqpixels, affine_mat);
    col_pixels(sqpixels, pixels, num_channels);
  }
}

void lbann_image_preprocessor::normalize(Mat& pixels, unsigned num_channels) {
  if (m_z_score || (m_mean_subtraction && m_unit_variance)) {
    z_score(pixels, num_channels);
  } else {
    if (m_scale) {
      unit_scale(pixels, num_channels);
    }
    if (m_mean_subtraction) {
      mean_subtraction(pixels, num_channels);
    }
    if (m_unit_variance) {
      unit_variance(pixels, num_channels);
    }
  }
}

void lbann_image_preprocessor::mean_subtraction(Mat& pixels,
    unsigned num_channels) {
  const unsigned height = pixels.Height();
  const unsigned height_per_channel = height / num_channels;
  for (unsigned channel = 0; channel < num_channels; ++channel) {
    const unsigned channel_start = channel*height_per_channel;
    const unsigned channel_end = (channel+1)*height_per_channel;
    Mat pixels_channel = El::View(pixels, El::IR(channel_start, channel_end), El::ALL);
    DataType mean, stdev;
    entrywise_mean_and_stdev(pixels_channel, mean, stdev);
    for (unsigned i = 0; i < height_per_channel; ++i) {
      DataType& pixels_entry = pixels_channel(i, 0);
      pixels_entry -= mean;
    }
  }
}

void lbann_image_preprocessor::unit_variance(
  Mat& pixels, unsigned num_channels) {

  // Get image parameters
  const unsigned height = pixels.Height();
  const unsigned height_per_channel = height / num_channels;

  // Scale each channel separately
  for (unsigned channel = 0; channel < num_channels; ++channel) {
    const unsigned channel_start = channel*height_per_channel;
    const unsigned channel_end = (channel+1)*height_per_channel;
    Mat pixels_channel = El::View(pixels, El::IR(channel_start, channel_end), El::ALL);
    DataType mean, stdev;
    entrywise_mean_and_stdev(pixels_channel, mean, stdev);
    if(stdev > DataType(1e-7)*Abs(mean)) {
      const DataType inv_stdev = 1 / stdev;
      for (unsigned i = 0; i < height_per_channel; ++i) {
        DataType& pixels_entry = pixels_channel(i, 0);
        pixels_entry = (pixels_entry - mean) * inv_stdev + mean;
      }
    }
  }

}

void lbann_image_preprocessor::unit_scale(Mat& pixels,
    unsigned num_channels) {
  // Pixels are in range [0, 255], normalize using that.
  // Channels are not relevant here.
  pixels *= DataType(1) / 255;
}


void lbann_image_preprocessor::pixel_noise(Mat& pixels) 
{
  if(m_noise_factor){
    Mat X_noise;
    El::Gaussian(X_noise, pixels.Height(), pixels.Width(), DataType(0), DataType(1));
    El::Axpy(m_noise_factor,X_noise,pixels);
    //@todo - clip to min and max of input entry
    auto clip = [](const DataType& z) { 
         return std::max(DataType(0), std::min(z,DataType(1)));
    };
    EntrywiseMap(pixels, El::MakeFunction(clip));
  } 
}

void lbann_image_preprocessor::z_score(Mat& pixels,
                                       unsigned num_channels) {

  // Get image parameters
  const unsigned height = pixels.Height();
  const unsigned height_per_channel = height / num_channels;

  // Shift and scale each channel separately
  for (unsigned channel = 0; channel < num_channels; ++channel) {
    const unsigned channel_start = channel*height_per_channel;
    const unsigned channel_end = (channel+1)*height_per_channel;
    Mat pixels_channel = El::View(pixels, El::IR(channel_start, channel_end), El::ALL);
    DataType mean, stdev;
    entrywise_mean_and_stdev(pixels_channel, mean, stdev);
    if(stdev > DataType(1e-7)*Abs(mean)) {
      const DataType inv_stdev = 1 / stdev;
      for (unsigned i = 0; i < height_per_channel; ++i) {
        DataType& pixels_entry = pixels_channel(i, 0);
        pixels_entry = (pixels_entry - mean) * inv_stdev;
      }
    } else {
      Zero(pixels_channel);
    }
  }

}

cv::Mat lbann_image_preprocessor::cv_pixels(const Mat& pixels,
    unsigned imheight,
    unsigned imwidth,
    unsigned num_channels) {
  if (num_channels == 1) {
    cv::Mat m(imheight, imwidth, CV_32FC1);
    for (unsigned y = 0; y < imheight; ++y) {
      for (unsigned x = 0; x < imwidth; ++x) {
        m.at<float>(y, x) = pixels(y * imwidth + x, 0);
      }
    }
    return m;
  } else if (num_channels == 3) {
    cv::Mat m(imheight, imwidth, CV_32FC3);
    for (unsigned y = 0; y < imheight; ++y) {
      for (unsigned x = 0; x < imwidth; ++x) {
        cv::Vec3f pixel;
        unsigned offset = y * imwidth + x;
        pixel[0] = pixels(offset, 0);
        pixel[1] = pixels(offset + imheight*imwidth, 0);
        pixel[2] = pixels(offset + 2*imheight*imwidth, 0);
        m.at<cv::Vec3f>(y, x) = pixel;
      }
    }
    return m;
  } else {
    throw lbann_exception(std::string{} + __FILE__ + " " +
                          std::to_string(__LINE__) +
                          "Only support 1 and 3 channels");
  }
}

void lbann_image_preprocessor::col_pixels(const cv::Mat& sqpixels, Mat& pixels,
    unsigned num_channels) {
  unsigned imheight = sqpixels.rows;
  unsigned imwidth = sqpixels.cols;
  if (num_channels == 1) {
    for (unsigned y = 0; y < imheight; ++y) {
      for (unsigned x = 0; x < imwidth; ++x) {
        pixels(y * imwidth + x, 0) = sqpixels.at<float>(y, x);
      }
    }
  } else if (num_channels == 3) {
    for (unsigned y = 0; y < imheight; ++y) {
      for (unsigned x = 0; x < imwidth; ++x) {
        cv::Vec3f pixel = sqpixels.at<cv::Vec3f>(y, x);
        unsigned offset = y * imwidth + x;
        pixels(offset, 0) = pixel[0];
        pixels(offset + imheight*imwidth, 0) = pixel[1];
        pixels(offset + 2*imheight*imwidth, 0) = pixel[2];
      }
    }
  } else {
    throw lbann_exception(std::string{} + __FILE__ + " " +
                          std::to_string(__LINE__) +
                          "Only support 1 and 3 channels");
  }
}

void lbann_image_preprocessor::flip(cv::Mat& sqpixels, int flip_flag) {
  // In/out must be different.
  cv::Mat sqpixels_copy = sqpixels.clone();
  cv::flip(sqpixels_copy, sqpixels, flip_flag);
}

void lbann_image_preprocessor::affine_trans(cv::Mat& sqpixels,
    const Mat& trans) {
  cv::Mat sqpixels_copy = sqpixels.clone();
  // Construct the OpenCV transformation matrix.
  cv::Mat cv_trans(2, 3, CV_32FC1);
  cv_trans.at<float>(0, 0) = trans(0, 0);
  cv_trans.at<float>(0, 1) = trans(0, 1);
  cv_trans.at<float>(0, 2) = trans(0, 2);
  cv_trans.at<float>(1, 0) = trans(1, 0);
  cv_trans.at<float>(1, 1) = trans(1, 1);
  cv_trans.at<float>(1, 2) = trans(1, 2);
  cv::warpAffine(sqpixels_copy, sqpixels, cv_trans, sqpixels.size(),
                 cv::INTER_LINEAR, cv::BORDER_REPLICATE);
}

void lbann_image_preprocessor::internal_save_image(
  Mat& pixels, const std::string filename, unsigned imheight, unsigned imwidth,
  unsigned num_channels, bool do_scale) {
  cv::Mat sqpixels = cv_pixels(pixels, imheight, imwidth, num_channels);
  cv::Mat converted_pixels;
  int dst_type = 0;
  if (num_channels == 1) {
    dst_type = CV_8UC1;
  } else if (num_channels == 3) {
    dst_type = CV_8UC3;
  }  // cv_pixels ensures no other case happens.
  sqpixels.convertTo(converted_pixels, dst_type, do_scale ? 255.0f : 1.0f);
  cv::imwrite(filename, converted_pixels);
}

}  // namespace lbann
