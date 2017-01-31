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
// lbann_image_preprocessor.cpp - Preprocessing utilities for image inputs
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_image_preprocessor.hpp"

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
  m_z_score(false) {
}

void lbann_image_preprocessor::preprocess(Mat& pixels,
                                          unsigned num_channels) {
  if (m_z_score) {
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

void lbann_image_preprocessor::mean_subtraction(
  Mat& pixels, unsigned num_channels) {
  const El::Int height = pixels.Height();
  for (unsigned channel = 0; channel < num_channels; ++channel) {
    // Compute the mean.
    DataType mean = 0.0f;
    for (unsigned i = channel; i < height; i += num_channels) {
      mean += pixels(i, 0);
    }
    mean /= height / num_channels;
    for (unsigned i = channel; i < height; i += num_channels) {
      pixels(i, 0) -= mean;
    }
  }
}

void lbann_image_preprocessor::unit_variance(
  Mat& pixels, unsigned num_channels) {
  const El::Int height = pixels.Height();
  for (unsigned channel = 0; channel < num_channels; ++channel) {
    // Compute the mean.
    DataType mean = 0.0f;
    for (unsigned i = channel; i < height; i += num_channels) {
      mean += pixels(i, 0);
    }
    mean /= height / num_channels;
    // Compute the standard deviation.
    DataType std = 0.0f;
    for (unsigned i = channel; i < height; i += num_channels) {
      std += (pixels(i, 0) - mean) * (pixels(i, 0) - mean);
    }
    std /= height / num_channels;
    std = std::sqrt(std) + 1e-7;  // Avoid division by 0.
    for (unsigned i = channel; i < height; i += num_channels) {
      pixels(i, 0) /= std;
    }
  }
}

void lbann_image_preprocessor::unit_scale(Mat& pixels, unsigned num_channels) {
  // Pixels are in range [0, 255], normalize using that.
  // Channels are not relevant here.
  const El::Int height = pixels.Height();
  for (unsigned i = 0; i < height; ++i) {
    pixels(i, 0) /= 255.0f;
  }
}

void lbann_image_preprocessor::z_score(Mat& pixels, unsigned num_channels) {
  const El::Int height = pixels.Height();
  for (unsigned channel = 0; channel < num_channels; ++channel) {
    // Compute the mean.
    DataType mean = 0.0f;
    for (unsigned i = channel; i < height; i += num_channels) {
      mean += pixels(i, 0);
    }
    mean /= height / num_channels;
    // Compute the standard deviation.
    DataType std = 0.0f;
    for (unsigned i = channel; i < height; i += num_channels) {
      std += (pixels(i, 0) - mean) * (pixels(i, 0) - mean);
    }
    std /= height / num_channels;
    std = std::sqrt(std) + 1e-7;  // Avoid division by 0.
    // Z-score is (x - mean) / std.
    for (unsigned i = channel; i < height; i += num_channels) {
      pixels(i, 0) = (pixels(i, 0) - mean) / std;
    }
  }
}

}  // namespace lbann
