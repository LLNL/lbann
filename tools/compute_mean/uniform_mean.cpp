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
////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <iostream>
#include <cstdint>
#include <limits>
#include "lbann/data_readers/opencv.hpp"
#include "lbann/data_readers/cv_subtractor.hpp"
#include "file_utils.hpp"

/**
 * Load an image from a file of the given name and return a cv:Mat object.
 * The image file can either be in a standard format handled by OpenCV,
 * or be in a proprietary format (a binary dump of a cv::Mat).
 */
cv::Mat read_image(const std::string image_file_name) {
  cv::Mat image;
  std::string ext = lbann::get_ext_name(image_file_name);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  if (ext == "bin") { // load the binary dump of cv::Mat data
    image = lbann::cv_subtractor::read_binary_image_file(image_file_name);
  } else { // let OpenCV handle
    image = cv::imread(image_file_name);
  }
  return image;
}


/// Return the Channel-wise mean (the mean pixel value) of an image
cv::Scalar reduce_image(const cv::Mat image) {
  cv::Mat row_reduced;
  cv::Mat col_reduced;
  cv::reduce(image, row_reduced, 0, cv::REDUCE_AVG, CV_64F);
  cv::reduce(row_reduced, col_reduced, 1, cv::REDUCE_AVG, CV_64F);

  cv::Scalar pixel(0);
  std::cout << "The mean pixel value :"; // (B G R) if color
  for (int i=0; i < col_reduced.channels(); ++i) {
    std::cout << ' ' << col_reduced.at<double>(i);
    pixel[i] = col_reduced.at<double>(i);
  }
  std::cout << std::endl;
  return pixel;
}


/**
 * Scale the mean pixel to the depth of the output image with the adjustment for rounding.
 * Convert to the unit scale if to_unit_scale is set when in_depth is of integral and
 * out_depth is of float.
 */
cv::Scalar scale_pixel(const cv::Scalar pixel_m, const int in_depth, const int out_depth) {
  const double alpha = lbann::get_depth_denormalizing_factor(out_depth)
                     * lbann::get_depth_normalizing_factor(in_depth);

  const double rounding_off_compensation = (lbann::is_float(out_depth)? 0.0 : 0.5);
  cv::Scalar pixel(0.0);
  for (int c = 0; c < cv::Scalar::channels; ++c) {
    pixel[c] = alpha*pixel_m[c] + rounding_off_compensation;
  }
  return pixel;
}


/**
 * Write the data of a cv::Mat object into a binary file, and encode the
 * specifics of the image into the filename.
 */
void write_bin_image(const int height, const int width,
                     const int cv_type, const cv::Scalar pixel) {
  cv::Mat mean_image(height, width, cv_type, pixel);
  std::string image_name = "mean_uniform-"
                           + std::to_string(mean_image.cols) + 'x'
                           + std::to_string(mean_image.rows) + 'x'
                           + std::to_string(mean_image.channels()) + '-'
                           + std::to_string(mean_image.depth()) + ".bin";

  const size_t sz = mean_image.rows * mean_image.cols
                  * mean_image.channels() * CV_ELEM_SIZE(mean_image.depth());

  tools_compute_mean::write_file(image_name, mean_image.data, sz);
}


void show_help(std::string cmd) {
  std::cout
    << "Create a mean image that is filled with the same pixel value given " << std::endl
    << "or computed from a given image (the channel-wise mean)." << std::endl
    << "Usage: > " << cmd << " width height B G R [depth] (for a color image)" << std::endl
    << "    or > " << cmd << " width height M [depth] (for a monochrome image)" << std::endl
    << "    or > " << cmd << " input_image [depth]" << std::endl
    << " - B, G, R, M: mean channel values" << std::endl
    << " - depth: OpenCV depth code [0|2]" << std::endl
       // CV_8U and CV_16U maps to 0 and 2 respectively.
    << "e.g.,  > " << cmd << " 256 256 104 117 123 0" << std::endl;
       //https://github.com/BVLC/caffe/wiki/Models-accuracy-on-ImageNet-2012-val
}


int main(int argc, char** argv) {
  if ((argc < 2) || (7 < argc)) {
    show_help(argv[0]);
    return 0;
  }

  int width = 0;         // The width of the output image
  int height = 0;        // The height of the output image
  int num_channels = 0;  // The number of channels of the output image
  int in_depth = CV_8U;  // The depth of the input pixel. The default is unsigned 8-bit [0 255]
  cv::Mat mean_image;    // The mean image to output
  cv::Scalar pixel_m(0.0); // The mean pixel value (in double precision)
  cv::Scalar pixel(0.0); // The pixel value converted to integers to populate the output image
  int out_depth = CV_8U; // The depth of output image. The default is unsigned 8-bit [0 255]


  if ((argc % 2) != 0) { // The depth code of output image is specified
    out_depth = atoi(argv[argc-1]);
    if ((out_depth != CV_8U) && (out_depth != CV_16U)) {
      std::cerr << "Invalid opencv depth code: " << out_depth << std::endl;
      return 0;
    }
    argc --; // so that the rest of argument parsing does not
    // need to care of the presence of the optional argument
  }

  if (argc < 4) { // Given an image, compute the mean pixel value out of the input image
    // Load the image
    cv::Mat mean_image = read_image(argv[1]);
    if (mean_image.empty()) {
      std::cerr << "Could not read the image file: " << argv[1] << std::endl;
      return 0;
    }
    width = mean_image.cols;
    height = mean_image.rows;
    num_channels = mean_image.channels();

    // Compute the mean pixel of the input image, reducing to one value per channel
    pixel_m = reduce_image(mean_image);

    in_depth = mean_image.depth();
  } else { // The pixel value is given
    width = atoi(argv[1]);
    height = atoi(argv[2]);
    num_channels = argc - 3;

    bool is_unit_scaled = true;

    // Set the mean pixel value given in the command-line arguments
    for (int c = 0; c < num_channels; ++c) {
      pixel_m[c] = atof(argv[c+3]);
      if ((pixel_m[c] < 0.0) || (std::numeric_limits<_depth_type(CV_8U)>::max() < pixel_m[c])) {
        std::cerr << "Pixel value must be specified in the scale [0.0 255.0] or in the unit scale [0.0 1.0]";
        return 0;
      }
      // Assume the unit scale if all the channel values are within the range [0.0 1.0]
      if (1.0 < pixel_m[c]) is_unit_scaled = false;
    }

    in_depth = (is_unit_scaled? CV_64F : CV_8U);
  }

  // Populate the output image with a mean pixel value
  pixel = scale_pixel(pixel_m, in_depth, out_depth);
  mean_image = cv::Mat(height, width, CV_MAKETYPE(out_depth, num_channels), pixel);
  const std::string filename = "mean_uniform.png";
  cv::imwrite(filename, mean_image);

  pixel = scale_pixel(pixel_m, in_depth, CV_64F);
  write_bin_image(height, width, CV_MAKETYPE(CV_64F, num_channels), pixel);

  return 0;
}
