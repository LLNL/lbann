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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_UTILS_IMAGE_HPP
#define LBANN_UTILS_IMAGE_HPP

#include "lbann/base.hpp"

namespace lbann {

/**
 * @brief Load an image from filename.
 * @param filename The path to the image to load.
 * @param dst Image will be loaded into this matrix, in OpenCV format.
 * @param dims Will contain the dimensions of the image as {channels, height,
 * width}.
 */
void load_image(const std::string& filename, El::Matrix<uint8_t>& dst,
                std::vector<size_t>& dims);

/**
 * @brief Decode an image from buf.
 * @param src A buffer containing image data to be decoded.
 * @param dst Image will be loaded into this matrix, in OpenCV format.
 * @param dims Will contain the dimensions of the image as {channels, height,
 * width}.
 */
void decode_image(El::Matrix<uint8_t>& src, El::Matrix<uint8_t>& dst,
                  std::vector<size_t>& dims);

/**
 * @brief Save an image to filename.
 * @param filename The path to the image to write.
 * @param src The image to save. This is in OpenCV format.
 * @param dims The dimensions of the image.
 */
void save_image(const std::string& filename, El::Matrix<uint8_t>& src,
                const std::vector<size_t>& dims);
/**
 * @brief Save an image to filename.
 * @param filename The path to the image to write.
 * @param src The image to save. This is in standard LBANN format, and will be
 * converted to a uint8_t matrix, interpolating between the min and max values
 * in it.
 * @param dims The dimensions of the image.
 */
void save_image(const std::string& filename, const CPUMat& src,
                const std::vector<size_t>& dims);
/**
 * @brief Convert image from El::Matrix<DataType> to El::Matrix<uint8_t>
 * @param image The image to convert.
 * @param dims The dimensions of the image.
 * @returns El::Matrix<uint8_t> Returns image in El::Matrix<uint8_t> format
 */
El::Matrix<uint8_t> get_uint8_t_image(const CPUMat& image,
                                      const std::vector<size_t>& dims);
/**
 * @brief Encodes image to std:string format
 * @param image The image to convert
 * @param dims The dimensions of the image.
 * @returns std::string Returns image in std::string format
 */
std::string encode_image(const El::Matrix<uint8_t>& image,
                         const std::vector<size_t>& dims);

}  // namespace lbann

#endif  // LBANN_UTILS_IMAGE_HPP
