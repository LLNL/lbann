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

#include "lbann/data_readers/data_reader_moving_mnist.hpp"
#include "lbann/utils/file_utils.hpp"
#include <fstream>
#include <functional>

namespace lbann {

moving_mnist_reader::moving_mnist_reader(El::Int num_frames,
                                         El::Int image_height,
                                         El::Int image_width,
                                         El::Int num_objects)
  : generic_data_reader(true),
    m_num_frames(num_frames),
    m_image_height(image_height),
    m_image_width(image_width),
    m_num_objects(num_objects) {}

// Data dimension access functions
const std::vector<int> moving_mnist_reader::get_data_dims() const {
  std::vector<int> dims(4);
  dims[0] = m_num_frames;
  dims[1] = 3;
  dims[2] = m_image_height;
  dims[3] = m_image_width;
  return dims;
}
int moving_mnist_reader::get_num_labels() const {
  return 1 + 9 * m_num_objects;
}
int moving_mnist_reader::get_linearized_data_size() const {
  const auto& dims = get_data_dims();
  return std::accumulate(dims.begin(), dims.end(), 1,
                         std::multiplies<int>());
}
int moving_mnist_reader::get_linearized_label_size() const {
  return get_num_labels();
}

bool moving_mnist_reader::fetch_datum(CPUMat& X, int data_id, int col, thread_pool& io_thread_pool) {

  //  int tid = io_thread_pool.get_local_thread_id();
  // Useful constants
  constexpr DataType zero = 0;
  constexpr DataType one = 1;

  // Image parameters
  const El::Int xmax = m_image_height - m_raw_image_height + 1;
  const El::Int ymax = m_image_width - m_raw_image_width + 1;
  const DataType vmax = std::sqrt(xmax*xmax + ymax*ymax) / 5;

  // Compute object positions
  std::vector<std::vector<DataType>> xpos, ypos;
  std::uniform_real_distribution<DataType> dist(zero, one);
  for (El::Int obj = 0; obj < m_num_objects; ++obj) {

    // Initialize initial position and velocity
    DataType x = xmax * dist(get_generator());
    DataType y = ymax * dist(get_generator());
    const DataType v = vmax * dist(get_generator());
    const DataType theta = 2 * M_PI * dist(get_generator());
    DataType vx = v * std::sin(theta);
    DataType vy = v * std::cos(theta);

    // Apply linear motion
    // Note: Objects are reflected when they reach boundary
    xpos.emplace_back(m_num_frames);
    ypos.emplace_back(m_num_frames);
    for (El::Int frame = 0; frame < m_num_frames; ++frame) {
      xpos[obj][frame] = x;
      ypos[obj][frame] = y;
      x += vx;
      y += vy;
      if (x <= zero || x >= DataType(xmax)) {
        x = std::min(std::max(x, zero), DataType(xmax));
        vx = -vx;
      }
      if (y <= zero || y >= DataType(ymax)) {
        y = std::min(std::max(y, zero), DataType(ymax));
        vy = -vy;
      }
    }

  }

  // Choose raw images
  /// @todo Implementation with uniform distribution
  std::vector<El::Int> raw_image_indices;
  for (El::Int i = 0; i < m_num_objects; ++i) {
    const El::Int hash = std::hash<int>()(data_id) ^ std::hash<El::Int>()(i);
    raw_image_indices.push_back(hash % m_num_raw_images);
  }

  // Populate frames
  std::memset(X.Buffer(0, col), 0, X.Height() * sizeof(DataType));
  for (El::Int obj = 0; obj < m_num_objects; ++obj) {

    // Get raw image
    const auto& raw_image_offset = (raw_image_indices[obj]
                                    * m_raw_image_height
                                    * m_raw_image_width);
    const auto* raw_image = &m_raw_image_data[raw_image_offset];

    // Copy raw image into each frame
    for (El::Int frame = 0; frame < m_num_frames; ++frame) {

      // Get image position in current frame
      El::Int xoff = xpos[obj][frame];
      El::Int yoff = ypos[obj][frame];
      xoff = std::min(std::max(xoff, El::Int(0)), xmax-1);
      yoff = std::min(std::max(yoff, El::Int(0)), ymax-1);

      // Copy raw image into position
      for (El::Int channel = 0; channel < 3; ++channel) {
        for (El::Int j = 0; j < m_raw_image_height; ++j) {
          for (El::Int i = 0; i < m_raw_image_width; ++i) {
            const auto& row = (frame * 3 * m_image_height * m_image_width
                               + channel * m_image_height * m_image_width
                               + (xoff+j) * m_image_width
                               + (yoff+i));
            auto& pixel = X(row, col);
            pixel += raw_image[i + j * m_raw_image_width] / 255.0;
            pixel = std::min(pixel, one);
          }
        }
      }

    }

  }

  return true;
}

bool moving_mnist_reader::fetch_label(CPUMat& Y, int data_id, int col, thread_pool& io_thread_pool) {

  //  int tid = io_thread_pool.get_local_thread_id();
  // Choose raw images
  /// @todo Implementation with uniform distribution
  std::vector<El::Int> raw_image_indices;
  for (El::Int i = 0; i < m_num_objects; ++i) {
    const El::Int hash = std::hash<int>()(data_id) ^ std::hash<El::Int>()(i);
    raw_image_indices.push_back(hash % m_num_raw_images);
  }

  // Label is sum of raw image labels
  El::Int sum = 0;
  for (const auto& i : raw_image_indices) {
    sum += m_raw_label_data[i];
  }
  Y(sum, col) = DataType(1);

  return true;
}

void moving_mnist_reader::load() {

  // Read image data
  const auto& image_file = get_file_dir() + "/" + get_data_filename();
  std::ifstream fs_image(image_file.c_str(),
                         std::fstream::in | std::fstream::binary);
  unsigned int num_images = 0;
  unsigned int image_height = 0;
  unsigned int image_width = 0;
  fs_image.ignore(4);
  fs_image.read(reinterpret_cast<char*>(&num_images), 4);
  fs_image.read(reinterpret_cast<char*>(&image_height), 4);
  fs_image.read(reinterpret_cast<char*>(&image_width), 4);
  __swapEndianInt(num_images);
  __swapEndianInt(image_height);
  __swapEndianInt(image_width);
  m_num_raw_images = num_images;
  m_raw_image_height = image_height;
  m_raw_image_width = image_width;
  m_raw_image_data.resize(num_images * image_height * image_width);
  fs_image.read(reinterpret_cast<char*>(m_raw_image_data.data()),
                num_images * image_height * image_width);
  fs_image.close();

  // Read labels
  const auto& label_file = get_file_dir() + "/" + get_label_filename();
  std::ifstream fs_label(label_file.c_str(),
                         std::fstream::in | std::fstream::binary);
  fs_label.ignore(8);
  m_raw_label_data.resize(num_images);
  fs_label.read(reinterpret_cast<char*>(m_raw_label_data.data()), num_images);
  fs_label.close();

  // Reset indices
  m_shuffled_indices.resize(num_images);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();

}

}  // namespace lbann
