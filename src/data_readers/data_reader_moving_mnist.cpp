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
#include "lbann/models/model.hpp"
#include <fstream>
#include <functional>

namespace lbann {

namespace {

/** Called repeatedly to incrementally create a hash value from
 *  several variables.
 *
 *  Copied from Boost. See
 *  https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine.
 */
template <typename T>
inline void hash_combine(size_t& seed, T v) {
  seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

} // namespace

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

bool moving_mnist_reader::fetch_datum(CPUMat& X, int data_id, int col) {

  // Useful constants
  constexpr DataType zero = 0;
  constexpr DataType one = 1;

  // Choose raw images
  /// @todo Implementation with uniform distribution
  std::vector<El::Int> raw_image_indices(m_num_objects);
  for (El::Int obj = 0; obj < m_num_objects; ++obj) {
    size_t hash = 1234;
    hash_combine(hash, data_id);
    hash_combine(hash, m_model->get_epoch());
    hash_combine(hash, obj);
    raw_image_indices[obj] = hash % m_num_raw_images;
  }

  // Determine object boundaries
  std::vector<std::array<El::Int, 4>> bounds(m_num_objects);
  for (El::Int obj = 0; obj < m_num_objects; ++obj) {
    auto& xmin = bounds[obj][0] = m_raw_image_width;
    auto& xmax = bounds[obj][1] = 0;
    auto& ymin = bounds[obj][2] = m_raw_image_height;
    auto& ymax = bounds[obj][3] = 0;
    const auto& raw_image_offset = (raw_image_indices[obj]
                                    * m_raw_image_height
                                    * m_raw_image_width);
    const auto* raw_image = &m_raw_image_data[raw_image_offset];
    for (El::Int j = 0; j < m_raw_image_height; ++j) {
      for (El::Int i = 0; i < m_raw_image_width; ++i) {
        if (raw_image[i + j * m_raw_image_width] != 0) {
          xmin = std::min(xmin, i);
          xmax = std::max(xmax, i+1);
          ymin = std::min(ymin, j);
          ymax = std::max(ymax, j+1);
        }
      }
    }
    xmin = std::min(xmin, xmax);
    ymin = std::min(ymin, ymax);
  }

  // Initial positions and velocities
  /// @todo Ensure objects don't overlap
  std::vector<std::vector<std::array<DataType, 2>>> pos(m_num_objects);
  std::vector<std::array<DataType, 2>> v(m_num_objects);
  std::uniform_real_distribution<DataType> dist(zero, one);
  const DataType vmax = std::hypot(m_image_width, m_image_height) / 5;
  for (El::Int obj = 0; obj < m_num_objects; ++obj) {
    const auto& object_width = bounds[obj][1] - bounds[obj][0];
    const auto& object_height = bounds[obj][3] - bounds[obj][2];
    pos[obj].resize(m_num_frames);
    pos[obj][0][0] = (m_image_width - object_width + 1) * dist(get_generator());
    pos[obj][0][1] = (m_image_height - object_height + 1) * dist(get_generator());
    const DataType vnorm = vmax * dist(get_generator());
    const DataType theta = 2 * M_PI * dist(get_generator());
    v[obj][0] = vnorm * std::sin(theta);
    v[obj][1] = vnorm * std::cos(theta);
  }

  // Determine object positions
  /// @todo Ensure objects don't overlap
  for (El::Int frame = 1; frame < m_num_frames; ++frame) {
    for (El::Int obj = 0; obj < m_num_objects; ++obj) {

      // Linear motion
      auto& x = pos[obj][frame][0];
      auto& y = pos[obj][frame][1];
      auto& vx = v[obj][0];
      auto& vy = v[obj][1];
      x = pos[obj][frame-1][0] + vx;
      y = pos[obj][frame-1][1] + vy;

      // Reflections at boundaries
      const auto& object_width = bounds[obj][1] - bounds[obj][0];
      const auto& object_height = bounds[obj][3] - bounds[obj][2];
      const DataType xmax = m_image_width - object_width + 1;
      const DataType ymax = m_image_height - object_height + 1;
      if (x <= zero || x >= xmax) {
        x = std::min(std::max(x, zero), xmax);
        vx = -vx;
      }
      if (y <= zero || y >= ymax) {
        y = std::min(std::max(y, zero), ymax);
        vy = -vy;
      }
    }
  }

  // Populate frames
  std::memset(X.Buffer(0, col), 0, X.Height() * sizeof(DataType));
  for (El::Int obj = 0; obj < m_num_objects; ++obj) {

    // Get raw image
    const auto& object_width = bounds[obj][1] - bounds[obj][0];
    const auto& object_height = bounds[obj][3] - bounds[obj][2];
    const auto& object_width_offset = bounds[obj][0];
    const auto& object_height_offset = bounds[obj][2];
    const auto& raw_image_offset = ((raw_image_indices[obj]
                                     * m_raw_image_height
                                     * m_raw_image_width)
                                    + object_width_offset
                                    + (object_height_offset
                                       * m_raw_image_width));
    const auto* raw_image = &m_raw_image_data[raw_image_offset];

    // Copy raw image into each frame
    const auto& xmax = m_image_width - object_width + 1;
    const auto& ymax = m_image_height - object_height + 1;
    for (El::Int frame = 0; frame < m_num_frames; ++frame) {

      // Get image position in current frame
      El::Int xoff = pos[obj][frame][0];
      El::Int yoff = pos[obj][frame][1];
      xoff = std::min(std::max(xoff, El::Int(0)), xmax-1);
      yoff = std::min(std::max(yoff, El::Int(0)), ymax-1);

      // Copy raw image into position
      for (El::Int channel = 0; channel < 3; ++channel) {
        for (El::Int j = 0; j < object_height; ++j) {
          for (El::Int i = 0; i < object_width; ++i) {
            const auto& row = (frame * 3 * m_image_height * m_image_width
                               + channel * m_image_height * m_image_width
                               + (yoff+j) * m_image_width
                               + (xoff+i));
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

bool moving_mnist_reader::fetch_label(CPUMat& Y, int data_id, int col) {

  // Choose raw images
  /// @todo Implementation with uniform distribution
  std::vector<El::Int> raw_image_indices(m_num_objects);
  for (El::Int obj = 0; obj < m_num_objects; ++obj) {
    size_t hash = 1234;
    hash_combine(hash, data_id);
    hash_combine(hash, m_model->get_epoch());
    hash_combine(hash, obj);
    raw_image_indices[obj] = hash % m_num_raw_images;
  }

  // Label is sum of raw image labels
  El::Int sum = 0;
  for (const auto& i : raw_image_indices) {
    sum += m_raw_label_data[i];
  }
  auto&& Y_col = El::View(Y, El::ALL, El::IR(col));
  El::Zero(Y_col);
  Y_col(sum, 0) = DataType(1);

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
