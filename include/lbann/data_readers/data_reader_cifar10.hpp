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
// lbann_data_reader_cifar10 .hpp .cpp - generic_data_reader class for CIFAR10 dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_CIFAR10_HPP
#define LBANN_DATA_READER_CIFAR10_HPP

#include "data_reader.hpp"

namespace lbann {

class cifar10_reader : public generic_data_reader {
 public:
  /// constructor
  cifar10_reader(int batchSize, bool shuffle = true);
  cifar10_reader(const cifar10_reader&) = default;
  cifar10_reader& operator=(const cifar10_reader&) = default;

  /// destructor
  ~cifar10_reader();

  cifar10_reader* copy() const { return new cifar10_reader(*this); }

  bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid);
  bool fetch_label(Mat& Y, int data_id, int mb_idx, int tid);
  void load();

  /// returns image width (which should be 32)
  int get_image_width() const {
    return m_image_width;
  }

  /// returns image height (which should be 32)
  int get_image_height() const {
    return m_image_height;
  }

  int get_num_labels() const {
    return 10;
  }

  /// returns image depth (which should be 3)
  int get_image_num_channels() const {
    return m_image_num_channels;
  }

  /// returns the number of pixels in the image (should be 3072)
  int get_linearized_data_size() const {
    return m_image_width * m_image_height * m_image_num_channels;
  }

  /// returns
  int get_linearized_label_size() const {
    return 10;
  }
  
  const std::vector<int> get_data_dims() const {
    return {m_image_num_channels, m_image_height, m_image_width};
  }
  
  void save_image(Mat& pixels, const std::string filename, bool do_scale = true) {
    internal_save_image(pixels, filename, m_image_height, m_image_width,
                        m_image_num_channels, do_scale);
  }

 private:
  std::vector<std::vector<unsigned char> > m_data;
  int m_image_width;
  int m_image_height;
  int m_image_num_channels;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_CIFAR10_HPP
