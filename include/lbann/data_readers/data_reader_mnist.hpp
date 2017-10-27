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
// lbann_data_reader_mnist .hpp .cpp - generic_data_reader class for MNIST dataset
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_MNIST_HPP
#define LBANN_DATA_READER_MNIST_HPP

#include "data_reader.hpp"
#include "image_preprocessor.hpp"

namespace lbann {

class mnist_reader : public generic_data_reader {
 public:
  mnist_reader(bool shuffle = true);
  mnist_reader();
  mnist_reader(const mnist_reader&) = default;
  mnist_reader& operator=(const mnist_reader&) = default;
  ~mnist_reader() {}
  mnist_reader* copy() const { return new mnist_reader(*this); }

  bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid);
  bool fetch_label(Mat& Y, int data_id, int mb_idx, int tid);

  int get_num_labels() const {
    return m_num_labels;
  }

  // MNIST-specific functions
  void load();

  int get_image_width() const {
    return m_image_width;
  }
  int get_image_height() const {
    return m_image_height;
  }
  int get_linearized_data_size() const {
    return m_image_width * m_image_height;
  }
  int get_linearized_label_size() const {
    return m_num_labels;
  }
  const std::vector<int> get_data_dims() const {
    return {1, m_image_height, m_image_width};
  }


  //don't need this, since default ctor is adequate; eliminating
  //explict implementation reduces possible bugs
  //mnist_reader& operator=(const mnist_reader& source);

  void save_image(Mat& pixels, const std::string filename, bool do_scale = true) {
    internal_save_image(pixels, filename, m_image_height, m_image_width, 1,
                        do_scale);
  }

 private:

  std::vector<std::vector<unsigned char>> m_image_data;
  int m_image_width;
  int m_image_height;
  int m_num_labels;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_MNIST_HPP
