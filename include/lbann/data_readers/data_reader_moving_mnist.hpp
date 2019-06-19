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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_MOVING_MNIST_HPP
#define LBANN_DATA_READER_MOVING_MNIST_HPP

#include "data_reader.hpp"

namespace lbann {

class moving_mnist_reader : public generic_data_reader {
public:
  moving_mnist_reader(El::Int num_frames,
                      El::Int image_height,
                      El::Int image_width,
                      El::Int num_objects);
  moving_mnist_reader(const moving_mnist_reader&) = default;
  moving_mnist_reader& operator=(const moving_mnist_reader&) = default;
  ~moving_mnist_reader() override = default;
  moving_mnist_reader* copy() const override { return new moving_mnist_reader(*this); }

  std::string get_type() const override {
    return "moving_mnist_reader";
  }

  void load() override;

  const std::vector<int> get_data_dims() const override;
  int get_num_labels() const override;
  int get_linearized_data_size() const override;
  int get_linearized_label_size() const override;

protected:
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;

private:

  /** Number of frames. */
  El::Int m_num_frames;
  /** Frame height. */
  El::Int m_image_height;
  /** Frame width. */
  El::Int m_image_width;
  /** Number of MNIST digits in each frame. */
  El::Int m_num_objects;

  /** Number of MNIST samples. */
  El::Int m_num_raw_images = 0;
  /** MNIST image height. */
  El::Int m_raw_image_height = 0;
  /** MNIST image width. */
  El::Int m_raw_image_width = 0;
  /** Raw MNIST image data. */
  std::vector<unsigned char> m_raw_image_data;
  /** Raw MNIST label data. */
  std::vector<unsigned char> m_raw_label_data;

};

}  // namespace lbann

#endif  // LBANN_DATA_READER_MOVING_MNIST_HPP
