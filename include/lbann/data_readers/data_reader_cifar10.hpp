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
// data_reader_cifar10 .hpp .cpp - Data reader for CIFAR-10/100
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_CIFAR10_HPP
#define LBANN_DATA_READER_CIFAR10_HPP

#include "data_reader_image.hpp"

namespace lbann {

/**
 * A data reader for the CIFAR-10/100 datasets.
 *
 * This requires the binary distributions of the datasets, which
 * must retain their original filenames.
 * CIFAR-10 vs -100 is inferred by the number of labels set.
 * @note This does not store the coarse labels from CIFAR-100.
 *
 * See:
 *     https://www.cs.toronto.edu/~kriz/cifar.html
 */
class cifar10_reader : public image_data_reader {
 public:
  cifar10_reader(bool shuffle = true);
  cifar10_reader(const cifar10_reader&) = default;
  cifar10_reader& operator=(const cifar10_reader&) = default;

  ~cifar10_reader() override;

  cifar10_reader* copy() const override { return new cifar10_reader(*this); }

  std::string get_type() const override {
    return "cifar10_reader";
  }

  void set_input_params(const int, const int, const int, const int) override { set_defaults(); }
  void load() override;

 protected:
  void set_defaults() override;
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;

 private:
  /**
   * Loaded image data.
   * This will be stored in "OpenCV" format for ease of preprocessing.
   */
  std::vector<std::vector<unsigned char>> m_images;
  /** Loaded label information. */
  std::vector<uint8_t> m_labels;
};

}  // namespace lbann

#endif  // LBANN_DATA_READER_CIFAR10_HPP
