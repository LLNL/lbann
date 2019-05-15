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
// data_reader_mnist_siamese .hpp .cpp - data reader class for mnist dataset
//                     employing two images per sample to feed siamese model
////////////////////////////////////////////////////////////////////////////////

#ifndef DATA_READER_MNIST_SIAMESE_HPP
#define DATA_READER_MNIST_SIAMESE_HPP

#include "data_reader_multi_images.hpp"
#include "cv_process.hpp"
#include <vector>
#include <string>
#include <utility>
#include <iostream>

namespace lbann {

/**
 * With MNIST dataset, there is no individual image file. All the images or
 * labels are packed into a single binary file respectively. This reader
 * pre-loads all the data into memory as minist_reader does.
 * However, to feed a siamese model, this reader randomly chooses the paired
 * input on-line. It maintains another data index list, 'm_shuffled_indices2'.
 * It first copies the primary list maintined by the base class to the secondary
 * list, and shuffles the secondary whenever the primary gets shuffled via the
 * overridden shuffle_indices() method.
 */
class data_reader_mnist_siamese : public data_reader_multi_images {
 public:
  using label_t = unsigned char;
  using sample_t = std::pair<int, int>;

  data_reader_mnist_siamese(const std::shared_ptr<cv_process>& pp, bool shuffle = true);
  data_reader_mnist_siamese(const data_reader_mnist_siamese&);
  data_reader_mnist_siamese& operator=(const data_reader_mnist_siamese&);
  ~data_reader_mnist_siamese() override;

  data_reader_mnist_siamese* copy() const override {
    return new data_reader_mnist_siamese(*this);
  }

  std::string get_type() const override {
    return "data_reader_mnist_siamese";
  }

  /** Set up MNIST dataset-specific input parameters, which are pre-defined
   *  and also set as the default. This does not change the setup, but only
   *  preserves the default.
   */
  void set_input_params(const int, const int, const int, const int) override;

  // dataset specific functions
  void load() override;

  /// Fetch this mini-batch's samples into X by calling the new overloaded fetch_datum()
  int fetch_data(CPUMat& X, El::Matrix<El::Int>& indices_fetched) override;
  /// Fetch this mini-batch's labels into Y by calling the new overloaded fetch_label()
  int fetch_labels(CPUMat& Y) override;

 protected:
  /**
   * Set the default configuration such as the width, height, and number of
   * channels of the image sample.
   */
  void set_defaults() override;

  // unused virtual interfaces replaced by the new interfaces that taks a pair
  // of indices to sample list.
  using data_reader_multi_images::fetch_datum;
  using data_reader_multi_images::fetch_label;
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;

  /**
   * Fetch two data items identified by the pair of indices to the pre-loaded data list,
   * and put them into the column mb_idx of matrix x.
   */
  virtual bool fetch_datum(CPUMat& X, std::pair<int, int> data_id, int mb_idx);
  /**
   * Take a pair of indices to the preloaded sample list, and compare the labels
   * of the corresponding samples. Store 1 if equal or 0 at the column mb_idx of
   * the given matrix Y.
   */
  virtual bool fetch_label(CPUMat& Y, std::pair<int, int> data_id, int mb_idx);

  /**
   * Shuffle the second index list added in this class as well as the one in the
   * base class whenever the latter gets shuffled.
   */
  void shuffle_indices() override;

 protected:
  using generic_data_reader::m_shuffled_indices;
  /// To randomly choose the siamese pair input online
  std::vector<int> m_shuffled_indices2;
  /// Store the preloaded data
  std::vector<std::vector<unsigned char>> m_image_data;
};

}  // namespace lbann

#endif  // DATA_READER_MNIST_SIAMESE_HPP
