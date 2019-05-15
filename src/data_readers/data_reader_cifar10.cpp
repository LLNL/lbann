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
// lbann_data_reader_cifar10 .hpp .cpp - generic_data_reader class for CIFAR10 dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_cifar10.hpp"

namespace lbann {

cifar10_reader::cifar10_reader(bool shuffle)
  : image_data_reader(shuffle) {
  set_defaults();
}

cifar10_reader::~cifar10_reader() {}

void cifar10_reader::set_defaults() {
  m_image_width = 32;
  m_image_height = 32;
  m_image_num_channels = 3;
  set_linearized_image_size();
  m_num_labels = 10;
}

void cifar10_reader::load() {
  //open data file
  std::string image_dir = get_file_dir();
  std::string filename = get_data_filename();
  std::string path = image_dir + "/" + filename;
  std::ifstream in(path, std::ios::binary);
  if (!in.good()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: failed to open " + path + " for reading");
  }

  //get number of images, with error checking
  int len = get_linearized_data_size() + 1;  //should be 3073
  in.seekg(0, in.end);
  std::streampos fs = in.tellg();
  in.seekg(0, in.beg);
  if (fs % len != 0) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " ::  fs % len != 0; fs: " + std::to_string(fs) + " len: " +
      std::to_string(len));
  }

  //reserve space for string images
  int num_images = fs / len;
  m_data.resize(num_images);
  for (auto & h : m_data) {
    h.resize(len);
  }

  //read in the images; each image is 1 byte, which is the
  //label (0-9), and 3072 pixels
  for (auto & h : m_data) {
    in.read((char *)&(h[0]), len);
  }
  in.close();

  m_shuffled_indices.resize(m_data.size());
  for (size_t n = 0; n < m_data.size(); n++) {
    m_shuffled_indices[n] = n;
  }

  select_subset_of_data();
}

bool cifar10_reader::fetch_datum(CPUMat& X, int data_id, int mb_idx) {
  for (size_t p = 1; p<m_data[data_id].size(); p++) {
    X.Set(p-1, mb_idx, m_data[data_id][p]);
  }

  auto pixel_col = X(El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));
  augment(pixel_col, m_image_height, m_image_width, m_image_num_channels);
  normalize(pixel_col, m_image_num_channels);
  pixel_noise(pixel_col); //add noise to image, disable by default
  return true;
}

bool cifar10_reader::fetch_label(CPUMat& Y, int data_id, int mb_idx) {
  auto label = (int)m_data[data_id][0];
  Y.Set(label, mb_idx, 1);
  return true;
}

}  // namespace lbann
