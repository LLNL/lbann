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
  // These are all specified by the CIFAR10/100 description.
  constexpr size_t num_channels = 3;
  constexpr size_t channel_size = 32*32;
  constexpr size_t image_size = num_channels*channel_size;
  constexpr size_t cifar10_label_size = 1;
  constexpr size_t cifar100_label_size = 2;

  if (m_num_labels != 10 && m_num_labels != 100) {
    LBANN_ERROR("Unsupported number of labels for CIFAR10/100.");
  }

  const bool cifar100 = m_num_labels == 100;

  std::string path = get_file_dir();
  // These filenames are specified by the CIFAR-10/100 dataset description.
  std::vector<std::string> filenames;
  size_t images_per_file = 10000;
  if (this->get_role() == "train") {
    if (cifar100) {
      filenames = {"train.bin"};
      images_per_file = 50000;
    } else {
      filenames = {
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin"
      };
    }
  } else if (this->get_role() == "test") {
    if (cifar100) {
      filenames = {"test.bin"};
    } else {
      filenames = {"test_batch.bin"};
    }
  } else {
    LBANN_ERROR("Unsupported training mode for CIFAR loading.");
  }

  for (const auto& filename : filenames) {
    std::ifstream f(path + "/" + filename,
                    std::ios::in | std::ios::binary);
    if (!f.good()) {
      LBANN_ERROR("Could not open " + path + "/" + filename);
    }
    // Temporary buffer to hold an image.
    std::vector<uint8_t> buf(image_size + (cifar100 ?
                                           cifar100_label_size :
                                           cifar10_label_size), 0);
    for (size_t i = 0; i < images_per_file; ++i) {
      f.read(reinterpret_cast<char*>(buf.data()), buf.size());
      if (static_cast<size_t>(f.gcount()) != buf.size()) {
        LBANN_ERROR("Could not read from " + path + "/" + filename);
      }
      // CIFAR-10 has only one label; for CIFAR-100, the second byte is the
      // fine label.
      m_labels.push_back(buf[cifar100 ? 1 : 0]);
      // Convert to OpenCV layout.
      std::vector<uint8_t> image(image_size);
      for (size_t channel = 0; channel < num_channels; ++channel) {
        const size_t src_start = channel*channel_size;
        for (size_t j = 0; j < channel_size; ++j) {
          image[j*num_channels + channel] = buf[src_start + j];
        }
      }
      m_images.push_back(std::move(image));
    }
    f.close();
  }

  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_images.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

bool cifar10_reader::fetch_datum(CPUMat& X, int data_id, int mb_idx) {
  // Copy to a matrix so we can do data augmentation.
  // Sizes per CIFAR-10/100 dataset description.
  El::Matrix<uint8_t> image(3*32*32, 1);
  std::vector<size_t> dims = {size_t(3), size_t(32), size_t(32)};
  std::copy_n(m_images[data_id].data(), 3*32*32, image.Buffer());
  auto X_v = X(El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));
  m_transform_pipeline.apply(image, X_v, dims);
  return true;
}

bool cifar10_reader::fetch_label(CPUMat& Y, int data_id, int mb_idx) {
  Y.Set(m_labels[data_id], mb_idx, 1);
  return true;
}

}  // namespace lbann
