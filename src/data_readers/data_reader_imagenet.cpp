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
// lbann_data_reader_imagenet .hpp .cpp - generic_data_reader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_imagenet.hpp"
#include "lbann/data_readers/image_utils.hpp"

#include <fstream>

namespace lbann {

imagenet_reader::imagenet_reader(int batchSize, bool shuffle)
  : generic_data_reader(batchSize, shuffle) {
  m_image_width = 256;
  m_image_height = 256;
  m_image_num_channels = 3;
  m_num_labels = 1000;
}

int imagenet_reader::fetch_data(Mat& X) {
  if(!generic_data_reader::position_valid()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: Imagenet data reader load error: !position_valid");
  }

  int num_channel_values = m_image_width * m_image_height * m_image_num_channels;
  int current_batch_size = getm_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+current_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    X.Width());

  El::Zeros(X, X.Height(), X.Width());
  El::Zeros(m_indices_fetched_per_mb, mb_size, 1);
  #pragma omp parallel for
  for (int s = 0; s < mb_size; s++) {
    int n = m_current_pos + (s * m_sample_stride);
    int index = m_shuffled_indices[n];
    string imagepath = m_image_dir + image_list[index].first;

    int width, height;
    unsigned char *pixels = (unsigned char *) std::malloc(num_channel_values*sizeof(unsigned char));
    bool ret = image_utils::loadJPG(imagepath.c_str(), width, height, false, pixels);
    if(!ret) {
      throw lbann_exception("ImageNet: image_utils::loadJPG failed to load");
    }
    if(width != m_image_width || height != m_image_height) {
      throw lbann_exception("ImageNet: mismatch data size -- either width or height");
    }

    m_indices_fetched_per_mb.Set(s, 0, index);

    for (int p = 0; p < num_channel_values; p++) {
      X.Set(p, s, pixels[p]);
    }
    std::free(pixels);

    auto pixel_col = X(El::IR(0, X.Height()), El::IR(s, s + 1));
    augment(pixel_col, m_image_height, m_image_width, m_image_num_channels);
    normalize(pixel_col, m_image_num_channels);
  }

  return mb_size;
}

int imagenet_reader::fetch_label(Mat& Y) {
  if(!position_valid()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: Imagenet data reader error: !position_valid");
  }

  int current_batch_size = getm_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+current_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    Y.Width());
  El::Zeros(Y, Y.Height(), Y.Width());

  for (int s = 0; s < mb_size; s++) {
    int n = m_current_pos + (s * m_sample_stride);
    int index = m_shuffled_indices[n];
    int label = image_list[index].second;

    Y.Set(label, s, 1);
  }
  return mb_size;
}

void imagenet_reader::load() {
  std::string imageDir = get_file_dir();
  std::string imageListFile = get_data_filename();

  m_image_dir = imageDir; /// Store the primary path to the images for use on fetch
  image_list.clear();

  // load image list
  FILE *fplist = fopen(imageListFile.c_str(), "rt");
  if (!fplist) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: failed to open: " + imageListFile);
  }

  while (!feof(fplist)) {
    char imagepath[512];
    int imagelabel;
    if (fscanf(fplist, "%s%d", imagepath, &imagelabel) <= 1) {
      break;
    }
    image_list.push_back(std::make_pair(imagepath, imagelabel));
  }
  fclose(fplist);

  // reset indices
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(image_list.size());
  for (size_t n = 0; n < image_list.size(); n++) {
    m_shuffled_indices[n] = n;
  }

  select_subset_of_data();
}

}  // namespace lbann
