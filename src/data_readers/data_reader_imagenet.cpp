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
#include <omp.h>

namespace lbann {

imagenet_reader::imagenet_reader(int batchSize, bool shuffle)
  : generic_data_reader(batchSize, shuffle) {
  m_image_width = 256;
  m_image_height = 256;
  m_image_num_channels = 3;
  m_num_labels = 1000;

  // Preallocate buffer space for each thread.
  m_pixel_bufs.resize(omp_get_max_threads());
  const int num_channel_values = m_image_width * m_image_height * m_image_num_channels;
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    m_pixel_bufs[i].resize(num_channel_values * sizeof(unsigned char));
  }
}

bool imagenet_reader::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {
  const int num_channel_values = m_image_width * m_image_height * m_image_num_channels;
  const std::string imagepath = get_file_dir() + m_image_list[data_id].first;

  int width, height;
  unsigned char *pixels = m_pixel_bufs[tid].data();
  bool ret = lbann::image_utils::loadJPG(imagepath, width, height, false, pixels);
  if(!ret) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + "ImageNet: image_utils::loadJPG failed to load - " 
                          + imagepath);
  }
  if(width != m_image_width || height != m_image_height) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + "ImageNet: mismatch data size -- either width or height - " 
                          + imagepath + "[w,h]=[" + std::to_string(width) + "x" + std::to_string(height) +"]");
  }

  for (int p = 0; p < num_channel_values; p++) {
    X(p, mb_idx) = pixels[p];
  }

  auto pixel_col = X(El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));
  augment(pixel_col, m_image_height, m_image_width, m_image_num_channels);
  normalize(pixel_col, m_image_num_channels);
  pixel_noise(pixel_col); //add noise to image, disable by default

  return true;
}

bool imagenet_reader::fetch_label(Mat& Y, int data_id, int mb_idx, int tid) {
  int label = m_image_list[data_id].second;
  Y.Set(label, mb_idx, 1);
  return true;
}

void imagenet_reader::load() {
  const std::string imageDir = get_file_dir(); // TODO: remove or use this
  const std::string imageListFile = get_data_filename();

  m_image_list.clear();

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
    m_image_list.push_back(std::make_pair(imagepath, imagelabel));
  }
  fclose(fplist);

  // reset indices
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_image_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);

  select_subset_of_data();
}

}  // namespace lbann
