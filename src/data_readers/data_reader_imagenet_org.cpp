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
// data_reader_imagenet_org .hpp .cpp - generic_data_reader class for ImageNet dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_imagenet_org.hpp"
#include "lbann/data_readers/image_utils.hpp"
#include <fstream>
#include <omp.h>

namespace lbann {

imagenet_reader_org::imagenet_reader_org(bool shuffle)
  : image_data_reader(shuffle) {
  set_defaults();
  allocate_pixel_bufs();
}

void imagenet_reader_org::set_defaults() {
  m_image_width = 256;
  m_image_height = 256;
  m_image_num_channels = 3;
  set_linearized_image_size();
  m_num_labels = 1000;
}

void imagenet_reader_org::set_input_params(const int width, const int height, const int num_ch, const int num_labels) {
  image_data_reader::set_input_params(width, height, num_ch, num_labels);
  allocate_pixel_bufs();
}

void imagenet_reader_org::allocate_pixel_bufs() {
  // Preallocate buffer space for each thread.
  m_pixel_bufs.resize(omp_get_max_threads());
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    m_pixel_bufs[i].resize(m_image_linearized_size * sizeof(unsigned char));
  }
}

bool imagenet_reader_org::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {
  const std::string imagepath = get_file_dir() + m_image_list[data_id].first;

  int width, height;
  unsigned char *pixels = m_pixel_bufs[tid].data();
  bool ret = lbann::image_utils::loadIMG(imagepath, width, height, false, pixels);
  if(!ret) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                          + get_type() + ": image_utils::loadIMG failed to load - "
                          + imagepath);
  }
  if(width != m_image_width || height != m_image_height) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                          + get_type() + ": mismatch data size -- either width or height - "
                          + imagepath + "[w,h]=[" + std::to_string(width) + "x" + std::to_string(height) +"]");
  }

  for (int p = 0; p < m_image_linearized_size; p++) {
    X(p, mb_idx) = pixels[p];
  }

  auto pixel_col = X(El::IR(0, X.Height()), El::IR(mb_idx, mb_idx + 1));
  augment(pixel_col, m_image_height, m_image_width, m_image_num_channels);
  normalize(pixel_col, m_image_num_channels);
  pixel_noise(pixel_col); //add noise to image, disable by default

  return true;
}

}  // namespace lbann
