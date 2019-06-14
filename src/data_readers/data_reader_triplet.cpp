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
// data_reader_triplet .hpp .cpp - data reader to use triplet patches
//                                 generated offline.
// Depreciated and replaced by data_reader_multihead_siamese .hpp .cpp.
// Kept here just for reference.
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_triplet.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/image.hpp"
#include <fstream>
#include <sstream>
#include <omp.h>

namespace lbann {

data_reader_triplet::data_reader_triplet(bool shuffle)
  : data_reader_multi_images(shuffle) {
  set_defaults();
}

data_reader_triplet::data_reader_triplet(const data_reader_triplet& rhs)
  : data_reader_multi_images(rhs),
    m_samples(rhs.m_samples)
{}

data_reader_triplet& data_reader_triplet::operator=(const data_reader_triplet& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  data_reader_multi_images::operator=(rhs);
  m_samples = rhs.m_samples;

  return (*this);
}

data_reader_triplet::~data_reader_triplet() {
}

void data_reader_triplet::set_defaults() {
  m_image_width = 110;
  m_image_height = 110;
  m_image_num_channels = 3;
  set_linearized_image_size();
  m_num_labels = 20;
  m_num_img_srcs = 3;
}

/**
 * Same as the parent class method except the default value of the last argument,
 * num_img_srcs, which is 3 here.
 */
void data_reader_triplet::set_input_params(const int width, const int height, const int num_ch, const int num_labels) {
  data_reader_multi_images::set_input_params(width, height, num_ch, num_labels, 3);
}


bool data_reader_triplet::fetch_datum(Mat& X, int data_id, int mb_idx) {
  std::vector<CPUMat> X_v = create_datum_views(X, mb_idx);
  sample_t sample = m_samples.get_sample(data_id);
  for (size_t i = 0; i < m_num_img_srcs; ++i) {
    El::Matrix<uint8_t> image;
    std::vector<size_t> dims;
    load_image(get_file_dir() + sample.first[i], image, dims);
    m_transform_pipeline.apply(image, X_v[i], dims);
  }
  return true;
}


bool data_reader_triplet::fetch_label(Mat& Y, int data_id, int mb_idx) {
  const label_t label = m_samples.get_label(data_id);
  Y.Set(label, mb_idx, 1);
  return true;
}


std::vector<data_reader_triplet::sample_t> data_reader_triplet::get_image_list_of_current_mb() const {
  std::vector<sample_t> ret;
  ret.reserve(m_mini_batch_size);
  return ret;
}


std::vector<data_reader_triplet::sample_t> data_reader_triplet::get_image_list() const {
  const size_t num_samples = m_samples.get_num_samples();
  std::vector<sample_t> ret;
  ret.reserve(num_samples);

  for (size_t i=0; i < num_samples; ++i) {
    ret.emplace_back(m_samples.get_sample(i));
  }
  return ret;
}


void data_reader_triplet::load() {
  const std::string data_filename = get_data_filename();

  // To support m_first_n semantic, m_samples.load() takes m_first_n
  // as an argument and attempt to shrink the CNPY arrays loaded as needed
  if (!m_samples.load(data_filename, m_first_n)) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                          + get_type() + ": failed to load the file " + data_filename);
  }

  size_t num_samples = m_samples.get_num_samples();

  if (m_first_n > 0) {
    num_samples = (static_cast<size_t>(m_first_n) <= num_samples)?
                   static_cast<size_t>(m_first_n) : num_samples;

    m_first_n = num_samples;
    set_use_percent(1.0);
    set_absolute_sample_count(0u);
  }

  // reset indices
  m_shuffled_indices.clear();

  m_shuffled_indices.resize(num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);

  select_subset_of_data();
}

}  // namespace lbann
