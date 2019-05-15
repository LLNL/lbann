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

#include "lbann/data_readers/data_reader_mnist_siamese.hpp"
#include "lbann/data_readers/image_utils.hpp"
#include "lbann/utils/file_utils.hpp"
#include <fstream>
#include <sstream>
#include <omp.h>
#include <algorithm> // shuffle()
#include <array>
#include <limits>

namespace lbann {

data_reader_mnist_siamese::data_reader_mnist_siamese(const std::shared_ptr<cv_process>& pp, bool shuffle)
  : data_reader_multi_images(pp, shuffle) {
  set_defaults();
}

data_reader_mnist_siamese::data_reader_mnist_siamese(const data_reader_mnist_siamese& rhs)
  : data_reader_multi_images(rhs),
    m_shuffled_indices2(rhs.m_shuffled_indices2),
    m_image_data(rhs.m_image_data)
{}

data_reader_mnist_siamese& data_reader_mnist_siamese::operator=(const data_reader_mnist_siamese& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  data_reader_multi_images::operator=(rhs);
  m_shuffled_indices2 = rhs.m_shuffled_indices2;
  m_image_data = rhs.m_image_data;

  return (*this);
}

data_reader_mnist_siamese::~data_reader_mnist_siamese() {
}

void data_reader_mnist_siamese::set_defaults() {
  m_image_width = 28;
  m_image_height = 28;
  m_image_num_channels = 1;
  set_linearized_image_size();
  m_num_labels = 2;
  m_num_img_srcs = 2;
}


void data_reader_mnist_siamese::set_input_params(
  const int, const int, const int, const int) {
  set_defaults();
}


/**
 * Fill the input minibatch matrix with the samples of image pairs by using
 * the overloaded fetch_datum()
 */
int data_reader_mnist_siamese::fetch_data(CPUMat& X, El::Matrix<El::Int>& indices_fetched) {
  int nthreads = m_io_thread_pool->get_num_threads();
  if(!position_valid()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__)
      + " :: " + get_type() + "  load error: !position_valid"
      + " -- current pos = " + std::to_string(m_current_pos)
      + " and there are " + std::to_string(m_shuffled_indices.size()) + " indices");
  }

  /// Allow each thread to perform any preprocessing necessary on the
  /// data source prior to fetching data
  for (int t = 0; t < nthreads; t++) {
    preprocess_data_source(t);
  }

  int loaded_batch_size = get_loaded_mini_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+loaded_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    X.Width());

  El::Zeros_seq(X, X.Height(), X.Width());
  El::Zeros_seq(indices_fetched, mb_size, 1);

  std::string error_message;
  for (int s = 0; s < mb_size; s++) {
    int n = m_current_pos + (s * m_sample_stride);
    sample_t index = std::make_pair(m_shuffled_indices[n], m_shuffled_indices2[n]);
    bool valid = fetch_datum(X, index, s);
    if (valid) {
      El::Int index_coded = m_shuffled_indices[n] + m_shuffled_indices2[n]*(std::numeric_limits<label_t>::max()+1);
      indices_fetched.Set(s, 0, index_coded);
    } else{
      error_message = "invalid datum";
    }
  }
  if (!error_message.empty()) { LBANN_ERROR(error_message); }

  /// Allow each thread to perform any postprocessing necessary on the
  /// data source prior to fetching data
  for (int t = 0; t < nthreads; t++) {
    postprocess_data_source(t);
  }

  return mb_size;
}


/**
 * Fill the ground truth table by using the overloaded fetch_label()
 */
int data_reader_mnist_siamese::fetch_labels(CPUMat& Y) {
  if(!position_valid()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: generic data reader load error: !position_valid");
  }

  int loaded_batch_size = get_loaded_mini_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+loaded_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    Y.Width());

  El::Zeros(Y, Y.Height(), Y.Width());

  std::string error_message;
  for (int s = 0; s < mb_size; s++) {
    int n = m_current_pos + (s * m_sample_stride);
    sample_t index = std::make_pair(m_shuffled_indices[n], m_shuffled_indices2[n]);
    bool valid = fetch_label(Y, index, s);
    if (!valid) {
      error_message = "invalid label";
    }
  }
  if (!error_message.empty()) { LBANN_ERROR(error_message); }

  return mb_size;
}


bool data_reader_mnist_siamese::fetch_datum(CPUMat& X, std::pair<int, int> data_id, int mb_idx) {
  int tid = m_io_thread_pool->get_local_thread_id();
  std::vector<CPUMat> X_v = create_datum_views(X, mb_idx);

  using raw_data_t = std::vector<unsigned char>;
  using local_sample_t = std::array<raw_data_t*, 2>;
  local_sample_t sample;
  sample[0] = &m_image_data[data_id.first];
  sample[1] = &m_image_data[data_id.second];

  for(size_t i=0u; i < sample.size(); ++i) {
    int width=0, height=0, img_type=0;
    bool ret = true;

#if 1
    // Construct a zero copying view to a portion of a preloaded data buffer
    // This has nothing to do with the image type but only to create view on a block of bytes
    using InputBuf_T = lbann::cv_image_type<uint8_t>;
    const cv::Mat image_buf(1, sample[i]->size()-1, InputBuf_T::T(1), &((*sample[i])[1]));
#else
    raw_data_t image_buf(sample[i]->begin()+1, sample[i]->end()); // make copy of the raw data
#endif
    ret = lbann::image_utils::import_image(image_buf, width, height, img_type, *(m_pps[tid]), X_v[i]);

    if(!ret) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                            + get_type() + ": image_utils::import_image failed to load");
    }
    if((width * height * CV_MAT_CN(img_type)) != m_image_linearized_size) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                            + get_type() + ": mismatch data size -- either width, height or channel - "
                            + " [w,h,c]=[" + std::to_string(width) + "x" + std::to_string(height)
                            + "x" + std::to_string(CV_MAT_CN(img_type)) + "] != " + std::to_string(m_image_linearized_size));
    }
  }
  return true;
}


bool data_reader_mnist_siamese::fetch_label(CPUMat& Y, std::pair<int, int> data_id, int mb_idx) {
  const label_t label_1 = m_image_data[data_id.first][0];
  const label_t label_2 = m_image_data[data_id.second][0];
  const label_t label = static_cast<label_t>(label_1 == label_2);
  Y.Set(label, mb_idx, 1);
  return true;
}


bool data_reader_mnist_siamese::fetch_datum(CPUMat& X, int data_id, int mb_idx) {
  throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                        + get_type() + ": unused interface is called");
  return false;
}


bool data_reader_mnist_siamese::fetch_label(CPUMat& Y, int data_id, int mb_idx) {
  throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                        + get_type() + ": unused interface is called");
  return false;
}


// The function is defined in data_readers/data_reader_mnist.cpp
extern void load_mnist_data(const std::string imagepath, const std::string labelpath,
  const int first_n, std::vector<std::vector<unsigned char> >& m_image_data);


void data_reader_mnist_siamese::load() {
  if (is_master()) {
    std::cerr << "starting lbann::" << get_type() << "::load\n";
  }
  m_image_data.clear();

  const std::string FileDir = get_file_dir();
  const std::string ImageFile = get_data_filename();
  const std::string LabelFile = get_label_filename();

  // set filepath
  const std::string imagepath = FileDir + "/" + ImageFile;
  const std::string labelpath = FileDir + "/" + LabelFile;

  if (is_master()) {
    std::cerr << "read labels!\n";
  }

  load_mnist_data(imagepath, labelpath, m_first_n, m_image_data);

  if (m_first_n > 0) {
    set_use_percent(1.0);
    set_absolute_sample_count(0u);
  }

  // reset indices
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_image_data.size());
  for (size_t n = 0; n < m_shuffled_indices.size(); n++) {
    m_shuffled_indices[n] = n;
  }
  if (is_master()) {
    std::cerr << "calling select_subset_of_data; m_shuffled_indices.size: " <<
      m_shuffled_indices.size() << std::endl;
  }
  select_subset_of_data();
}


void data_reader_mnist_siamese::shuffle_indices() {
  if (m_shuffled_indices2.size() != m_shuffled_indices.size()) {
    m_shuffled_indices2 = m_shuffled_indices;
    if (!m_shuffle) {
      std::shuffle(m_shuffled_indices2.begin(), m_shuffled_indices2.end(),
                   get_data_seq_generator());
    }
  }
  if (m_shuffle) {
    std::shuffle(m_shuffled_indices.begin(), m_shuffled_indices.end(),
                 get_data_seq_generator());
    std::shuffle(m_shuffled_indices2.begin(), m_shuffled_indices2.end(),
                 get_data_seq_generator());
  }
}


}  // namespace lbann
