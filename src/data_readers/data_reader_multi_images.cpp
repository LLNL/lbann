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
// data_reader_multi_images .hpp .cpp - generic data reader class for datasets
//                                      employing multiple images per sample
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_multi_images.hpp"
#include "lbann/data_readers/image_utils.hpp"
#include "lbann/utils/file_utils.hpp"
#include <fstream>
#include <sstream>
#include <omp.h>

namespace lbann {

data_reader_multi_images::data_reader_multi_images(const std::shared_ptr<cv_process>& pp, bool shuffle)
  : imagenet_reader(pp, shuffle) {
  set_defaults();
}

data_reader_multi_images::data_reader_multi_images(const data_reader_multi_images& rhs)
  : imagenet_reader(rhs),
    m_image_list(rhs.m_image_list),
    m_num_img_srcs(rhs.m_num_img_srcs)
{}

data_reader_multi_images& data_reader_multi_images::operator=(const data_reader_multi_images& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  imagenet_reader::operator=(rhs);
  m_image_list = rhs.m_image_list;
  m_num_img_srcs = rhs.m_num_img_srcs;

  return (*this);
}

data_reader_multi_images::~data_reader_multi_images() {
}

void data_reader_multi_images::set_defaults() {
  m_image_width = 28;
  m_image_height = 28;
  m_image_num_channels = 1;
  set_linearized_image_size();
  m_num_labels = 2;
  m_num_img_srcs = 1;
}

void data_reader_multi_images::set_input_params(const int width, const int height, const int num_ch, const int num_labels, const int num_img_srcs) {
  imagenet_reader::set_input_params(width, height, num_ch, num_labels);
  if (num_img_srcs > 0) {
    m_num_img_srcs = num_img_srcs;
  } else if (num_img_srcs < 0) {
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__<< " :: " << get_type() << " setup error: invalid number of image sources";
    throw lbann_exception(err.str());
  }
}

void data_reader_multi_images::set_input_params(const int width, const int height, const int num_ch, const int num_labels) {
  set_input_params(width, height, num_ch, num_labels, 1);
}

std::vector<CPUMat> data_reader_multi_images::create_datum_views(CPUMat& X, const int mb_idx) const {
  std::vector<CPUMat> X_v(m_num_img_srcs);
  El::Int h = 0;
  for(unsigned int i=0u; i < m_num_img_srcs; ++i) {
    El::View(X_v[i], X, El::IR(h, h + m_image_linearized_size), El::IR(mb_idx, mb_idx + 1));
    h = h + m_image_linearized_size;
  }
  return X_v;
}

bool data_reader_multi_images::fetch_datum(CPUMat& X, int data_id, int mb_idx) {
  int tid = m_io_thread_pool->get_local_thread_id();
  std::vector<CPUMat> X_v = create_datum_views(X, mb_idx);

  const img_src_t& img_src = m_image_list[data_id].first;
  for(size_t i=0u; i < m_num_img_srcs; ++i) {
    int width=0, height=0, img_type=0;
    const std::string imagepath = get_file_dir() + img_src[i];
    bool ret = true;
    ret = lbann::image_utils::load_image(imagepath, width, height, img_type, *(m_pps[tid]), X_v[i], m_thread_buffer[tid], &m_thread_cv_buffer[tid]);

    if(!ret) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                            + get_type() + ": image_utils::load_image failed to load - "
                            + imagepath);
    }
    if((width * height * CV_MAT_CN(img_type)) != m_image_linearized_size) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " "
                            + get_type() + ": mismatch data size -- either width, height or channel - "
                            + imagepath + " [w,h,c]=[" + std::to_string(width) + "x" + std::to_string(height)
                            + "x" + std::to_string(CV_MAT_CN(img_type)) + "] != " + std::to_string(m_image_linearized_size));
    }
  }
  return true;
}

bool data_reader_multi_images::fetch_label(CPUMat& Y, int data_id, int mb_idx) {
  const label_t label = m_image_list[data_id].second;
  Y.Set(label, mb_idx, 1);
  return true;
}

std::vector<data_reader_multi_images::sample_t> data_reader_multi_images::get_image_list_of_current_mb() const {
  std::vector<sample_t> ret;
  ret.reserve(m_mini_batch_size);

  // for (El::Int i = 0; i < m_indices_fetched_per_mb.Height(); ++i) {
  //   El::Int index = m_indices_fetched_per_mb.Get(i, 0);
  //   ret.push_back(m_image_list[index]);
  // }
  return ret;
}

bool data_reader_multi_images::read_text_stream(std::istream& text_stream,
  std::vector<data_reader_multi_images::sample_t>& list)
{
  std::string line;

  if (!std::getline(text_stream, line)) {
    return false;
  }

  while (text_stream) {
    img_src_t img_srcs;
    {
      std::stringstream sstr(line.c_str());
      for (unsigned int i=0u; i < m_num_img_srcs; ++i) {
        std::string path;
        sstr >> path;
        img_srcs.push_back(path);
      }
      label_t label;
      sstr >> label;
      if (sstr.bad()) {
        return false;
      }
      list.emplace_back(img_srcs, label);
    }
    std::getline(text_stream, line);
  }
  list.shrink_to_fit();
  return true;
}

bool data_reader_multi_images::load_list(const std::string file_name,
  std::vector<data_reader_multi_images::sample_t>& list, const bool fetch_list_at_once)
{
  bool ok = true;

  if (fetch_list_at_once) {
    int tid = omp_get_thread_num();
    ok = load_file(file_name, m_thread_buffer[tid]);
    std::istringstream text_stream(m_thread_buffer[tid].data());
    ok = ok && read_text_stream(text_stream, list);
  } else {
    std::ifstream text_stream(file_name.c_str(), std::ios_base::in);
    ok = read_text_stream(text_stream, list);
  }
  return ok;
}

void data_reader_multi_images::load() {
  m_image_list.clear();
  const std::string image_list_file = get_data_filename();

  bool ok = load_list(image_list_file, m_image_list);
  if (!ok) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: failed to load: " + image_list_file);
  }

  // reset indices
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_image_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);

  select_subset_of_data();
}

}  // namespace lbann
