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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/cnpy_utils.hpp"
#include "lbann/data_readers/opencv_extensions.hpp"
#include "lbann/data_readers/data_reader_jag.hpp"
#include <limits>     // numeric_limits
#include <algorithm>  // max_element
#include <numeric>    // accumulate
#include <functional> // multiplies

namespace lbann {

data_reader_jag::data_reader_jag(bool shuffle)
  : generic_data_reader(shuffle), m_model_mode(Inverse),
    m_image_loaded(false), m_scalar_loaded(false),
    m_input_loaded(false), m_num_samples(0u),
    m_linearized_image_size(0u),
    m_linearized_scalar_size(0u),
    m_linearized_input_size(0u),
    m_img_min(std::numeric_limits<data_t>::max()), 
    m_img_max(std::numeric_limits<data_t>::min()),
    m_alpha(1.0), m_beta(0.0) {
}


data_reader_jag::~data_reader_jag() {
}

void data_reader_jag::set_model_mode(const model_mode_t mm) {
  if (static_cast<int>(AutoS) < static_cast<int>(mm)) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: unrecognized mode " + std::to_string(static_cast<int>(mm)));
  }
  m_model_mode = mm;
}

size_t data_reader_jag::get_num_samples() const {
  return m_num_samples;
}

size_t data_reader_jag::get_linearized_image_size() const {
  return m_linearized_image_size;
}

size_t data_reader_jag::get_linearized_scalar_size() const {
  return m_linearized_scalar_size;
}

size_t data_reader_jag::get_linearized_input_size() const {
  return m_linearized_input_size;
}

void data_reader_jag::set_linearized_image_size() {
  if (!m_image_loaded) {
    m_linearized_image_size = 0u;
  } else {
    m_linearized_image_size
      = std::accumulate(m_images.shape.begin()+1, m_images.shape.end(),
                        1u, std::multiplies<size_t>());
  }
}

void data_reader_jag::set_linearized_scalar_size() {
  if (!m_scalar_loaded) {
    m_linearized_scalar_size = 0u;
  } else {
    m_linearized_scalar_size
      = std::accumulate(m_scalars.shape.begin()+1, m_scalars.shape.end(),
                        1u, std::multiplies<size_t>());
  }
}

void data_reader_jag::set_linearized_input_size() {
  if (!m_input_loaded) {
    m_linearized_input_size = 0u;
  } else {
    m_linearized_input_size
      = std::accumulate(m_inputs.shape.begin()+1, m_inputs.shape.end(),
                        1u, std::multiplies<size_t>());
  }
}

int data_reader_jag::get_linearized_data_size() const {
  switch (m_model_mode) {
    case Inverse:
      return static_cast<int>(m_linearized_image_size);
    case AutoI:
      return static_cast<int>(m_linearized_image_size);
    case AutoS:
      return static_cast<int>(m_linearized_scalar_size);
    default: {
      throw lbann_exception("data_reader_jag::get_linearized_data_size() : unknown mode");
    }
  }
  return 0;
}

int data_reader_jag::get_linearized_response_size() const {
  switch (m_model_mode) {
    case Inverse:
      return static_cast<int>(m_linearized_input_size);
    case AutoI:
      return static_cast<int>(m_linearized_image_size);
    case AutoS:
      return static_cast<int>(m_linearized_scalar_size);
    default: {
      throw lbann_exception("data_reader_jag::get_linearized_response_size() : unknown mode");
    }
  }
  return 0;
}

const std::vector<int> data_reader_jag::get_data_dims() const {
  switch (m_model_mode) {
    case Inverse:
      //return {1, 50, 50};
      return {static_cast<int>(m_linearized_image_size)};
    case AutoI:
      return {static_cast<int>(m_linearized_image_size)};
    case AutoS:
      return {static_cast<int>(m_linearized_scalar_size)};
    default: {
      throw lbann_exception("data_reader_jag::get_data_dims() : unknown mode");
    }
  }
  return {};
}


void data_reader_jag::load() {
  const std::string data_dir = add_delimiter(get_file_dir());
  const std::string namestr = get_data_filename();
  std::vector<std::string> file_names = get_tokens(namestr);
  if (file_names.size() != 3u) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: unexpected number of files " + std::to_string(file_names.size()));
  }

  for(auto& str : file_names) {
    str = data_dir + str;
  }
  load(file_names[0], file_names[1], file_names[2], m_first_n);

  size_t num_samples = get_num_samples();

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


void data_reader_jag::load(const std::string image_file,
            const std::string scalar_file,
            const std::string input_file,
            const size_t first_n) {
  if (!image_file.empty() && !check_if_file_exists(image_file)) {
    throw lbann_exception("data_reader_jag: failed to load " + image_file);
  }
  if (!scalar_file.empty() && !check_if_file_exists(scalar_file)) {
    throw lbann_exception("data_reader_jag: failed to load " + scalar_file);
  }
  if (!input_file.empty() && !check_if_file_exists(input_file)) {
    throw lbann_exception("data_reader_jag: failed to load " + input_file);
  }

  
  m_num_samples = 0u;

  if (!image_file.empty()) {
    m_images  = cnpy::npy_load(image_file);
    if (first_n > 0u) { // to use only first_n samples
      cnpy_utils::shrink_to_fit(m_images, first_n);
    }
    m_image_loaded = true;
    set_linearized_image_size();
  }
  if (!scalar_file.empty()) {
    m_scalars = cnpy::npy_load(scalar_file);
    if (first_n > 0u) { // to use only first_n samples
      cnpy_utils::shrink_to_fit(m_scalars, first_n);
    }
    m_scalar_loaded = true;
    set_linearized_scalar_size();
  }
  if (!input_file.empty()) {
    m_inputs  = cnpy::npy_load(input_file);
    if (first_n > 0u) { // to use only first_n samples
      cnpy_utils::shrink_to_fit(m_inputs, first_n);
    }
    m_input_loaded = true;
    set_linearized_input_size();
  }
  
  size_t num_samples = 0u;
  bool ok = check_data(num_samples);

  if (!ok) {
    get_description();
    throw lbann_exception("data_reader_jag: loaded data format not consistent");
  }
  m_num_samples = num_samples;
  m_img_min = get_image_min();
  m_img_max = get_image_max();
  if (m_img_min == m_img_max) {
    throw lbann_exception("data_reader_jag: no_variation in data");
  }
  m_alpha = 1.0/(m_img_max - m_img_min);
  m_beta = m_img_min/(m_img_min - m_img_max);
}


bool data_reader_jag::check_data(size_t& num_samples) const {
  bool ok = true;
  num_samples = 0u;
  if (ok && m_image_loaded) {
    ok = (m_linearized_image_size > 0u) && (m_images.word_size == sizeof(data_t));
    if (!ok) {
      std::cerr << "m_images.shape.size() = " << m_images.shape.size() << std::endl
                << "m_linearized_image_size = " << m_linearized_image_size << std::endl
                << "m_images.word_size = " << m_images.word_size << std::endl
                << "sizeof(data_t) = " << sizeof(data_t) << std::endl;
      return false;
    }
    num_samples = m_images.shape[0];
  }
  if (ok && m_scalar_loaded) {
    ok = (m_linearized_scalar_size > 0u) && (m_scalars.word_size == sizeof(scalar_t));
    if (!ok) {
      std::cerr << "m_scalars.shape.size() = " << m_scalars.shape.size() << std::endl
                << "m_linearized_scalar_size = " << m_linearized_scalar_size << std::endl
                << "m_scalars.word_size = " << m_scalars.word_size << std::endl
                << "sizeof(scalar_t) = " << sizeof(scalar_t) << std::endl;
      return false;
    }
    if (num_samples > 0u) {
      ok = ok && (num_samples == m_scalars.shape[0]);
    } else {
      num_samples = m_scalars.shape[0];
    }
  }
  if (ok && m_input_loaded) {
    ok = (m_linearized_input_size > 0u) && (m_inputs.word_size == sizeof(input_t));
    if (!ok) {
      std::cerr << "m_inputs.shape.size() = " << m_inputs.shape.size() << std::endl
                << "m_linearized_input_size = " << m_linearized_input_size << std::endl
                << "m_inputs.word_size = " << m_inputs.word_size << std::endl
                << "sizeof(input_t) = " << sizeof(input_t) << std::endl;
      return false;
    }
    if (num_samples > 0u) {
      ok = ok && (num_samples == m_inputs.shape[0]);
    } else {
      num_samples = m_inputs.shape[0];
    }
  }
  if (!ok) {
    num_samples = 0u;
  }

  return ok;
}


std::string data_reader_jag::get_description() const {
  using std::string;
  using std::to_string;
  string ret = string("data_reader_jag:\n")
    + " - mode: " + to_string(static_cast<int>(m_model_mode)) + "\n"
    + " - images: "   + cnpy_utils::show_shape(m_images) + "\n"
    + " - scalars: "  + cnpy_utils::show_shape(m_scalars) + "\n"
    + " - inputs: "   + cnpy_utils::show_shape(m_inputs) + "\n"
    + " - min pixel value: " + to_string(m_img_min) + "\n";
    + " - max pixel value: " + to_string(m_img_max) + "\n";
  return ret;
}


data_reader_jag::data_t* data_reader_jag::get_image_ptr(const size_t i) const {
  return (m_image_loaded? cnpy_utils::data_ptr<data_t>(m_images, {i}) : nullptr);
}

cv::Mat data_reader_jag::get_image(const size_t i) const {
  using InputBuf_T = cv_image_type<data_t>;
  
  data_t* const ptr = get_image_ptr(i);
  if (ptr == nullptr) {
    return cv::Mat();
  }
  // Construct a zero copying view to data
  const cv::Mat img_org(m_linearized_image_size, 1, InputBuf_T::T(1u),
                        reinterpret_cast<void*>(ptr));

  // Normalzie pixel values to the range [0.0 1.0].
  cv::Mat img;
#if 0 // sample-wide normalization
  cv::normalize(img_org, img, 0.0, 1.0, cv::NORM_MINMAX, cv_image_type<DataType>::T(1u));
#else // dataset-wide normalization
  img_org.convertTo(img, cv_image_type<DataType>::T(1u), m_alpha, m_beta);
#endif
  return img;
}

data_reader_jag::data_t data_reader_jag::get_image_max() const {
  const data_t* ptr = get_image_ptr(0);
  const size_t tot_num_pixels = m_images.shape[0]*m_linearized_image_size;
  return *std::max_element(ptr, ptr + tot_num_pixels);
}

data_reader_jag::data_t data_reader_jag::get_image_min() const {
  const data_t* ptr = get_image_ptr(0);
  const size_t tot_num_pixels = m_images.shape[0]*m_linearized_image_size;
  return *std::min_element(ptr, ptr + tot_num_pixels);
}

data_reader_jag::scalar_t* data_reader_jag::get_scalar_ptr(const size_t i) const {
  return (m_scalar_loaded? cnpy_utils::data_ptr<scalar_t>(m_scalars, {i}) : nullptr);
}

std::vector<DataType> data_reader_jag::get_scalar(const size_t i) const {
  const scalar_t* const ptr = get_scalar_ptr(i);
  if (ptr == nullptr) {
    return {};
  }
  std::vector<DataType> ret(m_linearized_scalar_size);
  for (size_t j = 0u; j < m_linearized_scalar_size; ++j) {
    ret[j] = static_cast<DataType>(ptr[j]);
  }
  return ret;
}

data_reader_jag::input_t* data_reader_jag::get_input_ptr(const size_t i) const {
  return (m_input_loaded? cnpy_utils::data_ptr<input_t>(m_inputs, {i}) : nullptr);
}

std::vector<DataType> data_reader_jag::get_input(const size_t i) const {
  const input_t* const ptr = get_input_ptr(i);
  if (ptr == nullptr) {
    return {};
  }
  std::vector<DataType> ret(m_linearized_input_size);
  for (size_t j = 0u; j < m_linearized_input_size; ++j) {
    ret[j] = static_cast<DataType>(ptr[j]);
  }
  return ret;
}


bool data_reader_jag::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {
  switch (m_model_mode) {
    case Inverse: {
      const data_t* ptr = get_image_ptr(data_id);
      set_minibatch_item<data_t>(X, mb_idx, ptr, m_linearized_image_size);
      break;
    }
    case AutoI: {
      const data_t* ptr = get_image_ptr(data_id);
      set_minibatch_item<data_t>(X, mb_idx, ptr, m_linearized_image_size);
      break;
    }
    case AutoS: {
      const scalar_t* ptr = get_scalar_ptr(data_id);
      set_minibatch_item<scalar_t>(X, mb_idx, ptr, m_linearized_scalar_size);
      break;
    }
    default: {
      throw lbann_exception("data_reader_jag::fetch_datum() : unknown mode");
    }
  }
  return true;
}

bool data_reader_jag::fetch_response(Mat& Y, int data_id, int mb_idx, int tid) {
  switch (m_model_mode) {
    case Inverse: {
      const input_t* ptr = get_input_ptr(data_id);
      set_minibatch_item<input_t>(Y, mb_idx, ptr, m_linearized_input_size);
      break;
    }
    case AutoI: {
      const data_t* ptr = get_image_ptr(data_id);
      set_minibatch_item<data_t>(Y, mb_idx, ptr, m_linearized_image_size);
      break;
    }
    case AutoS: {
      const scalar_t* ptr = get_scalar_ptr(data_id);
      set_minibatch_item<scalar_t>(Y, mb_idx, ptr, m_linearized_scalar_size);
      break;
    }
    default: {
      throw lbann_exception("data_reader_jag::fetch_response() : unknown mode");
    }
  }
  return true;
}
} // end of namespace lbann
