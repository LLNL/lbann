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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/cnpy_utils.hpp"
#include "lbann/data_readers/opencv_extensions.hpp"
#include "lbann/data_readers/data_reader_jag.hpp"
#include <limits>     // numeric_limits
#include <algorithm>  // max_element
#include <numeric>    // accumulate
#include <functional> // multiplies
#include <type_traits>// is_same

namespace lbann {

data_reader_jag::data_reader_jag(bool shuffle)
  : generic_data_reader(shuffle),
    m_independent({Undefined}), m_dependent({Undefined}),
    m_image_loaded(false), m_scalar_loaded(false),
    m_input_loaded(false), m_num_samples(0u),
    m_linearized_image_size(0u),
    m_linearized_scalar_size(0u),
    m_linearized_input_size(0u),
    m_image_normalization(0u),
    m_image_width(0u), m_image_height(0u),
    m_img_min(std::numeric_limits<data_t>::max()),
    m_img_max(std::numeric_limits<data_t>::min()) {
}


data_reader_jag::~data_reader_jag() {
}


void data_reader_jag::set_independent_variable_type(
  const std::vector< std::vector<data_reader_jag::variable_t> >& independent) {
  if (!independent.empty() && !m_independent.empty() && (m_independent[0] == Undefined)) {
    m_independent.clear();
  }
  for (const auto& group: independent) {
    for (const auto type: group) {
      add_independent_variable_type(type);
    }
  }
}

void data_reader_jag::add_independent_variable_type(
  const data_reader_jag::variable_t independent) {
  if (!(independent == JAG_Image || independent == JAG_Scalar ||
        independent == JAG_Input || independent == Undefined)) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: unrecognized variable type " + std::to_string(static_cast<int>(independent)));
  }
  m_independent.push_back(independent);
}

void data_reader_jag::set_dependent_variable_type(
  const std::vector < std::vector<data_reader_jag::variable_t> >& dependent) {
  if (!dependent.empty() && !m_dependent.empty() && (m_dependent[0] == Undefined)) {
    m_dependent.clear();
  }
  for (const auto& group: dependent) {
    for (const auto type: group) {
      add_dependent_variable_type(type);
    }
  }
}

void data_reader_jag::add_dependent_variable_type(
  const data_reader_jag::variable_t dependent) {
  if (!(dependent == JAG_Image || dependent == JAG_Scalar ||
        dependent == JAG_Input || dependent == Undefined)) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " :: unrecognized variable type " + std::to_string(static_cast<int>(dependent)));
  }
  m_dependent.push_back(dependent);
}


std::vector<data_reader_jag::variable_t> data_reader_jag::get_independent_variable_type() const {
  return m_independent;
}

std::vector<data_reader_jag::variable_t> data_reader_jag::get_dependent_variable_type() const {
  return m_dependent;
}

bool data_reader_jag::is_independent(const variable_t t) const {
  for(const auto i: m_independent) {
    if (i == t) return true;
  }
  return false;
}

bool data_reader_jag::is_dependent(const variable_t t) const {
  for(const auto d: m_dependent) {
    if (d == t) return true;
  }
  return false;
}

bool data_reader_jag::is_used(const variable_t t) const {
  return is_independent(t) || is_dependent(t);
}

void data_reader_jag::set_normalization_mode(int mode) {
  if ((mode < 0) || (2 < mode)) {
    throw lbann_exception("data_reader_jag: invalid normalization mode " +
                          std::to_string(mode));
  }
  m_image_normalization = mode;
}

void data_reader_jag::set_image_dims(const int width, const int height) {
  if ((width > 0) && (height > 0)) { // set and valid
    m_image_width = width;
    m_image_height = height;
  } else if (!((width == 0) && (height == 0))) { // set but not valid
    std::stringstream err;
    err << __FILE__<<" "<<__LINE__
        << " :: data_reader_jag::set_image_dims() invalid image dims";
    throw lbann_exception(err.str());
  }
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
    m_image_width = 0u;
    m_image_height = 0u;
  } else {
    m_linearized_image_size
      = std::accumulate(m_images.shape.begin()+1, m_images.shape.end(),
                        1u, std::multiplies<size_t>());
    if (m_linearized_image_size != static_cast<size_t>(m_image_width*m_image_height)) {
      if ((m_image_width == 0u) && (m_image_height == 0u)) {
        m_image_height = 1;
        m_image_width = static_cast<int>(m_linearized_image_size);
      } else {
        std::stringstream err;
        err << __FILE__<<" "<<__LINE__
            << " :: data_reader_jag::set_linearized_image_size() image size mismatch";
        throw lbann_exception(err.str());
      }
    }
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

size_t data_reader_jag::get_linearized_size(const data_reader_jag::variable_t t) const {
  switch (t) {
    case JAG_Image:
      return m_linearized_image_size;
    case JAG_Scalar:
      return m_linearized_scalar_size;
    case JAG_Input:
      return m_linearized_input_size;
    default: { // includes Undefined case
      throw lbann_exception(std::string("data_reader_jag::get_linearized_size() : ") +
                                        "unknown or undefined variable type");
    }
  }
  return 0u;
}

int data_reader_jag::get_linearized_data_size() const {
  size_t sz = 0u;
  for (const auto t: m_independent) {
    if (t == Undefined) {
      continue;
    }
    sz += get_linearized_size(t);
  }
  return static_cast<int>(sz);
}

int data_reader_jag::get_linearized_response_size() const {
  size_t sz = 0u;
  for (const auto t: m_dependent) {
    if (t == Undefined) {
      continue;
    }
    sz += get_linearized_size(t);
  }
  return static_cast<int>(sz);
}

std::vector<size_t> data_reader_jag::get_linearized_data_sizes() const {
  std::vector<size_t> all_dim;
  all_dim.reserve(m_independent.size());
  for (const auto t: m_independent) {
    if (t == Undefined) {
      continue;
    }
    all_dim.push_back(get_linearized_size(t));
  }
  return all_dim;
}

std::vector<size_t> data_reader_jag::get_linearized_response_sizes() const {
  std::vector<size_t> all_dim;
  all_dim.reserve(m_dependent.size());
  for (const auto t: m_dependent) {
    if (t == Undefined) {
      continue;
    }
    all_dim.push_back(get_linearized_size(t));
  }
  return all_dim;
}


const std::vector<int> data_reader_jag::get_dims(const data_reader_jag::variable_t t) const {
  switch (t) {
    case JAG_Image:
      return {1, m_image_height, m_image_width};
      //return {static_cast<int>(m_linearized_image_size)};
    case JAG_Scalar:
      return {static_cast<int>(m_linearized_scalar_size)};
    case JAG_Input:
      return {static_cast<int>(m_linearized_input_size)};
    default: {
      throw lbann_exception(std::string("data_reader_jag::get_dims() : ") +
                                        "unknown or undefined variable type");
    }
  }
  return {};
}

const std::vector<int> data_reader_jag::get_data_dims() const {
  std::vector<int> all_dim;
  for (const auto t: m_independent) {
    if (t == Undefined) {
      continue;
    }
    const std::vector<int> ld = get_dims(t);
    all_dim.insert(all_dim.end(), ld.begin(), ld.end());
  }
  return all_dim;
}


void data_reader_jag::load() {
  if (m_gan_labelling) {
    m_num_labels=2;
  }
  if (is_master()) {
    std::cout << "JAG load GAN m_gan_labelling : label_value "
              << m_gan_labelling <<" : " << m_gan_label_value << std::endl;
  }

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
  if ((m_independent.empty() || (m_independent[0] == Undefined)) &&
      !m_dependent.empty() && (m_dependent[0] == Undefined)) {
    throw lbann_exception("data_reader_jag: no type of variables to load is defined.");
  }
  if (is_used(JAG_Image) && !image_file.empty() && !check_if_file_exists(image_file)) {
    throw lbann_exception("data_reader_jag: failed to load " + image_file);
  }
  if (is_used(JAG_Scalar) && !scalar_file.empty() && !check_if_file_exists(scalar_file)) {
    throw lbann_exception("data_reader_jag: failed to load " + scalar_file);
  }
  if (is_used(JAG_Input) && !input_file.empty() && !check_if_file_exists(input_file)) {
    throw lbann_exception("data_reader_jag: failed to load " + input_file);
  }

  m_num_samples = 0u;

  // read in only those that will be used
  if (is_used(JAG_Image) && !image_file.empty()) {
    m_images  = cnpy::npy_load(image_file);
    if (first_n > 0u) { // to use only first_n samples
      cnpy_utils::shrink_to_fit(m_images, first_n);
    }
    m_image_loaded = true;
    set_linearized_image_size();
  }
  if (is_used(JAG_Scalar) && !scalar_file.empty()) {
    m_scalars = cnpy::npy_load(scalar_file);
    if (first_n > 0u) { // to use only first_n samples
      cnpy_utils::shrink_to_fit(m_scalars, first_n);
    }
    m_scalar_loaded = true;
    set_linearized_scalar_size();
  }
  if (is_used(JAG_Input) && !input_file.empty()) {
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
  if (m_image_loaded) {
    m_img_min = get_image_min();
    m_img_max = get_image_max();
    if (m_img_min == m_img_max) {
      throw lbann_exception("data_reader_jag: no_variation in data");
    }
    normalize_image();
  }
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
  } else {
    if (is_used(JAG_Image)) {
      ok = ok && m_image_loaded;
    }
    if (is_used(JAG_Scalar)) {
      ok = ok && m_scalar_loaded;
    }
    if (is_used(JAG_Input)) {
      ok = ok && m_input_loaded;
    }
  }

  return ok;
}


std::string data_reader_jag::get_description() const {
  using std::string;
  using std::to_string;
  string ret = string("data_reader_jag:\n")
    + " - images: "   + cnpy_utils::show_shape(m_images) + "\n"
    + " - scalars: "  + cnpy_utils::show_shape(m_scalars) + "\n"
    + " - inputs: "   + cnpy_utils::show_shape(m_inputs) + "\n";
  if (m_image_loaded) {
    ret += " - min pixel value: " + to_string(m_img_min) + "\n"
         + " - max pixel value: " + to_string(m_img_max) + "\n"
         + " - image width " + to_string(m_image_width) + "\n"
         + " - image height " + to_string(m_image_height) + "\n"
         + " - image normalization: " + to_string(m_image_normalization) + "\n";
  }
  return ret;
}


void data_reader_jag::normalize_image() {
  if (!m_image_loaded) {
    return;
  }
  using depth_t = cv_image_type<data_t>;
  const int type_code = depth_t::T(1u);

  if (m_image_normalization == 1) {
    data_t* const ptr = get_image_ptr(0);
    // Present the entire image data as a single image
    // and normalize it once and for all
    cv::Mat img(m_num_samples, m_linearized_image_size,
                type_code, reinterpret_cast<void*>(ptr));
    cv::normalize(img, img, 0.0, 1.0, cv::NORM_MINMAX);
  } else if (m_image_normalization == 2) {
    // normalize each image independently
    for (size_t i=0u; i < m_num_samples; ++i) {
      data_t* const ptr = get_image_ptr(i);
      cv::Mat img(1, m_linearized_image_size,
                  type_code, reinterpret_cast<void*>(ptr));
      cv::normalize(img, img, 0.0, 1.0, cv::NORM_MINMAX);
    }
  } else if (m_image_normalization != 0) {
    throw lbann_exception("data_reader_jag: invalid normalization mode " +
                          std::to_string(m_image_normalization));
  }
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

  cv::Mat img;
  if (std::is_same<data_t, DataType>::value) {
    img = img_org.clone();
  } else {
    img_org.convertTo(img, cv_image_type<DataType>::T(1u));
  }
  return img.reshape(0, m_image_height);
}

data_reader_jag::data_t data_reader_jag::get_image_max() const {
  if (!m_image_loaded) {
    return std::numeric_limits<data_t>::min();
  }
  const data_t* ptr = get_image_ptr(0);
  const size_t tot_num_pixels = m_images.shape[0]*m_linearized_image_size;
  return *std::max_element(ptr, ptr + tot_num_pixels);
}

data_reader_jag::data_t data_reader_jag::get_image_min() const {
  if (!m_image_loaded) {
    return std::numeric_limits<data_t>::max();
  }
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


std::vector<CPUMat>
data_reader_jag::create_datum_views(CPUMat& X, const std::vector<size_t>& sizes, const int mb_idx) const {
  std::vector<CPUMat> X_v(sizes.size());
  El::Int h = 0;
  for(size_t i=0u; i < sizes.size(); ++i) {
    const El::Int h_end =  h + static_cast<El::Int>(sizes[i]);
    El::View(X_v[i], X, El::IR(h, h_end), El::IR(mb_idx, mb_idx + 1));
    h = h_end;
  }
  return X_v;
}

bool data_reader_jag::fetch(CPUMat& X, int data_id, int mb_idx,
  const data_reader_jag::variable_t vt, const std::string tag) {
  switch (vt) {
    case JAG_Image: {
      const data_t* ptr = get_image_ptr(data_id);
      set_minibatch_item<data_t>(X, mb_idx, ptr, m_linearized_image_size);
      break;
    }
    case JAG_Scalar: {
      const scalar_t* ptr = get_scalar_ptr(data_id);
      set_minibatch_item<scalar_t>(X, mb_idx, ptr, m_linearized_scalar_size);
      break;
    }
    case JAG_Input: {
      const input_t* ptr = get_input_ptr(data_id);
      set_minibatch_item<input_t>(X, mb_idx, ptr, m_linearized_input_size);
      break;
    }
    default: { // includes Undefined case
      throw lbann_exception(std::string("data_reader_jag::fetch_") + tag +
                            "() : unknown or undefined variable type (" +
                            std::to_string(static_cast<int>(vt)) + ')');
    }
  }
  return true;
}

bool data_reader_jag::fetch_datum(CPUMat& X, int data_id, int mb_idx) {
  std::vector<size_t> sizes = get_linearized_data_sizes();
  std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);
  bool ok = true;
  for(size_t i = 0u; ok && (i < X_v.size()); ++i) {
    ok = fetch(X_v[i], data_id, 0, m_independent[i], "datum");
  }
  return ok;
}

bool data_reader_jag::fetch_response(CPUMat& X, int data_id, int mb_idx) {
  std::vector<size_t> sizes = get_linearized_response_sizes();
  std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);
  bool ok = true;
  for(size_t i = 0u; ok && (i < X_v.size()); ++i) {
    ok = fetch(X_v[i], data_id, 0, m_dependent[i], "response");
  }
  return ok;
}

bool data_reader_jag::fetch_label(CPUMat& Y, int data_id, int mb_idx) {
  if(m_gan_label_value) Y.Set(m_gan_label_value,mb_idx,1); //fake sample is set to 1; adversarial model
  else { //fake sample (second half of minibatch is set to 0;discriminator model
    //mb_idx < (m_mb_size/2) ? Y.Set(1,mb_idx,1) : Y.Set(m_gan_label_value,mb_idx,1);
    mb_idx < (get_current_mini_batch_size()/2) ? Y.Set(1,mb_idx,1) : Y.Set(m_gan_label_value,mb_idx,1);
  }
  //Y.Set(m_gan_label_value, mb_idx, 1);
  return true;
}

void data_reader_jag::save_image(Mat& pixels, const std::string filename, bool do_scale) {
  internal_save_image(pixels, filename, m_image_height, m_image_width, 1, do_scale);
}

} // end of namespace lbann
