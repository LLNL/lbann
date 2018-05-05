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

#ifndef _JAG_OFFLINE_TOOL_MODE_
#include "lbann/data_readers/data_reader_jag_conduit.hpp"
#include "lbann/utils/file_utils.hpp" // for add_delimiter() in load()
#else
#include "data_reader_jag_conduit.hpp"
#endif // _JAG_OFFLINE_TOOL_MODE_

#ifdef LBANN_HAS_CONDUIT
#include "lbann/data_readers/opencv_extensions.hpp"
#include <limits>     // numeric_limits
#include <algorithm>  // max_element
#include <numeric>    // accumulate
#include <functional> // multiplies
#include <type_traits>// is_same
#include <set>
#include <map>

// This macro may be moved to a global scope
#define _THROW_LBANN_EXCEPTION_(_CLASS_NAME_,_MSG_) { \
  std::stringstream err; \
  err << __FILE__ << ' '  << __LINE__ << " :: " \
      << _CLASS_NAME_ << "::" << _MSG_; \
  throw lbann_exception(err.str()); \
}

// This comes after all the headers, and is only visible within the current implementation file.
// To make sure, we put '#undef _CN_' at the end of this file
#define _CN_ "data_reader_jag_conduit"

namespace lbann {

data_reader_jag_conduit::data_reader_jag_conduit(bool shuffle)
  : generic_data_reader(shuffle),
    m_independent(JAG_Image), m_dependent(JAG_Input),
    m_linearized_image_size(0u),
    m_num_views(2u),
    m_image_width(0u), m_image_height(0u),
    m_is_data_loaded(false) {
}


data_reader_jag_conduit::~data_reader_jag_conduit() {
}


const conduit::Node& data_reader_jag_conduit::get_conduit_node(const std::string key) const {
  return m_data[key];
}


void data_reader_jag_conduit::set_independent_variable_type(
  const data_reader_jag_conduit::variable_t independent) {
  if (!(independent == JAG_Image || independent == JAG_Scalar || independent == JAG_Input)) {
    _THROW_LBANN_EXCEPTION_(_CN_, "unrecognized independent variable type ");
  }
  m_independent = independent;
}

void data_reader_jag_conduit::set_dependent_variable_type(
  const data_reader_jag_conduit::variable_t dependent) {
  if (!(dependent == JAG_Image || dependent == JAG_Scalar || dependent == JAG_Input)) {
    _THROW_LBANN_EXCEPTION_(_CN_, "unrecognized dependent variable type ");
  }
  m_dependent = dependent;
}

data_reader_jag_conduit::variable_t
data_reader_jag_conduit::get_independent_variable_type() const {
  return m_independent;
}

data_reader_jag_conduit::variable_t
data_reader_jag_conduit::get_dependent_variable_type() const {
  return m_dependent;
}

void data_reader_jag_conduit::set_image_dims(const int width, const int height) {
  if ((width > 0) && (height > 0)) { // set and valid
    m_image_width = width;
    m_image_height = height;
  } else if (!((width == 0) && (height == 0))) { // set but not valid
    _THROW_LBANN_EXCEPTION_(_CN_, "set_image_dims() : invalid image dims");
  }
}

void data_reader_jag_conduit::set_scalar_choices(const std::vector<std::string>& keys) {
  m_scalar_keys = keys;
}

void data_reader_jag_conduit::set_all_scalar_choices() {
  if (!check_sample_id(0)) {
    return;
  }
  const conduit::Node & n_scalar = get_conduit_node("0/outputs/scalars");
  m_scalar_keys.reserve(n_scalar.number_of_children());
  conduit::NodeConstIterator itr = n_scalar.children();
  while (itr.has_next()) {
    itr.next();
    m_scalar_keys.push_back(itr.name());
  }
}

const std::vector<std::string>& data_reader_jag_conduit::get_scalar_choices() const {
  return m_scalar_keys;
}


void data_reader_jag_conduit::set_input_choices(const std::vector<std::string>& keys) {
  m_input_keys = keys;
}

void data_reader_jag_conduit::set_all_input_choices() {
  if (!check_sample_id(0)) {
    return;
  }
  const conduit::Node & n_input = get_conduit_node("0/inputs");
  m_input_keys.reserve(n_input.number_of_children());
  conduit::NodeConstIterator itr = n_input.children();
  while (itr.has_next()) {
    itr.next();
    m_input_keys.push_back(itr.name());
  }
}

const std::vector<std::string>& data_reader_jag_conduit::get_input_choices() const {
  return m_input_keys;
}


void data_reader_jag_conduit::set_num_views() {
  if (!check_sample_id(0)) {
    return;
  }

  conduit::NodeConstIterator itr = get_conduit_node("0/outputs/images").children();

  using view_set = std::set< std::pair<float, float> >;
  view_set views;

  while (itr.has_next()) {
    const conduit::Node & n_image = itr.next();
    std::stringstream sstr(n_image["view"].as_string());
    double c1, c2;
    std::string tmp;
    sstr >> tmp >> c1 >> c2;

    views.insert(std::make_pair(c1, c2));
  }

  m_num_views = views.size();
  if (m_num_views == 0u) {
    m_num_views = 1u;
  }
}

void data_reader_jag_conduit::set_linearized_image_size() {
  if (!check_sample_id(0)) {
    m_linearized_image_size = 0u;
    return;
  }
  const conduit::Node & n_imageset = get_conduit_node("0/outputs/images");
  if (static_cast<size_t>(n_imageset.number_of_children()) == 0u) {
    m_linearized_image_size = 0u;
    return;
  }
  const conduit::Node & n_image = get_conduit_node("0/outputs/images/0/emi");
  conduit::float64_array emi = n_image.value();
  m_linearized_image_size = emi.number_of_elements()*get_num_views();
}

void data_reader_jag_conduit::check_image_size() {
  if (m_linearized_image_size != static_cast<size_t>(m_image_width*m_image_height*get_num_views())) {
    if ((m_image_width == 0u) && (m_image_height == 0u)) {
      m_image_height = 1;
      m_image_width = static_cast<int>(m_linearized_image_size/get_num_views());
    } else {
      _THROW_LBANN_EXCEPTION_(_CN_, "check_image_size() : image size mismatch");
    }
  }
}


void data_reader_jag_conduit::check_scalar_keys() {
  if (!check_sample_id(0)) {
    m_scalar_keys.clear();
    return;
  }

  const conduit::Node & n_scalar = get_conduit_node("0/outputs/scalars");
  conduit::NodeConstIterator itr = n_scalar.children();
  size_t num_found = 0u;
  std::vector<bool> found(m_scalar_keys.size(), false);
  std::set<std::string> keys_conduit;

  while (itr.has_next()) {
    itr.next();
    keys_conduit.insert(itr.name());
  }

  for (size_t i=0u; i < m_scalar_keys.size(); ++i) {
    std::set<std::string>::const_iterator it = keys_conduit.find(m_scalar_keys[i]);
    if (it != keys_conduit.end()) {
      num_found ++;
      found[i] = true;
    }
  }

  if (num_found != m_scalar_keys.size()) {
    std::string msg = "keys not found:";
    for (size_t i=0u; i < m_scalar_keys.size(); ++i) {
      if (!found[i]) {
        msg += ' ' + m_scalar_keys[i];
      }
    }
    _THROW_LBANN_EXCEPTION_(_CN_, "check_scalar_keys() : " + msg);
  }
}


void data_reader_jag_conduit::check_input_keys() {
  if (!check_sample_id(0)) {
    m_input_keys.clear();
    return;
  }

  const conduit::Node & n_input = get_conduit_node("0/inputs");
  conduit::NodeConstIterator itr = n_input.children();
  size_t num_found = 0u;
  std::vector<bool> found(m_input_keys.size(), false);
  std::set<std::string> keys_conduit;

  while (itr.has_next()) {
    itr.next();
    keys_conduit.insert(itr.name());
  }

  for (size_t i=0u; i < m_input_keys.size(); ++i) {
    std::set<std::string>::const_iterator it = keys_conduit.find(m_input_keys[i]);
    if (it != keys_conduit.end()) {
      num_found ++;
      found[i] = true;
    }
  }

  if (num_found != m_input_keys.size()) {
    std::string msg = "keys not found:";
    for (size_t i=0u; i < m_input_keys.size(); ++i) {
      if (!found[i]) {
        msg += ' ' + m_input_keys[i];
      }
    }
    _THROW_LBANN_EXCEPTION_(_CN_, "check_input_keys() : " + msg);
  }
}


#ifndef _JAG_OFFLINE_TOOL_MODE_
void data_reader_jag_conduit::load() {
  const std::string data_dir = add_delimiter(get_file_dir());
  const std::string conduit_file_name = get_data_filename();

  load_conduit(data_dir + conduit_file_name);

  if (m_first_n > 0) {
    _THROW_LBANN_EXCEPTION_(_CN_, "load() does not support first_n feature.");
  }

  // reset indices
  m_shuffled_indices.resize(get_num_samples());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);

  select_subset_of_data();
}
#endif // _JAG_OFFLINE_TOOL_MODE_

void data_reader_jag_conduit::load_conduit(const std::string conduit_file_path) {
  conduit::relay::io::load(conduit_file_path, "hdf5", m_data);

  set_num_views();
  set_linearized_image_size();
  check_image_size();
  check_scalar_keys();
  check_input_keys();

  m_is_data_loaded = true;
}


size_t data_reader_jag_conduit::get_num_samples() const {
  return static_cast<size_t>(m_data.number_of_children());
}

unsigned int data_reader_jag_conduit::get_num_views() const {
  return m_num_views;
}

size_t data_reader_jag_conduit::get_linearized_image_size() const {
  return m_linearized_image_size;
}

size_t data_reader_jag_conduit::get_linearized_scalar_size() const {
  return m_scalar_keys.size();
}

size_t data_reader_jag_conduit::get_linearized_input_size() const {
  return m_input_keys.size();
}


int data_reader_jag_conduit::get_linearized_data_size() const {
  switch (m_independent) {
    case JAG_Image:
      return static_cast<int>(get_linearized_image_size());
    case JAG_Scalar:
      return static_cast<int>(get_linearized_scalar_size());
    case JAG_Input:
      return static_cast<int>(get_linearized_input_size());
    default: {
      _THROW_LBANN_EXCEPTION_(_CN_, "get_linearized_data_size() : unknown variable type");
    }
  }
  return 0;
}

int data_reader_jag_conduit::get_linearized_response_size() const {
  switch (m_dependent) {
    case JAG_Image:
      return static_cast<int>(get_linearized_image_size());
    case JAG_Scalar:
      return static_cast<int>(get_linearized_scalar_size());
    case JAG_Input:
      return static_cast<int>(get_linearized_input_size());
    default: {
      _THROW_LBANN_EXCEPTION_(_CN_, "get_linearized_response_size() : unknown variable type");
    }
  }
  return 0;
}

const std::vector<int> data_reader_jag_conduit::get_data_dims() const {
  switch (m_independent) {
    case JAG_Image:
      return {static_cast<int>(get_num_views()), m_image_height, m_image_width};
      //return {static_cast<int>(get_linearized_image_size())};
    case JAG_Scalar:
      return {static_cast<int>(get_linearized_scalar_size())};
    case JAG_Input:
      return {static_cast<int>(get_linearized_input_size())};
    default: {
      _THROW_LBANN_EXCEPTION_(_CN_, "get_data_dims() : unknown variable type");
    }
  }
  return {};
}


std::string data_reader_jag_conduit::get_description() const {
  using std::string;
  using std::to_string;
  string ret = string("data_reader_jag_conduit:\n")
    + " - independent: " + to_string(static_cast<int>(m_independent)) + "\n"
    + " - dependent: " + to_string(static_cast<int>(m_dependent)) + "\n"
    + " - images: "   + to_string(m_num_views) + 'x'
                      + to_string(m_image_width) + 'x'
                      + to_string(m_image_height) + "\n"
    + " - scalars: "  + to_string(get_linearized_scalar_size()) + "\n"
    + " - inputs: "   + to_string(get_linearized_input_size()) + "\n";
  return ret;
}


bool data_reader_jag_conduit::check_sample_id(const size_t sample_id) const {
  return (static_cast<conduit_index_t>(sample_id) < m_data.number_of_children());
}


std::vector<int> data_reader_jag_conduit::choose_image_near_bang_time(const size_t sample_id) const {
  using view_map = std::map<std::pair<float, float>, std::pair<int, double> >;

  conduit::NodeConstIterator itr = get_conduit_node(std::to_string(sample_id) + "/outputs/images").children();
  view_map near_bang_time;
  int idx = 0;

  while (itr.has_next()) {
    const conduit::Node & n_image = itr.next();
    std::stringstream sstr(n_image["view"].as_string());
    double c1, c2;
    std::string tmp;
    sstr >> tmp >> c1 >> c2;
    const double t = n_image["time"].value();
    const double t_abs = std::abs(t);

    view_map::iterator it = near_bang_time.find(std::make_pair(c1, c2));

    if (it == near_bang_time.end()) {
      near_bang_time.insert(std::make_pair(std::make_pair(c1, c2), std::make_pair(idx, t_abs)));
    } else if ((it->second).second > t) { // currently ignore tie
      it->second = std::make_pair(idx, t_abs);
    }

    idx++;
  }

  std::vector<int> img_indices;
  img_indices.reserve(near_bang_time.size());
  for(const auto& view: near_bang_time) {
    img_indices.push_back(view.second.first);
  }
  return img_indices;
}

std::vector< std::pair<size_t, const data_reader_jag_conduit::ch_t*> >
data_reader_jag_conduit::get_image_ptrs(const size_t sample_id) const {
  if (!check_sample_id(sample_id)) {
    _THROW_LBANN_EXCEPTION_(_CN_, "get_images() : invalid sample index");
  }
  std::vector<int> img_indices = choose_image_near_bang_time(sample_id);
  std::vector< std::pair<size_t, const ch_t*> >image_ptrs;
  image_ptrs.reserve(img_indices.size());

  for (const auto idx: img_indices) {
    std::string img_key = std::to_string(sample_id) + "/outputs/images/" + std::to_string(idx) + "/emi";
    const conduit::Node & n_image = get_conduit_node(img_key);
    conduit::float64_array emi = n_image.value();
    const size_t num_pixels = emi.number_of_elements();
    const ch_t* emi_data = n_image.value();

    image_ptrs.push_back(std::make_pair(num_pixels, emi_data));
  }
  return image_ptrs;
}

std::vector<cv::Mat> data_reader_jag_conduit::get_cv_images(const size_t sample_id) const {
  std::vector< std::pair<size_t, const ch_t*> > img_ptrs(get_image_ptrs(sample_id));
  std::vector<cv::Mat> images;
  images.reserve(img_ptrs.size());
  using InputBuf_T = cv_image_type<ch_t>;

  for (const auto& img: img_ptrs) {
    const int num_pixels = static_cast<int>(img.first);
    const ch_t* ptr = img.second;

    // add a zero copying view to data
    const cv::Mat image(num_pixels, 1, InputBuf_T::T(1u),
                        reinterpret_cast<void*>(const_cast<ch_t*>(ptr)));
    // reshape and clone (deep-copy) the image
    // to preserve the constness of the original data
    images.emplace_back((image.reshape(0, m_image_height)).clone());
  }
  return images;
}

std::vector<data_reader_jag_conduit::ch_t> data_reader_jag_conduit::get_images(const size_t sample_id) const {
  std::vector< std::pair<size_t, const ch_t*> > img_ptrs(get_image_ptrs(sample_id));
  std::vector<ch_t> images;
  images.reserve(get_linearized_image_size());

  for (const auto& img: img_ptrs) {
    const size_t num_pixels = img.first;
    const ch_t* ptr = img.second;
    images.insert(images.end(), ptr, ptr + num_pixels);
  }

  return images;
}

std::vector<data_reader_jag_conduit::scalar_t> data_reader_jag_conduit::get_scalars(const size_t sample_id) const {
  if (!check_sample_id(sample_id)) {
    _THROW_LBANN_EXCEPTION_(_CN_, "get_scalars() : invalid sample index");
  }

  std::vector<scalar_t> scalars;
  scalars.reserve(m_scalar_keys.size());

  for(const auto key: m_scalar_keys) {
    std::string scalar_key = std::to_string(sample_id) + "/outputs/scalars/" + key;
    const conduit::Node & n_scalar = get_conduit_node(scalar_key);
    scalars.push_back(n_scalar.value());
  }
  return scalars;
}

std::vector<data_reader_jag_conduit::input_t> data_reader_jag_conduit::get_inputs(const size_t sample_id) const {
  if (!check_sample_id(sample_id)) {
    _THROW_LBANN_EXCEPTION_(_CN_, "get_inputs() : invalid sample index");
  }

  std::vector<input_t> inputs;
  inputs.reserve(m_input_keys.size());

  for(const auto key: m_input_keys) {
    std::string input_key = std::to_string(sample_id) + "/inputs/" + key;
    const conduit::Node & n_input = get_conduit_node(input_key);
    inputs.push_back(n_input.value());
  }
  return inputs;
}

int data_reader_jag_conduit::check_exp_success(const size_t sample_id) const {
  if (!check_sample_id(sample_id)) {
    _THROW_LBANN_EXCEPTION_(_CN_, "check_exp_success() : invalid sample index");
  }

  return static_cast<int>(get_conduit_node(std::to_string(sample_id) + "performance/success").value());
}


bool data_reader_jag_conduit::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {
  switch (m_independent) {
    case JAG_Image: {
      // TODO: use get_image_ptrs() and views of matrix X to avoid data copying
      const std::vector<ch_t> images(get_images(data_id));
      set_minibatch_item<ch_t>(X, mb_idx, images.data(), get_linearized_image_size());
      break;
    }
    case JAG_Scalar: {
      const std::vector<scalar_t> scalars(get_scalars(data_id));
      set_minibatch_item<scalar_t>(X, mb_idx, scalars.data(), get_linearized_scalar_size());
      break;
    }
    case JAG_Input: {
      const std::vector<input_t> inputs(get_inputs(data_id));
      set_minibatch_item<input_t>(X, mb_idx, inputs.data(), get_linearized_input_size());
      break;
    }
    default: {
      _THROW_LBANN_EXCEPTION_(_CN_, "fetch_datum() : unknown variable type");
    }
  }
  return true;
}

bool data_reader_jag_conduit::fetch_response(Mat& Y, int data_id, int mb_idx, int tid) {
  switch (m_dependent) {
    case JAG_Image: {
      // TODO: use get_image_ptrs() and views of matrix Y to avoid data copying
      const std::vector<ch_t> images(get_images(data_id));
      set_minibatch_item<ch_t>(Y, mb_idx, images.data(), get_linearized_image_size());
      break;
    }
    case JAG_Scalar: {
      const std::vector<scalar_t> scalars(get_scalars(data_id));
      set_minibatch_item<scalar_t>(Y, mb_idx, scalars.data(), get_linearized_scalar_size());
      break;
    }
    case JAG_Input: {
      const std::vector<input_t> inputs(get_inputs(data_id));
      set_minibatch_item<input_t>(Y, mb_idx, inputs.data(), get_linearized_input_size());
      break;
    }
    default: {
      _THROW_LBANN_EXCEPTION_(_CN_, "fetch_response() : unknown variable type");
    }
  }
  return true;
}

void data_reader_jag_conduit::save_image(Mat& pixels, const std::string filename, bool do_scale) {
#ifndef _JAG_OFFLINE_TOOL_MODE_
  internal_save_image(pixels, filename, m_image_height, m_image_width, 1, do_scale);
#endif // _JAG_OFFLINE_TOOL_MODE_
}

} // end of namespace lbann

#undef _CN_
#endif // LBANN_HAS_CONDUIT
