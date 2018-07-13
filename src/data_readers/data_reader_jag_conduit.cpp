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
//#include "lbann/data_store/data_store_jag_conduit.hpp"
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
#include "lbann/data_readers/image_utils.hpp"
#include <omp.h>


// This macro may be moved to a global scope
#define _THROW_LBANN_EXCEPTION_(_CLASS_NAME_,_MSG_) { \
  std::stringstream err; \
  err << __FILE__ << ' '  << __LINE__ << " :: " \
      << (_CLASS_NAME_) << "::" << (_MSG_); \
  throw lbann_exception(err.str()); \
}

#define _THROW_LBANN_EXCEPTION2_(_CLASS_NAME_,_MSG1_,_MSG2_) { \
  std::stringstream err; \
  err << __FILE__ << ' '  << __LINE__ << " :: " \
      << (_CLASS_NAME_) << "::" << (_MSG1_) << (_MSG2_); \
  throw lbann_exception(err.str()); \
}

// This comes after all the headers, and is only visible within the current implementation file.
// To make sure, we put '#undef _CN_' at the end of this file
#define _CN_ "data_reader_jag_conduit"

namespace lbann {

const std::set<std::string> data_reader_jag_conduit::non_numeric_vars = {
  "fusion_reaction",
  "fusion_model_reaction",
  "radial_profile",
  "postp_timeseries_vars",
  "name",
  "solver",
  "mesh_def",
  "hs_volume_integral",
  "fusion_model_sv",
  "shell_model",
  "shape_model",
  "ablation_cv_model",
  "infalling_model",
  "radiation_model",
  "hotspot_model",
  "shape_model_initial_velocity_amplitude",
  "stopping_model",
  "energy_balance_model_ablation_cv_model",
  "solver_method",
  "conduction_model_conductivity",
  "solver_mode"
};

data_reader_jag_conduit::data_reader_jag_conduit(const std::shared_ptr<cv_process>& pp, bool shuffle)
  : generic_data_reader(shuffle) {
  set_defaults();

  if (!pp) {
    _THROW_LBANN_EXCEPTION_(get_type(), " construction error: no image processor");
  }

  replicate_processor(*pp);
}

void data_reader_jag_conduit::copy_members(const data_reader_jag_conduit& rhs) {
  m_independent = rhs.m_independent;
  m_dependent = rhs.m_dependent;
  m_image_width = rhs.m_image_width;
  m_image_height = rhs.m_image_height;
  m_image_num_channels = rhs.m_image_num_channels;
  set_linearized_image_size();
  m_num_img_srcs = rhs.m_num_img_srcs;
  m_is_data_loaded = rhs.m_is_data_loaded;
  m_scalar_keys = rhs.m_scalar_keys;
  m_input_keys = rhs.m_input_keys;
  m_success_map = rhs.m_success_map;

  if (rhs.m_pps.size() == 0u || !rhs.m_pps[0]) {
    _THROW_LBANN_EXCEPTION_(get_type(), " construction error: no image processor");
  }

  replicate_processor(*rhs.m_pps[0]);

  m_data = rhs.m_data;
  m_uniform_input_type = rhs.m_uniform_input_type;

  m_scalar_filter = rhs.m_scalar_filter;
  m_scalar_prefix_filter = rhs.m_scalar_prefix_filter;
  m_input_filter = rhs.m_input_filter;
  m_input_prefix_filter = rhs.m_input_prefix_filter;
}

data_reader_jag_conduit::data_reader_jag_conduit(const data_reader_jag_conduit& rhs)
  : generic_data_reader(rhs) {
  copy_members(rhs);
}

data_reader_jag_conduit& data_reader_jag_conduit::operator=(const data_reader_jag_conduit& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  generic_data_reader::operator=(rhs);

  copy_members(rhs);

  return (*this);
}

data_reader_jag_conduit::~data_reader_jag_conduit() {
}

void data_reader_jag_conduit::set_defaults() {
  m_independent.assign(1u, Undefined);
  m_dependent.assign(1u, Undefined);
  m_image_width = 0;
  m_image_height = 0;
  m_image_num_channels = 1;
  set_linearized_image_size();
  m_num_img_srcs = 1u;
  m_is_data_loaded = false;
  m_num_labels = 0;
  m_scalar_keys.clear();
  m_input_keys.clear();
  m_uniform_input_type = false;
  m_scalar_filter.clear();
  m_scalar_prefix_filter.clear();
  m_input_filter.clear();
  m_input_prefix_filter.clear();
}

/// Replicate image processor for each OpenMP thread
bool data_reader_jag_conduit::replicate_processor(const cv_process& pp) {
  const int nthreads = omp_get_max_threads();
  m_pps.resize(nthreads);

  // Construct thread private preprocessing objects out of a shared pointer
  #pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < nthreads; ++i) {
    //auto ppu = std::make_unique<cv_process>(pp); // c++14
    std::unique_ptr<cv_process> ppu(new cv_process(pp));
    m_pps[i] = std::move(ppu);
  }

  bool ok = true;
  for (int i = 0; ok && (i < nthreads); ++i) {
    if (!m_pps[i]) ok = false;
  }

  if (!ok || (nthreads <= 0)) {
    _THROW_LBANN_EXCEPTION_(get_type(), " cannot replicate image processor");
    return false;
  }

  const std::vector<unsigned int> dims = pp.get_data_dims();
  if ((dims.size() == 2u) && (dims[0] != 0u) && (dims[1] != 0u)) {
    m_image_width = static_cast<int>(dims[0]);
    m_image_height = static_cast<int>(dims[1]);
  }

  return true;
}

const conduit::Node& data_reader_jag_conduit::get_conduit_node(const std::string key) const {
  return m_data[key];
}


void data_reader_jag_conduit::set_independent_variable_type(
  const std::vector<data_reader_jag_conduit::variable_t> independent) {
  if (!independent.empty() && !m_independent.empty() && (m_independent[0] == Undefined)) {
    m_independent.clear();
  }
  for (const auto t: independent) {
    add_independent_variable_type(t);
  }
}

void data_reader_jag_conduit::add_independent_variable_type(
  const data_reader_jag_conduit::variable_t independent) {
  if (!(independent == JAG_Image || independent == JAG_Scalar ||
        independent == JAG_Input || independent == Undefined)) {
    _THROW_LBANN_EXCEPTION_(_CN_, "unrecognized independent variable type ");
  }
  m_independent.push_back(independent);
}

void data_reader_jag_conduit::set_dependent_variable_type(
  const std::vector<data_reader_jag_conduit::variable_t> dependent) {
  if (!dependent.empty() && !m_dependent.empty() && (m_dependent[0] == Undefined)) {
    m_dependent.clear();
  }
  for (const auto t: dependent) {
    add_dependent_variable_type(t);
  }
}

void data_reader_jag_conduit::add_dependent_variable_type(
  const data_reader_jag_conduit::variable_t dependent) {
  if (!(dependent == JAG_Image || dependent == JAG_Scalar ||
        dependent == JAG_Input || dependent == Undefined)) {
    _THROW_LBANN_EXCEPTION_(_CN_, "unrecognized dependent variable type ");
  }
  m_dependent.push_back(dependent);
}

std::vector<data_reader_jag_conduit::variable_t>
data_reader_jag_conduit::get_independent_variable_type() const {
  return m_independent;
}

std::vector<data_reader_jag_conduit::variable_t>
data_reader_jag_conduit::get_dependent_variable_type() const {
  return m_dependent;
}

void data_reader_jag_conduit::set_image_dims(const int width, const int height, const int ch) {
  if ((width > 0) && (height > 0)) { // set and valid
    m_image_width = width;
    m_image_height = height;
    m_image_num_channels = ch;
  } else if (!((width == 0) && (height == 0))) { // set but not valid
    _THROW_LBANN_EXCEPTION_(_CN_, "set_image_dims() : invalid image dims");
  }
  set_linearized_image_size();
}

void data_reader_jag_conduit::add_scalar_filter(const std::string& key) {
  m_scalar_filter.insert(key);
}

void data_reader_jag_conduit::add_scalar_prefix_filter(const prefix_t& p) {
  m_scalar_prefix_filter.push_back((p.first.length() > p.second)? prefix_t(p.first, p.first.length()) : p);
}

void data_reader_jag_conduit::add_input_filter(const std::string& key) {
  m_input_filter.insert(key);
}

void data_reader_jag_conduit::add_input_prefix_filter(const prefix_t& p) {
  m_input_prefix_filter.push_back((p.first.length() > p.second)? prefix_t(p.first, p.first.length()) : p);
}

/**
 * First, it checks if the key is in the list of keys to filter.
 * Then, it checks if the key contains any prefix string to filter
 * while sayisfying the mininum length requirement.
 */
bool data_reader_jag_conduit::filter(const std::set<std::string>& filter,
  const std::vector<data_reader_jag_conduit::prefix_t>& prefix_filter, const std::string& key) const {
  if (filter.find(key) != filter.end()) {
    return true;
  }
  for (const auto& pf: prefix_filter) {
    if (key.length() < pf.second) { // minimum length requirement
      continue;
    }
    if (key.compare(0, pf.first.length(), pf.first) == 0) { // match
      return true;
    }
  }
  return false;
}

/**
 * To use no key, set 'Undefined' to the corresponding variable type,
 * or call this with an empty vector argument after loading data.
 */
void data_reader_jag_conduit::set_scalar_choices(const std::vector<std::string>& keys) {
  m_scalar_keys = keys;
  // If this call is made after loading data, check the keys
  if (m_is_data_loaded) {
    check_scalar_keys();
  } else if (keys.empty()) {
    _THROW_LBANN_EXCEPTION2_(_CN_, "set_scalar_choices() : ", \
                                   "empty keys not allowed before data loading");
  }
}

void data_reader_jag_conduit::set_all_scalar_choices() {
  if (m_success_map.size() == 0) {
    return;
  }
  const conduit::Node & n_scalar = get_conduit_node(m_success_map[0] + "/outputs/scalars");
  m_scalar_keys.reserve(n_scalar.number_of_children());
  const std::vector<std::string>& child_names = n_scalar.child_names();
  for (const auto& key: child_names) {
    if (filter(m_scalar_filter, m_scalar_prefix_filter, key)) {
      continue;
    }
    m_scalar_keys.push_back(key);
  }
}

const std::vector<std::string>& data_reader_jag_conduit::get_scalar_choices() const {
  return m_scalar_keys;
}


/**
 * To use no key, set 'Undefined' to the corresponding variable type,
 * or call this with an empty vector argument after loading data.
 */
void data_reader_jag_conduit::set_input_choices(const std::vector<std::string>& keys) {
  m_input_keys = keys;
  // If this call is made after loading data, check the keys
  if (m_is_data_loaded) {
    check_input_keys();
  } else if (keys.empty()) {
    _THROW_LBANN_EXCEPTION2_(_CN_, "set_input_choices() : ", \
                                   "empty keys not allowed before data loading");
  }
}

void data_reader_jag_conduit::set_all_input_choices() {
  if (m_success_map.size() == 0) {
    return;
  }
  const conduit::Node & n_input = get_conduit_node(m_success_map[0] + "/inputs");
  m_input_keys.reserve(n_input.number_of_children());
  const std::vector<std::string>& child_names = n_input.child_names();
  for (const auto& key: child_names) {
    if (filter(m_input_filter, m_input_prefix_filter, key)) {
      continue;
    }
    m_input_keys.push_back(key);
  }
}

const std::vector<std::string>& data_reader_jag_conduit::get_input_choices() const {
  return m_input_keys;
}


void data_reader_jag_conduit::set_num_img_srcs() {
  m_num_img_srcs = m_emi_selectors.size();
#if 0

  if (m_success_map.size() == 0) {
    return;
  }

  conduit::NodeConstIterator itr = get_conduit_node(m_success_map[0] + "/outputs/images").children();

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

  m_num_img_srcs = views.size();
  if (m_num_img_srcs == 0u) {
    m_num_img_srcs = 1u;
  }
#endif
}

void data_reader_jag_conduit::set_linearized_image_size() {
  m_image_linearized_size = m_image_width * m_image_height;
  //m_image_linearized_size = m_image_width * m_image_height * m_image_num_channels;
  // TODO: we do not know how multi-channel image data will be formatted yet.
}

void data_reader_jag_conduit::check_image_size() {
  if (m_success_map.size() == 0) {
    return;
  }

  const conduit::Node & n_imageset = get_conduit_node(m_success_map[0] + "/outputs/images");
  if (static_cast<size_t>(n_imageset.number_of_children()) == 0u) {
    //m_image_width = 0;
    //m_image_height = 0;
    //set_linearized_image_size();
    _THROW_LBANN_EXCEPTION_(_CN_, "check_image_size() : no image in data");
    return;
  }
  const conduit::Node & n_image = get_conduit_node(m_success_map[0] + "/outputs/images/(0.0, 0.0)/0.0/emi");
  conduit::float32_array emi = n_image.value();

  m_image_linearized_size = static_cast<size_t>(emi.number_of_elements());

#if 0
//dah: the following block is throwing an error, since m_image_linearized_size = 64*64, which is != emi.number_of_elements(), which is 16384

  if (m_image_linearized_size != static_cast<size_t>(emi.number_of_elements())) {
    if ((m_image_width == 0) && (m_image_height == 0)) {
      m_image_height = 1;
      m_image_width = static_cast<int>(emi.number_of_elements());
      set_linearized_image_size();
    } else {
      //_THROW_LBANN_EXCEPTION_(_CN_, "check_image_size() : image size mismatch");
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          <<"check_image_size() : image size mismatch; m_image_width: " 
          << m_image_width << " m_image_height: " << m_image_height 
          << " m_image_linearized_size: " << m_image_linearized_size << std::endl;
    }
  }
#endif
}

void data_reader_jag_conduit::check_scalar_keys() {
  if (m_success_map.size() == 0) {
    m_scalar_keys.clear();
    return;
  }

  size_t num_found = 0u;
  std::vector<bool> found(m_scalar_keys.size(), false);
  std::set<std::string> keys_conduit;

  const conduit::Node & n_scalar = get_conduit_node(m_success_map[0] + "/outputs/scalars");
  const std::vector<std::string>& child_names = n_scalar.child_names();
  for (const auto& key: child_names) {
    keys_conduit.insert(key);
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
  if (m_success_map.size() == 0) {
    m_input_keys.clear();
    return;
  }

  size_t num_found = 0u;
  std::vector<bool> found(m_input_keys.size(), false);
  std::map<std::string, TypeID> keys_conduit;

  const conduit::Node & n_input = get_conduit_node(m_success_map[0] + "/inputs");
  conduit::NodeConstIterator itr = n_input.children();

  while (itr.has_next()) {
    const conduit::Node & n = itr.next();
    keys_conduit.insert(std::pair<std::string, TypeID>(itr.name(), static_cast<TypeID>(n.dtype().id())));
  }

  bool is_input_t = true;

  for (size_t i=0u; i < m_input_keys.size(); ++i) {
    std::map<std::string, TypeID>::const_iterator it = keys_conduit.find(m_input_keys[i]);
    if (it != keys_conduit.end()) {
      num_found ++;
      found[i] = true;
      is_input_t = is_input_t && is_same_type<input_t>(it->second);
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

  m_uniform_input_type = (m_input_keys.size() == 0u)? false : is_input_t;
}


#ifndef _JAG_OFFLINE_TOOL_MODE_
void data_reader_jag_conduit::load() {
  if(m_gan_labelling) {
    m_num_labels=2;
  }

  if (is_master()) {
    std::cout << "JAG load GAN m_gan_labelling : label_value "
              << m_gan_labelling <<" : " << m_gan_label_value << std::endl;
  }

  const std::string data_dir = add_delimiter(get_file_dir());
  const std::string conduit_file_name = get_data_filename();

  m_emi_selectors.insert("(0.0, 0.0)");
  m_emi_selectors.insert("(90.0, 0.0)");
  m_emi_selectors.insert("(90.0, 78.0)");

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

  // set up mapping: need to do this since some of the data may be bad
  const std::vector<std::string> &children_names = m_data.child_names();
  int idx = 0;
  for (auto t : children_names) {
    const std::string key = "/" + t + "/performance/success";
    const conduit::Node& n_ok = get_conduit_node(key);
    int success = n_ok.to_int64();
    if (success == 1) {
      m_success_map[idx++] = t;
    } 
  }

  set_num_img_srcs();
  check_image_size();

  if (!m_is_data_loaded) {
    if (m_scalar_keys.size() == 0u) {
      set_all_scalar_choices(); // use all by default if none is specified
    }
    check_scalar_keys();

    if (m_input_keys.size() == 0u) {
      set_all_input_choices(); // use all by default if none is specified
    }
    check_input_keys();
  }

  m_is_data_loaded = true;
}


size_t data_reader_jag_conduit::get_num_samples() const {
  return m_success_map.size();
}

unsigned int data_reader_jag_conduit::get_num_img_srcs() const {
  return m_num_img_srcs;
}

size_t data_reader_jag_conduit::get_linearized_image_size() const {
  return m_image_linearized_size;
}

size_t data_reader_jag_conduit::get_linearized_scalar_size() const {
  return m_scalar_keys.size();
}

size_t data_reader_jag_conduit::get_linearized_input_size() const {
  return m_input_keys.size();
}


size_t data_reader_jag_conduit::get_linearized_size(const data_reader_jag_conduit::variable_t t) const {
  switch (t) {
    case JAG_Image:
      return get_linearized_image_size() * get_num_img_srcs();
    case JAG_Scalar:
      return get_linearized_scalar_size();
    case JAG_Input:
      return get_linearized_input_size();
    default: { // includes Unefined case
      _THROW_LBANN_EXCEPTION2_(_CN_, "get_linearized_size() : ", \
                                     "unknown or undefined variable type");
    }
  }
  return 0u;
}

int data_reader_jag_conduit::get_linearized_data_size() const {
  size_t sz = 0u;
  for (const auto t: m_independent) {
    if (t == Undefined) {
      continue;
    }
    sz += get_linearized_size(t);
  }
  return static_cast<int>(sz);
}

int data_reader_jag_conduit::get_linearized_response_size() const {
  size_t sz = 0u;
  for (const auto t: m_dependent) {
    if (t == Undefined) {
      continue;
    }
    sz += get_linearized_size(t);
  }
  return static_cast<int>(sz);
}

std::vector<size_t> data_reader_jag_conduit::get_linearized_data_sizes() const {
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

std::vector<size_t> data_reader_jag_conduit::get_linearized_response_sizes() const {
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

const std::vector<int> data_reader_jag_conduit::get_dims(const data_reader_jag_conduit::variable_t t) const {
  switch (t) {
    case JAG_Image:
      return {static_cast<int>(get_num_img_srcs()), m_image_height, m_image_width};
      //return {static_cast<int>(get_linearized_image_size())};
    case JAG_Scalar:
      return {static_cast<int>(get_linearized_scalar_size())};
    case JAG_Input:
      return {static_cast<int>(get_linearized_input_size())};
    default: { // includes Undefined case
      _THROW_LBANN_EXCEPTION2_(_CN_, "get_dims() : ", \
                                     "unknown or undefined variable type");
    }
  }
  return {};
}

const std::vector<int> data_reader_jag_conduit::get_data_dims() const {
  std::vector<int> all_dim;
  for (const auto t: m_independent) {
    if (t == Undefined) {
      continue;
    }
    const std::vector<int> ld = get_dims(t);
    all_dim.insert(all_dim.end(), ld.begin(), ld.end());
  }

if (is_master()) {
  std::cerr << "\ndebug from data_reader_jag_conduit::get_data_dims(); data dims: ";
  for (auto t : all_dim) std::cerr << t << " ";
  std::cerr << "\n\n";
}

  return all_dim;
}

int data_reader_jag_conduit::get_num_labels() const {
  return m_num_labels;
}

int data_reader_jag_conduit::get_linearized_label_size() const {
  return m_num_labels;
}


std::string data_reader_jag_conduit::to_string(const variable_t t) {
  switch (t) {
    case Undefined:  return "Undefined";
    case JAG_Image:  return "JAG_Image";
    case JAG_Scalar: return "JAG_Scalar";
    case JAG_Input:  return "JAG_Input";
  }
  return "Undefined";
}

std::string data_reader_jag_conduit::to_string(const std::vector<data_reader_jag_conduit::variable_t>& vec) {
  std::string str("[");
  for (const auto& el: vec) {
    str += ' ' + data_reader_jag_conduit::to_string(el);
  }
  str += " ]";
  return str;
}

std::string data_reader_jag_conduit::get_description() const {
  std::string ret = std::string("data_reader_jag_conduit:\n")
    + " - independent: " + data_reader_jag_conduit::to_string(m_independent) + "\n"
    + " - dependent: " + data_reader_jag_conduit::to_string(m_dependent) + "\n"
    + " - images: "   + std::to_string(m_num_img_srcs) + 'x'
                      + std::to_string(m_image_width) + 'x'
                      + std::to_string(m_image_height) + "\n"
    + " - scalars: "  + std::to_string(get_linearized_scalar_size()) + "\n"
    + " - inputs: "   + std::to_string(get_linearized_input_size()) + "\n"
    + " - uniform_input_type: " + (m_uniform_input_type? "true" : "false") + '\n';
  if (!m_scalar_filter.empty()) {
    ret += " - scalar filter:";
    for (const auto& f: m_scalar_filter) {
      ret += " \"" + f + '"';
    }
    ret += '\n';
  }
  if (!m_scalar_prefix_filter.empty()) {
    ret += " - scalar prefix filter:";
    for (const auto& f: m_scalar_prefix_filter) {
      ret += " [\"" + f.first + "\" " + std::to_string(f.second) + ']';
    }
    ret += '\n';
  }
  if (!m_input_filter.empty()) {
    ret += " - input filter:";
    for (const auto& f: m_input_filter) {
      ret += " \"" + f + '"';
    }
    ret += '\n';
  }
  if (!m_input_prefix_filter.empty()) {
    ret += " - input prefix filter:";
    for (const auto& f: m_input_prefix_filter) {
      ret += " [\"" + f.first + "\" " + std::to_string(f.second) + ']';
    }
    ret += '\n';
  }
  return ret;
}


/// I think this is no longer relevant, since calls: check_sample_id(0)
/// have been replaced by: if (m_success_map.size() == 0) return;
bool data_reader_jag_conduit::check_sample_id(const size_t sample_id) const {
  return (static_cast<conduit_index_t>(sample_id) < m_data.number_of_children());
}

bool data_reader_jag_conduit::check_non_numeric(const std::string key) {
  std::set<std::string>::const_iterator kit = non_numeric_vars.find(key);
  if (kit != non_numeric_vars.end()) {
    std::string err = "data_reader_jag_conduit::add_val() : non-numeric '" + key
                    + "' requires a conversion method.";
   #if 1
    std::cerr << err << " Skipping for now." << std::endl;
   #else
    throw lbann_exception(err);
   #endif
    return true;
  }
  return false;
}


std::vector<int> data_reader_jag_conduit::choose_image_near_bang_time(const size_t sample_id) const {
  std::vector<int> img_indices;
  return img_indices;
#if 0
  using view_map = std::map<std::pair<float, float>, std::pair<int, double> >;

  conduit::NodeConstIterator itr = get_conduit_node(m_success_map[sample_id] + "/outputs/images").children();
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
#endif
}

std::vector< std::pair<size_t, const data_reader_jag_conduit::ch_t*> >
data_reader_jag_conduit::get_image_ptrs(const size_t sample_id) const {
  if (sample_id >= m_success_map.size()) {
    _THROW_LBANN_EXCEPTION_(_CN_, "get_images() : invalid sample index");
  }

  std::vector< std::pair<size_t, const ch_t*> >image_ptrs;
  std::unordered_map<int, std::string>::const_iterator it = m_success_map.find(sample_id);

  for (auto t : m_emi_selectors) {
    std::string img_key = it->second + "/outputs/images/" + t + "/0.0/emi";
    const conduit::Node & n_image = get_conduit_node(img_key);
    conduit::float64_array emi = n_image.value();
    const size_t num_pixels = emi.number_of_elements();
    const ch_t* emi_data = n_image.value();
    image_ptrs.push_back(std::make_pair(num_pixels, emi_data));
  }  

  return image_ptrs;
}

cv::Mat data_reader_jag_conduit::cast_to_cvMat(const std::pair<size_t, const ch_t*> img, const int height) {
  const int num_pixels = static_cast<int>(img.first);
  const ch_t* ptr = img.second;

  // add a zero copying view to data
  using InputBuf_T = cv_image_type<ch_t>;
  const cv::Mat image(num_pixels, 1, InputBuf_T::T(1u),
                      reinterpret_cast<void*>(const_cast<ch_t*>(ptr)));
  // reshape the image. Furter need to clone (deep-copy) the image
  // to preserve the constness of the original data
  return (image.reshape(0, height));
}

std::vector<cv::Mat> data_reader_jag_conduit::get_cv_images(const size_t sample_id) const {
  std::vector< std::pair<size_t, const ch_t*> > img_ptrs(get_image_ptrs(sample_id));
  std::vector<cv::Mat> images;
  images.reserve(img_ptrs.size());

  for (const auto& img: img_ptrs) {
    images.emplace_back(cast_to_cvMat(img, m_image_height).clone());
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
    // All the scalar output currently seems to be scalar_t
    //add_val(key, n_scalar, scalars);
    scalars.push_back(static_cast<scalar_t>(n_scalar.to_value()));
  }
  return scalars;
}

std::vector<data_reader_jag_conduit::input_t> data_reader_jag_conduit::get_inputs(const size_t sample_id) const {
  if (!check_sample_id(sample_id)) {
    _THROW_LBANN_EXCEPTION_(_CN_, "get_inputs() : invalid sample index");
  }

  std::vector<input_t> inputs;
  inputs.reserve(m_input_keys.size());

  // automatically determine which method to use based on if all the variables are of input_t
  if (m_uniform_input_type) {
    for(const auto key: m_input_keys) {
      std::string input_key = std::to_string(sample_id) + "/inputs/" + key;
      const conduit::Node & n_input = get_conduit_node(input_key);
      inputs.push_back(n_input.value()); // less overhead
    }
  } else {
    for(const auto key: m_input_keys) {
      std::string input_key = std::to_string(sample_id) + "/inputs/" + key;
      const conduit::Node & n_input = get_conduit_node(input_key);
      add_val(key, n_input, inputs); // more overhead but general
    }
  }
  return inputs;
}

int data_reader_jag_conduit::check_exp_success(const size_t sample_id) const {
  if (!check_sample_id(sample_id)) {
    _THROW_LBANN_EXCEPTION_(_CN_, "check_exp_success() : invalid sample index");
  }

  return static_cast<int>(get_conduit_node(std::to_string(sample_id) + "performance/success").value());
}


std::vector<CPUMat>
data_reader_jag_conduit::create_datum_views(CPUMat& X, const std::vector<size_t>& sizes, const int mb_idx) const {
  std::vector<CPUMat> X_v(sizes.size());
  El::Int h = 0;

  for(size_t i=0u; i < sizes.size(); ++i) {
    const El::Int h_end =  h + static_cast<El::Int>(sizes[i]);
    El::View(X_v[i], X, El::IR(h, h_end), El::IR(mb_idx, mb_idx + 1));
    h = h_end;
  }
  return X_v;
}

bool data_reader_jag_conduit::fetch(CPUMat& X, int data_id, int mb_idx, int tid,
  const data_reader_jag_conduit::variable_t vt, const std::string tag) {
  switch (vt) {
    case JAG_Image: {
      const std::vector<size_t> sizes(get_num_img_srcs(), get_linearized_image_size());
      std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);
      std::vector<cv::Mat> images = get_cv_images(data_id);

      if (images.size() != get_num_img_srcs()) {
        _THROW_LBANN_EXCEPTION2_(_CN_, "fetch() : the number of images is not as expected", \
          std::to_string(images.size()) + "!=" + std::to_string(get_num_img_srcs()));
      }

      for(size_t i=0u; i < get_num_img_srcs(); ++i) {
        int width, height, img_type;
        image_utils::process_image(images[i], width, height, img_type, *(m_pps[tid]), X_v[i]);
      }
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
    default: { // includes Undefined case
      _THROW_LBANN_EXCEPTION_(_CN_, "fetch_" + tag + "() : unknown or undefined variable type");
    }
  }
  return true;
}

bool data_reader_jag_conduit::fetch_datum(CPUMat& X, int data_id, int mb_idx, int tid) {
  std::vector<size_t> sizes = get_linearized_data_sizes();
  std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);
  bool ok = true;
  for(size_t i = 0u; ok && (i < X_v.size()); ++i) {
    // The third argument mb_idx below is 0 because it is for the view of X not X itself
    ok = fetch(X_v[i], data_id, 0, tid, m_independent[i], "datum");
  }
  return ok;
}

bool data_reader_jag_conduit::fetch_response(CPUMat& X, int data_id, int mb_idx, int tid) {
  std::vector<size_t> sizes = get_linearized_response_sizes();
  std::vector<CPUMat> X_v = create_datum_views(X, sizes, mb_idx);
  bool ok = true;
  for(size_t i = 0u; ok && (i < X_v.size()); ++i) {
    ok = fetch(X_v[i], data_id, 0, tid, m_dependent[i], "response");
  }
  return ok;
}

bool data_reader_jag_conduit::fetch_label(CPUMat& Y, int data_id, int mb_idx, int tid) {
  if(m_gan_label_value) Y.Set(m_gan_label_value,mb_idx,1); //fake sample is set to 1; adversarial model
  else { //fake sample (second half of minibatch is set to 0;discriminator model
    //mb_idx < (m_mb_size/2) ? Y.Set(1,mb_idx,1) : Y.Set(m_gan_label_value,mb_idx,1);
    mb_idx < (get_current_mini_batch_size()/2) ? Y.Set(1,mb_idx,1) : Y.Set(m_gan_label_value,mb_idx,1);
  }
  //Y.Set(m_gan_label_value, mb_idx, 1);
  return true;
}

#ifndef _JAG_OFFLINE_TOOL_MODE_
void data_reader_jag_conduit::setup_data_store(model *m) {
  if (m_data_store != nullptr) {
    //delete m_data_store;
  }
/*
  m_data_store = new data_store_jag_conduit(this, m);
  if (m_data_store != nullptr) {
    m_data_store->setup();
  }
*/
}
#endif // _JAG_OFFLINE_TOOL_MODE_

void data_reader_jag_conduit::save_image(Mat& pixels, const std::string filename, bool do_scale) {
#ifndef _JAG_OFFLINE_TOOL_MODE_
  internal_save_image(pixels, filename, m_image_height, m_image_width, 1, do_scale);
#endif // _JAG_OFFLINE_TOOL_MODE_
}

void data_reader_jag_conduit::print_schema() const {
  m_data.schema().print();
}

void data_reader_jag_conduit::print_schema(const size_t sample_id) const {
  const conduit::Node & n = get_conduit_node(std::to_string(sample_id));
  n.schema().print();
}

} // end of namespace lbann

#undef _CN_
#endif // LBANN_HAS_CONDUIT
