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
//#include "lbann/data_store/data_store_jag_conduit.hpp"
#else
#include "data_reader_jag_conduit.hpp"
#endif // _JAG_OFFLINE_TOOL_MODE_

#ifdef LBANN_HAS_CONDUIT
#include "lbann/utils/file_utils.hpp" // for add_delimiter() in load()
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
#include "lbann/utils/timer.hpp"
#include "lbann/utils/glob.hpp"


// This macro may be moved to a global scope
#define _THROW_LBANN_EXCEPTION_(_CLASS_NAME_,_MSG_) { \
  std::stringstream _err; \
  _err << __FILE__ << ' '  << __LINE__ << " :: " \
      << (_CLASS_NAME_) << "::" << (_MSG_); \
  throw lbann_exception(_err.str()); \
}

#define _THROW_LBANN_EXCEPTION2_(_CLASS_NAME_,_MSG1_,_MSG2_) { \
  std::stringstream _err; \
  _err << __FILE__ << ' '  << __LINE__ << " :: " \
      << (_CLASS_NAME_) << "::" << (_MSG1_) << (_MSG2_); \
  throw lbann_exception(_err.str()); \
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

#ifndef _JAG_OFFLINE_TOOL_MODE_
// These methods are overriden to allow each process to load and consume unique set of data files
bool data_reader_jag_conduit::position_valid() const {
  const bool ok = (static_cast<size_t>(m_shuffled_indices[m_current_pos]) < m_valid_samples.size())
    && (m_current_pos < (int)m_shuffled_indices.size());
  if (!ok) {
    const size_t my_rank = static_cast<size_t>(m_comm->get_rank_in_model());
    std::stringstream err;
    err << "rank " << my_rank << " position invalid: m_shuffled_indices["
        << m_current_pos << "] (" << m_shuffled_indices[m_current_pos]
        << ") >= m_valid_samples.size() (" << m_valid_samples.size() << ")" << std::endl;
    std::cerr << err.str();
  }
  return ok;
}

void data_reader_jag_conduit::set_initial_position() {
  set_base_offset(0);
  generic_data_reader::set_initial_position();
}

int data_reader_jag_conduit::get_num_data() const {
  return m_global_num_samples_to_use;
}

void data_reader_jag_conduit::shuffle_indices() {
  // Shuffle the data
  if (m_shuffle) {
    std::shuffle(m_valid_samples.begin(), m_valid_samples.end(),
                 get_data_seq_generator());
  }
  m_valid_samples.resize(m_local_num_samples_to_use);
}
#endif // _JAG_OFFLINE_TOOL_MODE_

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
  m_emi_image_keys = rhs.m_emi_image_keys;
  m_scalar_keys = rhs.m_scalar_keys;
  m_input_keys = rhs.m_input_keys;

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
  m_valid_samples = rhs.m_valid_samples;
  m_local_num_samples_to_use = rhs.m_local_num_samples_to_use;
  m_global_num_samples_to_use = rhs.m_global_num_samples_to_use;
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
  m_emi_image_keys.clear();
  m_scalar_keys.clear();
  m_input_keys.clear();
  m_uniform_input_type = false;
  m_scalar_filter.clear();
  m_scalar_prefix_filter.clear();
  m_input_filter.clear();
  m_input_prefix_filter.clear();
  m_valid_samples.clear();
  m_local_num_samples_to_use = 0ul;
  m_global_num_samples_to_use = 0ul;
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
  if ((width > 0) && (height > 0) && (ch > 0)) { // set and valid
    m_image_width = width;
    m_image_height = height;
    m_image_num_channels = ch;
  } else if (!((width == 0) && (height == 0) && (ch == 1))) { // set but not valid
    _THROW_LBANN_EXCEPTION_(_CN_, "set_image_dims() : invalid image dims");
  }
  set_linearized_image_size();
}

void data_reader_jag_conduit::set_image_keys(const std::vector<std::string> image_keys) {
  m_emi_image_keys = image_keys;
  //image_keys: ["(0.0, 0.0)/0.0","(90.0, 0.0)/0.0","(90.0, 78.0)/0.0"];

  m_num_img_srcs = m_emi_image_keys.size();
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
bool data_reader_jag_conduit::filter(const std::set<std::string>& key_filter,
  const std::vector<data_reader_jag_conduit::prefix_t>& prefix_filter, const std::string& key) const {
  if (key_filter.find(key) != key_filter.cend()) {
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
  if (m_valid_samples.empty()) {
    return;
  }
  const conduit::Node & n_scalar = get_conduit_node(m_valid_samples[0] + "/outputs/scalars");
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
  if (m_valid_samples.empty()) {
    return;
  }
  const conduit::Node & n_input = get_conduit_node(m_valid_samples[0] + "/inputs");
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


void data_reader_jag_conduit::set_linearized_image_size() {
  m_image_linearized_size = m_image_width * m_image_height * m_image_num_channels;
}

void data_reader_jag_conduit::check_image_data() {
  if (m_valid_samples.empty()) {
    return;
  }

  if (!m_data.has_path(m_valid_samples[0])) {
    _THROW_LBANN_EXCEPTION_(_CN_, "check_image_data() : no sample by " + m_valid_samples[0]);
    return;
  }
  const conduit::Node & n_imageset = get_conduit_node(m_valid_samples[0] + "/outputs/images");
  if (static_cast<size_t>(n_imageset.number_of_children()) == 0u) {
    _THROW_LBANN_EXCEPTION_(_CN_, "check_image_data() : no image in data");
    return;
  }
  if (m_emi_image_keys.size() == 0u) {
    _THROW_LBANN_EXCEPTION_(_CN_, "check_image_data() : no image is selected");
    return;
  }
  for (const auto& emi_tag: m_emi_image_keys) {
    if (!m_data.has_path(m_valid_samples[0] + "/outputs/images/" + emi_tag + "/emi")) {
      _THROW_LBANN_EXCEPTION_(_CN_, "check_image_data() : no emi image by " + emi_tag);
      return;
    }
  }
  const conduit::Node & n_image
    = get_conduit_node(m_valid_samples[0] + "/outputs/images/" + m_emi_image_keys[0] + "/emi");
  conduit_ch_t emi = n_image.value();

  if (m_image_linearized_size != static_cast<size_t>(emi.number_of_elements())) {
    if ((m_image_width == 0) && (m_image_height == 0)) {
      m_image_height = 1;
      m_image_width = static_cast<int>(emi.number_of_elements());
      m_image_num_channels = 1;
      set_linearized_image_size();
    } else {
      std::string msg = "expected linearized emi image size: "
                      + std::to_string(emi.number_of_elements()) + '\n';
      _THROW_LBANN_EXCEPTION_(_CN_, msg + get_description());
    }
  }
}

void data_reader_jag_conduit::check_scalar_keys() {
  if (m_valid_samples.empty()) {
    m_scalar_keys.clear();
    return;
  }

  size_t num_found = 0u;
  std::vector<bool> found(m_scalar_keys.size(), false);
  std::set<std::string> keys_conduit;

  const conduit::Node & n_scalar = get_conduit_node(m_valid_samples[0] + "/outputs/scalars");
  const std::vector<std::string>& child_names = n_scalar.child_names();
  for (const auto& key: child_names) {
    keys_conduit.insert(key);
  }

  for (size_t i=0u; i < m_scalar_keys.size(); ++i) {
    std::set<std::string>::const_iterator it = keys_conduit.find(m_scalar_keys[i]);
    if (it != keys_conduit.cend()) {
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
  if (m_valid_samples.empty()) {
    m_input_keys.clear();
    return;
  }

  size_t num_found = 0u;
  std::vector<bool> found(m_input_keys.size(), false);
  std::map<std::string, TypeID> keys_conduit;

  const conduit::Node & n_input = get_conduit_node(m_valid_samples[0] + "/inputs");
  conduit::NodeConstIterator itr = n_input.children();

  while (itr.has_next()) {
    const conduit::Node & n = itr.next();
    keys_conduit.insert(std::pair<std::string, TypeID>(itr.name(), static_cast<TypeID>(n.dtype().id())));
  }

  bool is_input_t = true;

  for (size_t i=0u; i < m_input_keys.size(); ++i) {
    std::map<std::string, TypeID>::const_iterator it = keys_conduit.find(m_input_keys[i]);
    if (it != keys_conduit.cend()) {
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
void data_reader_jag_conduit::determine_num_samples_to_use() {
#if 1
  // The meaning of m_first_n is slightly different in this data reader as it
  // represents the first n local samples instead of the first n global samples.
  if (m_first_n > 0) {
    const size_t num_samples = std::min(static_cast<size_t>(m_first_n), get_num_valid_local_samples());
    m_valid_samples.resize(num_samples); // this does not work with unordered_map but with vector
  }
#else
  if (m_first_n > 0) {
    _THROW_LBANN_EXCEPTION_(_CN_, "load() does not support first_n feature.");
  }
#endif

#if 1
  // We do not support "percent_of_data_to_use" or "absolute_sample_count" yet.
  if ((get_use_percent() != 1.0) || (get_absolute_sample_count() != static_cast<size_t>(0u))) {
    _THROW_LBANN_EXCEPTION_(get_type(), \
      "'percent_of_data_to_use' and 'absolute_sample_count' are not supported with this data reader");
  }
  if (get_validation_percent() != 0.0) {
    _THROW_LBANN_EXCEPTION_(get_type(), \
      "'validation_percent' is not supported with this data reader");
  }
#else
  select_subset_of_data();
#endif

  const size_t num_valid_samples = get_num_valid_local_samples();

  const size_t my_rank = static_cast<size_t>(m_comm->get_rank_in_model());
  const size_t num_ranks = static_cast<size_t>(m_comm->get_procs_per_model());

  // Find the minimum of the number of valid samples locally available
  unsigned long long n_loc = static_cast<unsigned long long>(num_valid_samples);
  unsigned long long n_min = static_cast<unsigned long long>(num_valid_samples);
  m_comm->model_allreduce(&n_loc, 1, &n_min, El::mpi::MIN);

  // Find the first rank that has the minimum number of valid samples 
  int rank_tmp_1st = (n_loc == n_min)? static_cast<int>(my_rank) : static_cast<int>(num_ranks);
  int rank_min_1st;
  m_comm->model_allreduce(&rank_tmp_1st, 1, &rank_min_1st, El::mpi::MIN);

  // Determine the number of samples to use
  m_global_num_samples_to_use = 0u;
  m_local_num_samples_to_use = 0u;

  m_global_num_samples_to_use = static_cast<size_t>(n_min * num_ranks + rank_min_1st);
  if (m_global_num_samples_to_use == static_cast<size_t>(0u)) {
    _THROW_LBANN_EXCEPTION_(get_type(), "No valid sample found.");
  }
  m_local_num_samples_to_use = n_min;
  m_local_num_samples_to_use = (static_cast<int>(my_rank) < rank_min_1st)? (n_min+1) : n_min;


  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_global_num_samples_to_use);

  int s = 0;
  for(size_t n = 0u; n < m_shuffled_indices.size() ; n += num_ranks) {
    for(size_t r = 0; (r < num_ranks) && (n+r < m_shuffled_indices.size()); ++r) {
      m_shuffled_indices[n+r] = s;
    }
    ++s;
  }


  // Compute data yield
  unsigned long long n_valid_local = num_valid_samples;
  unsigned long long n_valid_global = 0u;
  m_comm->model_allreduce(&n_valid_local, 1, &n_valid_global, El::mpi::SUM);

  if (is_master()) {
    const double yield = static_cast<double>(m_global_num_samples_to_use)/n_valid_global;
    std::cout << "Data yield: " << yield << std::endl;
  }

  std::cout << "rank " << my_rank << '/' << num_ranks
            << " has " << m_local_num_samples_to_use << '/' << m_global_num_samples_to_use
            << " samples to use out of total " << n_valid_local << '/' << n_valid_global
            << " valid local samples." << std::endl;
}

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
  const std::string pattern = data_dir + conduit_file_name;
  std::vector<std::string> filenames = glob(pattern);
  if (filenames.size() < 1) {
    _THROW_LBANN_EXCEPTION_(get_type(), " failed to get data filenames");
  }

  // Shuffle the file names
  if (m_shuffle) {
    std::shuffle(filenames.begin(), filenames.end(), get_data_seq_generator());
  }

  const size_t num_files_to_load =
    (m_max_files_to_load > 0u)? std::min(m_max_files_to_load, filenames.size()) : filenames.size();

  filenames.resize(num_files_to_load);


  double tm1 = get_time();

  // Reserve m_valid_samples
  const size_t my_rank = static_cast<size_t>(m_comm->get_rank_in_model());
  const size_t num_ranks = static_cast<size_t>(m_comm->get_procs_per_model());
  const size_t max_num_files_to_load_per_rank = (num_files_to_load + num_ranks - 1u) / num_ranks;
  bool valid_samples_reserved = false;

  size_t idx = static_cast<size_t>(0ul);
  for (size_t n = my_rank; n < num_files_to_load; n += num_ranks) {
    load_conduit(filenames[n], idx);
    if (!valid_samples_reserved) {
      // reserve the maximum capacity required assuming that files have the same number of samples
      m_valid_samples.reserve(m_data.number_of_children() * max_num_files_to_load_per_rank);
      valid_samples_reserved = true;
    }
    if (is_master()) {
      std::cerr << "time to load: " << n << " files: " << get_time() - tm1 << std::endl;
    }
  }
  if (is_master()) {
    std::cerr << "time to load conduit files: " << get_time() - tm1
              << "  num samples: " << m_data.number_of_children() << std::endl;
  }

  check_image_data();
  determine_num_samples_to_use();

  if (is_master()) {
    std::cout << std::endl << get_description() << std::endl << std::endl;
  }
}
#endif // _JAG_OFFLINE_TOOL_MODE_

void data_reader_jag_conduit::load_conduit(const std::string conduit_file_path, size_t& idx) {
  if (!check_if_file_exists(conduit_file_path)) {
    _THROW_LBANN_EXCEPTION_(get_type(), " failed to open " + conduit_file_path);
  }
#ifndef _JAG_OFFLINE_TOOL_MODE_
  const size_t my_rank = static_cast<size_t>(m_comm->get_rank_in_model());
  std::cerr << ("rank "  + std::to_string(my_rank) + " loading: " + conduit_file_path) << std::endl;
#else
  std::cerr << "loading: " << conduit_file_path << std::endl;
#endif
  conduit::relay::io::load_merged(conduit_file_path, "hdf5", m_data);

  // set up mapping: need to do this since some of the data may be bad
  const std::vector<std::string> &children_names = m_data.child_names();
  size_t bad = 0u;
  for (auto t : children_names) {
    const std::string key = "/" + t + "/performance/success";
    const conduit::Node& n_ok = get_conduit_node(key);
    int success = n_ok.to_int64();
    if (success == 1) {
      m_valid_samples.push_back(t);
    } else {
      ++bad;
    }
  }
  idx = m_valid_samples.size();
  if (is_master()) {
    std::cerr << "data_reader_jag_conduit::load_conduit: num good samples: "
              << m_valid_samples.size() << "  num bad: " << bad << std::endl;
  }

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


size_t data_reader_jag_conduit::get_num_valid_local_samples() const {
  return m_valid_samples.size();
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
#if 1
  return {get_linearized_data_size()};
#else
  std::vector<int> all_dim;
  for (const auto t: m_independent) {
    if (t == Undefined) {
      continue;
    }
    const std::vector<int> ld = get_dims(t);
    all_dim.insert(all_dim.end(), ld.begin(), ld.end());
  }
  return all_dim;
#endif
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
    + " - images: "   + std::to_string(m_num_img_srcs) + " of "
                      + std::to_string(m_image_num_channels) + 'x'
                      + std::to_string(m_image_width) + 'x'
                      + std::to_string(m_image_height) + "\n"
    + " - scalars: "  + std::to_string(get_linearized_scalar_size()) + "\n"
    + " - inputs: "   + std::to_string(get_linearized_input_size()) + "\n"
    + " - linearized data size: "   + std::to_string(get_linearized_data_size()) + "\n"
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


bool data_reader_jag_conduit::check_sample_id(const size_t sample_id) const {
  return (sample_id < m_valid_samples.size());
}

bool data_reader_jag_conduit::check_non_numeric(const std::string key) {
  std::set<std::string>::const_iterator kit = non_numeric_vars.find(key);
  if (kit != non_numeric_vars.cend()) {
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


std::vector< std::pair<size_t, const data_reader_jag_conduit::ch_t*> >
data_reader_jag_conduit::get_image_ptrs(const size_t sample_id) const {
  if (sample_id >= m_valid_samples.size()) {
    _THROW_LBANN_EXCEPTION_(_CN_, "get_image_ptrs() : invalid sample index");
  }

  std::vector< std::pair<size_t, const ch_t*> >image_ptrs;
  image_ptrs.reserve(m_emi_image_keys.size());

  for (const auto& emi_tag : m_emi_image_keys) {
    std::string img_key = m_valid_samples[sample_id] + "/outputs/images/" + emi_tag + "/emi";
    const conduit::Node & n_image = get_conduit_node(img_key);
    conduit_ch_t emi = n_image.value();
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
  if (sample_id >= m_valid_samples.size()) {
    _THROW_LBANN_EXCEPTION_(_CN_, "get_scalars() : invalid sample index");
  }

  const std::string sample_scalars = m_valid_samples[sample_id] + "/outputs/scalars/";

  std::vector<scalar_t> scalars;
  scalars.reserve(m_scalar_keys.size());

  for(const auto key: m_scalar_keys) {
    const std::string scalar_key = sample_scalars + key;
    const conduit::Node & n_scalar = get_conduit_node(scalar_key);
    // All the scalar output currently seems to be scalar_t
    //add_val(key, n_scalar, scalars);
    scalars.push_back(static_cast<scalar_t>(n_scalar.to_value()));
  }
  return scalars;
}

std::vector<data_reader_jag_conduit::input_t> data_reader_jag_conduit::get_inputs(const size_t sample_id) const {
  if (sample_id >= m_valid_samples.size()) {
    _THROW_LBANN_EXCEPTION_(_CN_, "get_inputs() : invalid sample index");
  }

  const std::string sample_inputs = m_valid_samples[sample_id] + "/inputs/";

  std::vector<input_t> inputs;
  inputs.reserve(m_input_keys.size());

  // automatically determine which method to use based on if all the variables are of input_t
  if (m_uniform_input_type) {
    for(const auto key: m_input_keys) {
      const std::string input_key = sample_inputs + key;
      const conduit::Node & n_input = get_conduit_node(input_key);
      inputs.push_back(n_input.value()); // less overhead
    }
  } else {
    for(const auto key: m_input_keys) {
      const std::string input_key = sample_inputs + key;
      const conduit::Node & n_input = get_conduit_node(input_key);
      add_val(key, n_input, inputs); // more overhead but general
    }
  }
  return inputs;
}

int data_reader_jag_conduit::check_exp_success(const std::string sample_key) const {
  if (!m_data.has_path(sample_key)) {
    _THROW_LBANN_EXCEPTION_(_CN_, "check_exp_success() : invalid key to sample: " + sample_key);
  }

  return static_cast<int>(get_conduit_node(sample_key + "/performance/success").value());
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
  if (sample_id >= m_valid_samples.size()) {
    _THROW_LBANN_EXCEPTION_(_CN_, "get_inputs() : invalid sample index");
  }
  const conduit::Node & n = get_conduit_node(m_valid_samples[sample_id]);
  n.schema().print();
}

} // end of namespace lbann

#undef _CN_
#endif // LBANN_HAS_CONDUIT
