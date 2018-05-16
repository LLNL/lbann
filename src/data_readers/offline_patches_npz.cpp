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

#include "lbann/data_readers/offline_patches_npz.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/cnpy_utils.hpp"
#include <set>

namespace lbann {

offline_patches_npz::offline_patches_npz()
  : m_checked_ok(false), m_num_patches(3u), m_variant_divider(".JPEG.")
{}


bool offline_patches_npz::load(const std::string filename, size_t first_n) {
  m_item_class_list.clear();
  m_file_root_list.clear();
  m_file_variant_list.clear();

  // The list of arrays expected to be packed in the file.
  // 'max_class' and 'variant_divider' are scalar values, and known in advance.
  const std::set<std::string> dict =
    {"item_root_list",
     "item_variant_list",
     "item_class_list",
     "file_root_list",
     "file_variant_list",
     "max_class",
     "variant_divider"};

  if (!check_if_file_exists(filename)) {
    return false;
  }
  cnpy::npz_t dataset = cnpy::npz_load(filename);

  // check if all the arrays are included
  for (const auto& np : dataset) {
    if (dict.find(np.first) == dict.end()) {
      return false;
    }
  }

  if (first_n > 0u) { // to use only first_n samples
    cnpy_utils::shrink_to_fit(dataset["item_root_list"], first_n);
    cnpy_utils::shrink_to_fit(dataset["item_variant_list"], first_n);
  }
  // Set the array of index sequences for root type
  m_item_root_list = dataset["item_root_list"];

  if (first_n > 0u) { // to use only first_n samples
  }
  // Set the array of index sequences for variant type
  m_item_variant_list = dataset["item_variant_list"];

  { // load the label array into a vector of label_t (uint8_t)
    cnpy::NpyArray d_item_class_list = dataset["item_class_list"];
    m_checked_ok = (d_item_class_list.shape.size() == 1u);
    if (m_checked_ok) {
      // In case of shrinking to first_n, make sure the size is consistent
      const size_t num_samples = m_item_root_list.shape[0];
      m_item_class_list.resize(num_samples);
      for (size_t i=0u; i < num_samples; ++i) {
        std::string digits(cnpy_utils::data_ptr<char>(d_item_class_list, {i}), d_item_class_list.word_size);
        m_item_class_list[i] = static_cast<label_t>(atoi(digits.c_str()));
      }
    }
    cnpy::npz_t::iterator it = dataset.find("item_class_list");
    dataset.erase(it); // to keep memory footprint as low as possible
  }

  { // load the array of dictionary substrings of root type
    cnpy::NpyArray d_file_root_list = dataset["file_root_list"];
    m_checked_ok = m_checked_ok && (d_file_root_list.shape.size() == 1u);
    if (m_checked_ok) {
      const size_t num_roots = d_file_root_list.shape[0];
      m_file_root_list.resize(num_roots);
      for (size_t i=0u; i < num_roots; ++i) {
        std::string file_root(cnpy_utils::data_ptr<char>(d_file_root_list, {i}), d_file_root_list.word_size);
        m_file_root_list[i] = std::string(file_root.c_str());
      }
    }
    cnpy::npz_t::iterator it = dataset.find("file_root_list");
    dataset.erase(it); // to keep memory footprint as low as possible
  }
  //for (const auto& fl: m_file_root_list) std::cout << fl << std::endl;

  { // load the array of dictionary substrings of variant type
    cnpy::NpyArray d_file_variant_list = dataset["file_variant_list"];
    m_checked_ok = m_checked_ok && (d_file_variant_list.shape.size() == 1u);
    if (m_checked_ok) {
      const size_t num_variants = d_file_variant_list.shape[0];
      m_file_variant_list.resize(num_variants);
      for (size_t i=0u; i < num_variants; ++i) {
        std::string file_variant(cnpy_utils::data_ptr<char>(d_file_variant_list, {i}), d_file_variant_list.word_size);
        m_file_variant_list[i] = std::string(file_variant.c_str());
      }
    }
    cnpy::npz_t::iterator it = dataset.find("file_variant_list");
    dataset.erase(it); // to keep memory footprint as low as possible
  }
  //for (const auto& fl: m_file_variant_list) std::cout << fl << std::endl;

  m_checked_ok = m_checked_ok && check_data();

  if (!m_checked_ok) {
    //std::cout << get_description();
    m_item_class_list.clear();
    m_file_root_list.clear();
    m_file_variant_list.clear();
    throw lbann_exception("offline_patches_npz: loaded data not consistent");
  }

  return m_checked_ok;
}


bool offline_patches_npz::check_data() const {
  bool ok = (m_item_root_list.shape.size() == 2u) &&
            (m_item_variant_list.shape.size() == 3u) &&
            (m_file_root_list.size() > 0u) &&
            (m_file_variant_list.size() > 0u) &&
            (m_item_root_list.shape[0] == m_item_class_list.size()) &&
            (m_item_variant_list.shape[0] == m_item_class_list.size()) &&
            (m_item_root_list.shape[1] == m_num_patches) &&
            (m_item_variant_list.shape[1] == m_num_patches) &&
            (m_item_variant_list.shape[2] > 0u) &&
            (m_item_root_list.word_size == sizeof(size_t)) &&
            (m_item_variant_list.word_size == sizeof(size_t));
  return ok;
}


std::string offline_patches_npz::get_description() const {
  using std::string;
  using std::to_string;
  string ret = string("offline_patches_npz:\n")
    + " - item_root_list: "    + cnpy_utils::show_shape(m_item_root_list) + "\n"
    + " - item_variant_list: " + cnpy_utils::show_shape(m_item_variant_list) + "\n"
    + " - item_class_list: "   + to_string(m_item_class_list.size()) + "\n"
    + " - file_root_list: "    + to_string(m_file_root_list.size()) + "\n"
    + " - file_variant_list: " + to_string(m_file_variant_list.size()) + "\n"
    + " - variant_divider: "   + m_variant_divider + "\n"
    + " - num of samples: "    + to_string(get_num_samples()) + "\n"
    + " - num of patches: "    + to_string(m_num_patches) + "\n";
  return ret;
}


offline_patches_npz::sample_t offline_patches_npz::get_sample(const size_t idx) const {
  if (!m_checked_ok || idx >= get_num_samples()) {
    throw lbann_exception("offline_patches_npz: invalid sample index");
  }

  std::vector<std::string> file_names;

  for (size_t p = 0u; p < m_num_patches; ++p) {
    const size_t root = cnpy_utils::data<size_t>(m_item_root_list, {idx, p});
    if (root >= m_file_root_list.size()) {
      using std::to_string;
      throw lbann_exception("offline_patches_npz: invalid file_root_list index: "
                          + to_string(root) + " >= " + to_string(m_file_root_list.size()));
    }
    std::string file_name = m_file_root_list.at(root);

    const size_t* variant = &(cnpy_utils::data<size_t>(m_item_variant_list, {idx, p, 0u}));
    const int ve = m_item_variant_list.shape.back()-1;
    for (int i = 0; i < ve; ++i) {
      file_name += m_file_variant_list.at(variant[i]) + m_variant_divider;
    }

    file_name += m_file_variant_list.at(variant[ve]);
    file_names.push_back(file_name);
  }
  return std::make_pair(file_names, m_item_class_list[idx]);
}


offline_patches_npz::label_t offline_patches_npz::get_label(const size_t idx) const {
  if (!m_checked_ok || idx >= get_num_samples()) {
    throw lbann_exception("offline_patches_npz: invalid sample index");
  }

  return m_item_class_list[idx];
}


#ifdef _OFFLINE_PATCHES_NPZ_OFFLINE_TOOL_MODE_
/// count samples of first n roots
size_t offline_patches_npz::count_samples(const size_t num_roots) const {
  if (!m_checked_ok || num_roots > m_file_root_list.size()) {
    throw lbann_exception("invalid sample index");
  }

  std::vector<std::string> file_names;
  size_t num_samples = 0u;
  const size_t total_samples = get_num_samples();

  for (size_t s = 0u; s < total_samples; ++s) {
    const size_t root = cnpy_utils::data<size_t>(m_item_root_list, {s, 0});
    if (root >= m_file_root_list.size()) {
      throw lbann_exception("invalid file_root_list index");
    }
    if (root >= num_roots) break;
    num_samples ++;
  }
  return num_samples;
}

std::vector<std::string> offline_patches_npz::get_file_roots() const {
  std::vector<std::string> root_names;
  const size_t num_samples = get_num_samples();
  root_names.reserve(num_samples);
  for (size_t i = 0u; i < num_samples; ++i) {
    const size_t root = cnpy_utils::data<size_t>(m_item_root_list, {i, 0});
    if (root >= m_file_root_list.size()) {
      throw lbann_exception("invalid file_root_list index");
    }
    std::string file_name = m_file_root_list.at(root);
    root_names.push_back(file_name);
  }
  return root_names;
}
#endif // _OFFLINE_PATCHES_NPZ_OFFLINE_TOOL_MODE_

} // end of namespace lbann
