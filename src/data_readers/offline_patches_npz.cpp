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

namespace lbann {

offline_patches_npz::offline_patches_npz()
  : m_checked_ok(false), m_num_patches(3u), m_variant_divider(".JPEG.")
{}


bool offline_patches_npz::load(const std::string filename) {
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

  cnpy::npz_t dataset = cnpy::npz_load(filename);

  // check if all the arrays are included
  for (const auto& np : dataset) {
    if (dict.find(np.first) == dict.end()) {
      return false;
    }
  }

  // Load the array of index sequences for root type into a cnpy structure
  m_item_root_list = dataset["item_root_list"];
  // Load the array of index sequences for variant type into a cnpy structure
  m_item_variant_list = dataset["item_variant_list"];

  { // load the label array into a vector of label_t (uint8_t)
    cnpy::NpyArray d_item_class_list = dataset["item_class_list"];
    m_checked_ok = (d_item_class_list.shape.size() == 1u);
    if (m_checked_ok) {
      const size_t num_samples = d_item_class_list.shape[0];
      m_item_class_list.resize(num_samples);
      for (size_t i=0u; i < num_samples; ++i) {
        std::string digits(data_ptr<char>(d_item_class_list, {i}), d_item_class_list.word_size);
        m_item_class_list[i] = static_cast<label_t>(atoi(digits.c_str()));
      }
    }
  }

  { // load the array of dictionary substrings of root type
    cnpy::NpyArray d_file_root_list = dataset["file_root_list"];
    m_checked_ok = m_checked_ok && (d_file_root_list.shape.size() == 1u);
    if (m_checked_ok) {
      const size_t num_roots = d_file_root_list.shape[0];
      m_file_root_list.resize(num_roots);
      for (size_t i=0u; i < num_roots; ++i) {
        std::string file_root(data_ptr<char>(d_file_root_list, {i}), d_file_root_list.word_size);
        m_file_root_list[i] = std::string(file_root.c_str());
      }
    }
  }
  //for (const auto& fl: m_file_root_list) std::cout << fl << std::endl;

  { // load the array of dictionary substrings of variant type
    cnpy::NpyArray d_file_variant_list = dataset["file_variant_list"];
    m_checked_ok = m_checked_ok && (d_file_variant_list.shape.size() == 1u);
    if (m_checked_ok) {
      const size_t num_variants = d_file_variant_list.shape[0];
      m_file_variant_list.resize(num_variants);
      for (size_t i=0u; i < num_variants; ++i) {
        std::string file_variant(data_ptr<char>(d_file_variant_list, {i}), d_file_variant_list.word_size);
        m_file_variant_list[i] = std::string(file_variant.c_str());
      }
    }
  }
  //for (const auto& fl: m_file_variant_list) std::cout << fl << std::endl;

  m_checked_ok = m_checked_ok && check_data();

  if (!m_checked_ok) {
    m_item_class_list.clear();
    m_file_root_list.clear();
    m_file_variant_list.clear();
    throw lbann_exception("loaded data not consistent");
  }

  return m_checked_ok;
}


bool offline_patches_npz::check_data() const {
  bool ok = (m_item_root_list.shape.size() == 2u) &&
            (m_item_variant_list.shape.size() == 3u) &&
            (m_file_root_list.size() > 0u) &&
            (m_file_variant_list.size() > 0u) &&
            (m_item_root_list.shape[0] == m_item_class_list.size()) &&
            (m_item_variant_list.shape[0] == m_item_class_list.size());
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
    + " - item_root_list: "    + show_shape(m_item_root_list) + "\n"
    + " - item_variant_list: " + show_shape(m_item_variant_list) + "\n"
    + " - item_class_list: "   + to_string(m_item_class_list.size()) + "\n"
    + " - file_root_list: "    + to_string(m_file_root_list.size()) + "\n"
    + " - file_variant_list: " + to_string(m_file_variant_list.size()) + "\n"
    + " - variant_divider: "   + m_variant_divider + "\n"
    + " - num of samples: "    + to_string(get_num_samples()) + "\n"
    + " - num of patches: "    + to_string(m_num_patches) + "\n";
  return ret;
}


std::string offline_patches_npz::show_shape(const cnpy::NpyArray& na) {
  std::string ret;
  for (const size_t s: na.shape) {
    ret += std::to_string(s) + 'x';
  }
  if (ret.size() == 0u) {
    return "empty";
  } else {
    ret.pop_back(); // remove the last 'x'
    ret += " " + std::to_string(na.word_size);
  }
  return ret;
}


offline_patches_npz::sample_t offline_patches_npz::get_sample(const size_t idx) const {
  if (!m_checked_ok || idx >= get_num_samples()) {
    throw lbann_exception("invalid sample index");
  }

  std::vector<std::string> file_names;

  for (size_t p = 0u; p < m_num_patches; ++p) {
    const size_t root = data<size_t>(m_item_root_list, {idx, p});
    if (root >= m_file_root_list.size()) {
      throw lbann_exception("invalid file_root_list index");
    }
    std::string file_name = m_file_root_list.at(root);

    const size_t* variant = &(data<size_t>(m_item_variant_list, {idx, p, 0u}));
    const int ve = m_item_variant_list.shape.back()-1;
    for (int i = 0; i < ve; ++i) {
      file_name += m_file_variant_list.at(variant[i]) + m_variant_divider;
    }

    file_name += m_file_variant_list.at(variant[ve]);
    file_names.push_back(file_name);
  }
  return std::make_pair(file_names, m_item_class_list[idx]);
}


size_t offline_patches_npz::compute_cnpy_array_offset(
  const cnpy::NpyArray& na, const std::vector<size_t> indices) {

  bool ok = (indices.size() == na.shape.size());
  size_t unit_stride = 1u;
  size_t offset = 0u;

  for (size_t i = indices.size(); ok && (i-- > 0u); ) {
    ok = (indices[i] < na.shape[i]);
    offset += indices[i] * unit_stride;
    unit_stride *= na.shape[i];
  }
  if (!ok) {
    throw lbann_exception("invalid data index");
  }
  return offset;
}

} // end of namespace lbann
