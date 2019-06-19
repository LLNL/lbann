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

#include "lbann/data_readers/offline_patches_npz.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/cnpy_utils.hpp"
#include <set>
#include <algorithm>

#include <iostream>

namespace lbann {

offline_patches_npz::offline_patches_npz(size_t npatches, std::string divider)
  : m_checked_ok(false), m_lbann_format(false)
{
  m_num_patches = npatches;
  m_variant_divider = divider;
}

offline_patches_npz::offline_patches_npz(size_t npatches)
  : m_checked_ok(false), m_lbann_format(false)
{
  m_num_patches = npatches;
  m_variant_divider = ".JPEG.";
}

offline_patches_npz::offline_patches_npz(std::string divider)
  : m_checked_ok(false), m_lbann_format(false)
{
  m_num_patches = 3u;
  m_variant_divider = divider;
}

offline_patches_npz::offline_patches_npz()
  : m_checked_ok(false), m_num_patches(3u), m_variant_divider(".JPEG."),
    m_lbann_format(false)
{}

bool offline_patches_npz::load(const std::string filename, size_t first_n,
  bool keep_file_lists) {
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

  m_lbann_format = false;

  // check if all the arrays are included
  for (const auto& np : dataset) {
    if (dict.find(np.first) == dict.end()) {
      if (np.first == "lbann_format") {
        cnpy::NpyArray d_lbann_format = dataset["lbann_format"];
        m_lbann_format = cnpy_utils::data<bool>(d_lbann_format, {0});
      } else {
        return false;
      }
    }
  }

  if (first_n > 0u) { // to use only first_n samples
    cnpy_utils::shrink_to_fit(dataset["item_root_list"], first_n);
    cnpy_utils::shrink_to_fit(dataset["item_variant_list"], first_n);
  }
  // Set the array of index sequences for root type
  m_item_root_list = dataset["item_root_list"];

  // Set the array of index sequences for variant type
  m_item_variant_list = dataset["item_variant_list"];

  { // load the label array into a vector of label_t (uint8_t)
    cnpy::NpyArray d_item_class_list = dataset["item_class_list"];
    m_checked_ok = (d_item_class_list.shape.size() == 1u);

    if (m_checked_ok) {
      // In case of shrinking to first_n, make sure the size is consistent
      const size_t num_samples = m_item_root_list.shape[0];
      if (m_lbann_format) {
        const label_t* ptr = cnpy_utils::data_ptr<label_t>(d_item_class_list, {0});
        m_item_class_list.assign(ptr, ptr + num_samples);
      } else {
        m_item_class_list.resize(num_samples);
        for (size_t i=0u; i < num_samples; ++i) {
          std::string digits(cnpy_utils::data_ptr<char>(d_item_class_list, {i}), d_item_class_list.word_size);
          m_item_class_list[i] = static_cast<label_t>(atoi(digits.c_str()));
        }
      }
    }
    cnpy::npz_t::iterator it = dataset.find("item_class_list");
    dataset.erase(it); // to keep memory footprint as low as possible
  }

  { // load the array of dictionary substrings of root type
    cnpy::NpyArray d_file_root_list = dataset["file_root_list"];
    m_checked_ok = m_checked_ok &&
                   ( (d_file_root_list.shape.size() == 1u) ||
                    ((d_file_root_list.shape.size() == 2u) && m_lbann_format));
    if (m_checked_ok) {
      const size_t num_roots = d_file_root_list.shape[0];
      m_file_root_list.resize(num_roots);

      const size_t len = (m_lbann_format? d_file_root_list.shape[1]
                                        : d_file_root_list.word_size);

      for (size_t i=0u; i < num_roots; ++i) {
        std::string file_root(cnpy_utils::data_ptr<char>(d_file_root_list, {i}), len);
        m_file_root_list[i] = std::string(file_root.c_str()); // to remove the trailing spaces
      }
    }
    if (keep_file_lists) {
      m_file_root_list_org = d_file_root_list;
    } else {
      cnpy::npz_t::iterator it = dataset.find("file_root_list");
      dataset.erase(it); // to keep memory footprint as low as possible
    }
  }

  { // load the array of dictionary substrings of variant type
    cnpy::NpyArray d_file_variant_list = dataset["file_variant_list"];
    m_checked_ok = m_checked_ok &&
                   ( (d_file_variant_list.shape.size() == 1u) ||
                    ((d_file_variant_list.shape.size() == 2u) && m_lbann_format));
    if (m_checked_ok) {
      const size_t num_variants = d_file_variant_list.shape[0];
      m_file_variant_list.resize(num_variants);

      const size_t len = (m_lbann_format? d_file_variant_list.shape[1]
                                        : d_file_variant_list.word_size);

      for (size_t i=0u; i < num_variants; ++i) {
        std::string file_variant(cnpy_utils::data_ptr<char>(d_file_variant_list, {i}), len);
        m_file_variant_list[i] = std::string(file_variant.c_str());
      }
    }
    if (keep_file_lists) {
      m_file_root_list_org = d_file_variant_list;
    } else {
      cnpy::npz_t::iterator it = dataset.find("file_variant_list");
      dataset.erase(it); // to keep memory footprint as low as possible
    }
  }

  m_checked_ok = m_checked_ok && check_data();

  if (!m_checked_ok) {
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


bool offline_patches_npz::select(const std::string out_file, const size_t sample_start, size_t& sample_end) {
  if ( sample_start >= sample_end) {
    std::cerr << "sample_end (" << sample_end
              << ") is not larger than sample_start ("
              << sample_start << ")." << std::endl;
    return false;
  }

  // Set the array of index sequences for root type
  const size_t num_samples_org = m_item_root_list.shape[0];
  if (sample_end > num_samples_org) {
    std::cerr << "sample_end exceed the number of of samples in data."
              << "Adjusting it to the number of existing samples." << std::endl;
    sample_end = num_samples_org;
  }

  { // create output directory if needed
    std::string out_dir;
    std::string out_filename;
    parse_path(out_file, out_dir, out_filename);

    if (!check_if_dir_exists(out_dir)) {
      create_dir(out_dir);
    }
  }

  std::pair<size_t, size_t> file_root_range;
  std::pair<size_t, size_t> file_variant_range;

  { // write item_root_list
    const size_t* data_ptr = cnpy_utils::data_ptr<size_t>(m_item_root_list, {sample_start});
    size_t* out_ptr        = cnpy_utils::data_ptr<size_t>(m_item_root_list, {sample_start});
    const size_t* data_ptr_end = cnpy_utils::data_ptr<size_t>(m_item_root_list, {sample_end});

    // compute the min-max range of indieces reference
    auto result = std::minmax_element(data_ptr, data_ptr_end);
    file_root_range = std::make_pair(*result.first, *result.second + 1);

    // adjust indices by subtracting file_root_range.first from each
    std::transform(data_ptr, data_ptr_end, out_ptr, [&](size_t id) -> size_t { return (id - file_root_range.first); });

    // write the updated array into the output file
    std::vector<size_t> out_shape = m_item_root_list.shape;
    out_shape[0] = sample_end - sample_start;
    cnpy::npz_save(out_file, "item_root_list", data_ptr, out_shape, "w");
  }

  { // write item_variant_list
    const size_t* data_ptr = cnpy_utils::data_ptr<size_t>(m_item_variant_list, {sample_start});
    size_t* out_ptr        = cnpy_utils::data_ptr<size_t>(m_item_variant_list, {sample_start});
    const size_t* data_ptr_end = cnpy_utils::data_ptr<size_t>(m_item_variant_list, {sample_end});

    // compute the min-max range of indieces reference
    auto result = std::minmax_element(data_ptr, data_ptr_end);
    file_variant_range = std::make_pair(*result.first, *result.second + 1);

    // adjust indices by subtracting file_variant_range.first from each
    std::transform(data_ptr, data_ptr_end, out_ptr, [&](size_t id) -> size_t { return (id - file_variant_range.first); });

    // write the updated array into the output file
    std::vector<size_t> out_shape = m_item_variant_list.shape;
    out_shape[0] = sample_end - sample_start;
    cnpy::npz_save(out_file, "item_variant_list", data_ptr, out_shape, "a");
  }

  { // write item_class_list
    const label_t* data_ptr = &(m_item_class_list[sample_start]);
    cnpy::npz_save(out_file, "item_class_list", data_ptr, {sample_end - sample_start}, "a");
  }

  { // load the array of dictionary substrings of root type
    cnpy::NpyArray org_list = m_file_root_list_org;
    bool org_readable = (((!m_lbann_format && (org_list.shape.size() == 1u)) &&
                           ( m_lbann_format && (org_list.shape.size() == 2u))) &&
                          (org_list.shape[0] > 0u));

    std::vector<size_t> out_shape(2u);
    const size_t id_start = file_root_range.first;
    const size_t id_end = file_root_range.second;
    out_shape[0] = id_end - id_start;
    size_t len = 0u;

    std::vector<char> tmp;
    char* data_ptr = nullptr;

    if (org_readable) {
      len = (m_lbann_format? org_list.shape[1] : org_list.word_size);
      data_ptr = cnpy_utils::data_ptr<char>(org_list, {id_start});
    } else {
      for (size_t i = id_start; i < id_end; ++i) {
        size_t sz = m_file_root_list[i].size();
        if (len < sz) {
          len = sz;
        }
      }
      tmp.resize((id_end - id_start)*len, '\0');
      data_ptr = &(tmp[0]);
      for (size_t i = id_start; i < id_end; ++i) {
        const std::string& str = m_file_root_list[i];
        std::copy(str.begin(), str.end(), data_ptr);
        data_ptr += len;
      }
      data_ptr = &(tmp[0]);
    }
    out_shape[1] = len;
    cnpy::npz_save(out_file, "file_root_list", data_ptr, out_shape, "a");
  }

  { // load the array of dictionary substrings of variant type
    cnpy::NpyArray org_list = m_file_variant_list_org;
    bool org_readable = (((!m_lbann_format && (org_list.shape.size() == 1u)) &&
                           ( m_lbann_format && (org_list.shape.size() == 2u))) &&
                          (org_list.shape[0] > 0u));

    std::vector<size_t> out_shape(2u);
    const size_t id_start = file_variant_range.first;
    const size_t id_end = file_variant_range.second;
    out_shape[0] = id_end - id_start;
    size_t len = 0u;

    std::vector<char> tmp;
    char* data_ptr = nullptr;

    if (org_readable) {
      len = (m_lbann_format? org_list.shape[1] : org_list.word_size);
      data_ptr = cnpy_utils::data_ptr<char>(org_list, {id_start});
    } else {
      for (size_t i = id_start; i < id_end; ++i) {
        size_t sz = m_file_variant_list[i].size();
        if (len < sz) {
          len = sz;
        }
      }
      tmp.resize((id_end - id_start)*len, '\0');
      data_ptr = &(tmp[0]);
      for (size_t i = id_start; i < id_end; ++i) {
        const std::string& str = m_file_variant_list[i];
        std::copy(str.begin(), str.end(), data_ptr);
        data_ptr += len;
      }
      data_ptr = &(tmp[0]);
    }
    out_shape[1] = len;
    cnpy::npz_save(out_file, "file_variant_list", data_ptr, out_shape, "a");
  }

  {
    bool lbann_format = true;
    cnpy::npz_save(out_file, "lbann_format", &lbann_format, {1}, "a");
  }

  return true;
}
#endif // _OFFLINE_PATCHES_NPZ_OFFLINE_TOOL_MODE_

} // end of namespace lbann
