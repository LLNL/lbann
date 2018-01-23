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
////////////////////////////////////////////////////////////////////////////////

#ifndef OFFLINE_PATCHES_NPZ_H
#define OFFLINE_PATCHES_NPZ_H

#include "cnpy.h"
#include <iostream>
#include <string>
#include <set>
#include <vector>
#include "lbann/utils/exception.hpp"

namespace lbann {

class offline_patches_npz {
 public:
  using label_t = uint8_t;
  using sample_t = std::pair<std::vector<std::string>, label_t>;

  offline_patches_npz();
  bool load(const std::string filename);
  std::string get_description() const;

  size_t get_num_samples() const {
    return m_item_class_list.size();
  }
  size_t get_num_patches() const {
    return m_num_patches;
  }
  /// Return the meta-data (patch file names and the label) of idx-th sample
  sample_t get_sample(const size_t idx) const;
  /// Return the label of idx-th sample
  label_t get_label(const size_t idx) const;

 protected:
  bool check_data() const;
  static std::string show_shape(const cnpy::NpyArray& na);

  static size_t compute_cnpy_array_offset(const cnpy::NpyArray& na, const std::vector<size_t> indices);

  template<typename T>
  T& data(const cnpy::NpyArray& na, const std::vector<size_t> indices) const;

  template<typename T>
  T* data_ptr(const cnpy::NpyArray& na, const std::vector<size_t> indices) const;

 protected:
  /// Whether loaded data have passed the format check
  bool m_checked_ok;
  /// The number of image patches per sample (i.e. the num of patch files to read)
  size_t m_num_patches;
  /// index to the list of common file path substrings (m_file_root_list) per patch file (dimension: num_samples * num_patches)
  cnpy::NpyArray m_item_root_list;
  cnpy::NpyArray m_item_variant_list;
  /// list of labels (dimension: num_samples)
  std::vector<label_t> m_item_class_list;
  /// The list of common substrings that a file path starts with (dimension is, for example 1000 in case of imagenet data)
  std::vector<std::string> m_file_root_list;
  
  std::vector<std::string> m_file_variant_list;
  /// The text file name of file_root_list
  std::string m_file_root_list_name;
  /// The text file name of file_variant_list
  std::string m_file_variant_list_name;
  /// A substring after which the file name of variants begins to differ (e.g., ".JPEG.")
  std::string m_variant_divider;
  /// control how the text dictionary files are loaded: whether to load all at once and parse or to stream in
  bool m_fetch_text_dict_at_once;
};


inline offline_patches_npz::label_t offline_patches_npz::get_label(const size_t idx) const {
  return m_item_class_list[idx];
}

template<typename T>
inline T& offline_patches_npz::data(
  const cnpy::NpyArray& na, const std::vector<size_t> indices) const {
  if ((sizeof(T) != na.word_size) && (sizeof(T) != 1u)) {
    throw lbann_exception("The data type is not consistent with the word size of the array.");
  }
  const size_t offset = compute_cnpy_array_offset(na, indices)
                        * ((sizeof(T) == 1u)? na.word_size : 1u);
  return *(reinterpret_cast<T*>(&(* na.data_holder)[0]) + offset);
}

template<typename T>
inline T* offline_patches_npz::data_ptr(
  const cnpy::NpyArray& na, const std::vector<size_t> indices) const {
  if ((sizeof(T) != na.word_size) && (sizeof(T) != 1u)) {
    throw lbann_exception("The data type is not consistent with the word size of the array.");
  }
  const size_t offset = compute_cnpy_array_offset(na, indices)
                        * ((sizeof(T) == 1u)? na.word_size : 1u);
  return (reinterpret_cast<T*>(&(* na.data_holder)[0]) + offset);
}

} // end of namespace lbann
#endif // OFFLINE_PATCHES_NPZ_H
