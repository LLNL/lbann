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

/**
 * Loads the list of patche files, generated off-line, and the label per sample.
 * As the list is quite large itself in the ASCII text format, it is packed and
 * loaded as a compressed NumPy file (*.npz).
 * Each image file name is compressed further by representing it as a sequence of
 * indices to common substring dictionaries. There are two types of substring
 * dictionaries, root and variant. There is an array of index sequences and an
 * array of dictionary substrings per type, and a label array.
 * For example, a file path train/n000111/abc.tag1.tag2.jpg would be represented
 * as 'r[i][j][k]', 'v[i][j][x]', 'v[i][j][y]', 'v[i][j][z]' for the j-th patch
 * of the i-th sample where 'r[i][j][k]' is "train/n000111", and 'v[i][j][x]',
 * 'v[i][j][y]' and 'v[i][j][z]' is "abc", "tag1", and "tag2" respectively.
 * 'r' is the root dictionary and 'v' is the variant dictionary.
 * The list is kept in a compressed form, and uncompressed on-demand during execution.
 * Each index sequence array is kept as a CNPY data structure, and each dictionary
 * array is loaded into a vector of strings. The label array is loaded into a
 * vector of uint8_t.
 */
class offline_patches_npz {
 public:
  using label_t = uint8_t;
  using sample_t = std::pair<std::vector<std::string>, label_t>;

  offline_patches_npz();
  // TODO: copy constructor and assignment operator for deep-copying if needed
  // The cnpy structure relies on shared_ptr

  /**
   * Load the data in the compressed numpy format file.
   * Use only first_n available samples if specified.
   */
  bool load(const std::string filename, size_t first_n = 0u);
  /// Show the description
  std::string get_description() const;

  /// Return the number of samples
  size_t get_num_samples() const {
    return m_item_class_list.size();
  }
  /// Return the number of patches per sample (the number of image data sources)
  size_t get_num_patches() const {
    return m_num_patches;
  }
  /// Reconsturct and return the meta-data (patch file names and the label) of idx-th sample
  sample_t get_sample(const size_t idx) const;
  /// Return the label of idx-th sample
  label_t get_label(const size_t idx) const;

 protected:
  /// Check the dimensions of loaded data
  bool check_data() const;
  /// Show the dimensions of loaded data
  static std::string show_shape(const cnpy::NpyArray& na);

  /**
   * Return the offset to the element (in terms of the number of elements from
   * the beginning of the array) of a loaded numpy array na specified by indices
   */
  static size_t compute_cnpy_array_offset(const cnpy::NpyArray& na, const std::vector<size_t> indices);

  /**
   * Allow the access to the data element identified by the indices and the
   * word_size of the array na, but present it as a type T element at the address.
   */
  template<typename T>
  T& data(const cnpy::NpyArray& na, const std::vector<size_t> indices) const;

  /**
   * Return the address of the data element identified by the indices and the
   * word_size of the array na, but present it as the address of a type T element
   */
  template<typename T>
  T* data_ptr(const cnpy::NpyArray& na, const std::vector<size_t> indices) const;

  /**
   * Shrink the first dimension of cnpy::NpyArray to the given size.
   * This is used to choose only first sz samples in data.
   */
  void shrink_to_fit(cnpy::NpyArray& na, size_t sz);

 protected:
  /// Whether loaded data have passed the format check
  bool m_checked_ok;
  /// The number of image patches per sample (i.e. the num of patch files to read)
  size_t m_num_patches;
  /**
   * List of index sequences to the dictionary of common file path substrings (m_file_root_list)
   * per patch file (dimension: num_samples * num_patches)
   */
  cnpy::NpyArray m_item_root_list;
  /**
   * List of index sequences to the dictionary of common file path substrings (m_file_variant_list)
   * per patch file (dimension: num_samples * num_patches)
   */
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
