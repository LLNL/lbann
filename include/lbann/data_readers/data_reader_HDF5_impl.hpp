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
////////////////////////////////////////////////////////////////////////////////
#ifndef LBANN_DATA_READER_HDF5_REVISED_IMPL_HPP
#define LBANN_DATA_READER_HDF5_REVISED_IMPL_HPP

#include "lbann/data_readers/data_reader_HDF5.hpp"

namespace lbann {

template<typename T_from, typename T_to>
void hdf5_data_reader::coerceme(const T_from* data_in, size_t n_bytes, std::vector<T_to> & data_out) {
  size_t n_elts = n_bytes / sizeof(T_from);
  data_out.resize(0);
  data_out.reserve(n_elts);
  for (size_t j=0; j<n_elts; j++) {
    data_out.push_back(*data_in++);
  }
}

template<typename T>
void hdf5_data_reader::normalizeme(T* data, double scale, double bias, size_t n_bytes) {
  size_t n_elts = n_bytes / sizeof(T);
  for (size_t j=0; j<n_elts; j++) {
    data[j] = ( data[j]*scale+bias );
  }
}

template<typename T>
void hdf5_data_reader::normalizeme(T* data, const double* scale, const double* bias, size_t n_bytes, size_t n_channels) {
  size_t n_elts = n_bytes / sizeof(T);
  size_t n_elts_per_channel = n_elts / n_channels;
  for (size_t j=0; j<n_elts_per_channel; j++) {
    for (size_t k=0; k<n_channels; k++) {
      size_t idx = j*n_channels+k;
      data[idx] = ( data[idx]*scale[k] + bias[k] );
    }
  }
}

template<typename T>
void hdf5_data_reader::repack_image(T* src_buf, size_t n_bytes, size_t n_rows, size_t n_cols, int n_channels) {
  size_t size = n_rows*n_cols;
  size_t n_elts = n_bytes / sizeof(T);
  std::vector<T> work(n_elts);
  T* dst_buf = work.data();
  for (size_t row = 0; row < n_rows; ++row) {
    for (size_t col = 0; col < n_cols; ++col) {
      int N = n_channels;
      // Multiply by N because there are N channels.
      const size_t src_base = N*(row + col*n_rows);
      const size_t dst_base = row + col*n_rows;
      switch(N) {
      case 4:
        dst_buf[dst_base + 3*size] = src_buf[src_base + 3];
        [[fallthrough]];
      case 3:
        dst_buf[dst_base + 2*size] = src_buf[src_base + 2];
        [[fallthrough]];
      case 2:
        dst_buf[dst_base + size] = src_buf[src_base + 1];
        [[fallthrough]];
      case 1:
        dst_buf[dst_base] = src_buf[src_base];
        break;
      default:
        LBANN_ERROR("Unsupported number of channels");
      }
    }
  }
}

template<typename T>
void hdf5_data_reader::pack(std::string group_name, conduit::Node& node, size_t index) {
  if (m_packing_groups.find(group_name) == m_packing_groups.end()) {
    LBANN_ERROR("(m_packing_groups.find(", group_name, ") failed");
  }
  const PackingGroup& g = m_packing_groups[group_name];
  std::vector<T> data(g.n_elts);
  size_t idx = 0;
  for (size_t k=0; k<g.names.size(); k++) {
    size_t n_elts = g.sizes[k];
    std::stringstream ss;
    ss << node.name() << node.child(0).name() + "/" << g.names[k];
    if (!node.has_path(ss.str())) {
      LBANN_ERROR("no leaf for path: ", ss.str());
    }
    conduit::Node& leaf = node[ss.str()];
    memcpy(data.data()+idx, leaf.data_ptr(), n_elts*sizeof(T));
    if (m_delete_packed_fields) {
      node.remove(ss.str());
    }
    idx += n_elts;
  }
  if (idx != g.n_elts) {
    LBANN_ERROR("idx != g.n_elts*sizeof(T): ", idx, " ", g.n_elts*sizeof(T));
  }
  std::stringstream ss;
  ss << '/' << LBANN_DATA_ID_STR(index) + '/' + group_name;
  node[ss.str()] = data;

  // this is clumsy and should be done better
  if (m_add_to_map.find(group_name) == m_add_to_map.end()) {
    m_add_to_map.insert(group_name);
    conduit::Node metadata;
    metadata[s_composite_node] = true;
    m_experiment_schema[group_name][s_metadata_node_name] = metadata;
    m_data_schema[group_name][s_metadata_node_name] = metadata;
    m_useme_node_map[group_name] = m_experiment_schema[group_name];
    m_useme_node_map_ptrs[group_name] = &(m_experiment_schema[group_name]);
  }
}

} // namespace lbann

#endif // LBANN_DATA_READER_HDF5_REVISED_IMPL_HPP
