//////////////////////////////////////////////////////////////////////////////
//
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
#ifndef __REVISED_LBANN_DATA_READER_HDF5_HPP__
#define __REVISED_LBANN_DATA_READER_HDF5_HPP__

#include "data_reader_sample_list.hpp"
#include "lbann/data_store/data_store_conduit.hpp"

namespace lbann {
/**
 * A generalized data reader for data stored in HDF5 files.
 */
class hdf5_data_reader : public data_reader_sample_list {
public:
  hdf5_data_reader(bool shuffle = true);
  hdf5_data_reader(const hdf5_data_reader&);
  hdf5_data_reader& operator=(const hdf5_data_reader&);
  hdf5_data_reader* copy() const override { return new hdf5_data_reader(*this); }
  void copy_members(const hdf5_data_reader &rhs);
  ~hdf5_data_reader() override {}

  std::string get_type() const override {
    return "hdf5_data_reader";
  }

  void load() override;

  /** Returns a raw pointer to the data. 
   * The data fields that are returned are determined by the
   * "pack" field in the "lbann_metadata" subtree in the user supplied schema
   */
  const unsigned char* get_raw_data(const size_t sample_id, const std::string &field_name, size_t &num_bytes) const {
    const conduit::Node &node = m_data_store->get_conduit_node(sample_id);
    num_bytes = node.allocated_bytes();
    std::stringstream ss;
    ss << '/' << LBANN_DATA_ID_STR(sample_id) + '/' + field_name;
    return node[ss.str()].as_unsigned_char_ptr();
  }

  /** Returns a raw pointer to the data.  */
  const DataType* get_data(const size_t sample_id, const std::string &field_name, size_t &num_bytes) const {
  #if 0
    const conduit::Node &node = m_data_store->get_conduit_node(sample_id);
    num_bytes = node.allocated_bytes();
    std::stringstream ss;
    ss << '/' << LBANN_DATA_ID_STR(sample_id) + '/' + field_name;
char *x = "asdf";
    return static_cast<DataType*>(x);
    //return static_cast<DataType*>(node[ss.str()].as_unsigned_char_ptr());
  #endif    
    return nullptr;
  }

  /** returns the number of elements (not bytes) for the datum
   * associated with the 'key'
   * Ceveloper's note: if this is called frequently we should
   * perhaps cache the values
   */
  int get_linearized_size(const std::string &key) const override;

  // TODO, perhaps
  // should go away in future?
  int fetch_data(CPUMat& X, El::Matrix<El::Int>& indices_fetched) override;

  // TODO, perhaps
  // should go away in future?
  int fetch_responses(CPUMat& Y) override {
    LBANN_ERROR("fetch_response() is not implemented");
  }

  // TODO, perhaps
  // should go away in future?
  int fetch_labels(CPUMat& Y) override {
    LBANN_ERROR("fetch_labels() is not implemented");
  }

  void set_experiment_schema_filename(std::string fn) {
    m_experiment_schema_filename = fn;
  }
  const std::string& get_experiment_schema_filename() {
    return m_experiment_schema_filename;
  }

  void set_data_schema_filename(std::string fn) {
    m_data_schema_filename = fn;
  }
  const std::string& get_data_schema_filename() {
    return m_data_schema_filename;
  }

private:

  std::string m_experiment_schema_filename;

  std::string m_data_schema_filename;

  // to set to false, use the cmd line flag: --keep_packed_fields
  // I don't know a use case for keeping the original (possibly coerced)
  // field if it's been packed, but someone else might ...
  // note that setting to 'false' invokes both a memory and communication
  // penalty
  //TODO: not yet implemented
  bool m_delete_packed_fields = true;

  struct PackingData {
    PackingData(std::string s, int n_elts, int order) 
      : field_name(s), num_elts(n_elts), ordering(order) {}
    PackingData() {}
    std::string field_name;
    int num_elts;
    conduit::index_t ordering;
  };

  std::unordered_map<std::string, std::string> m_packing_types;
  std::unordered_map<std::string, size_t> m_packing_num_elts;
  std::unordered_map<std::string, std::vector<conduit::Node*>> m_packing_nodes;
  std::unordered_map<std::string, std::vector<PackingData>> m_packing_data;

  /** Name of nodes in schemas that contain instructions 
   * on normalizing, packing, and casting data, etc.
   */
  const std::string s_metadata_node_name = "metadata";

  const std::string s_coerce_name = "coerce";

  /** constains leaf nodes whose fields are used in the current experiment */
  std::vector<conduit::Node*> m_useme_nodes;

  /** maps: Node's path -> the Node */
  std::unordered_map<std::string, conduit::Node*> m_useme_node_map;

  /** Schema supplied by the user; this contains a listing of the fields
   *  that will be used in an experiment; additionally may contain processing
   *  directives related to type coercion, packing, etc. */
  conduit::Node m_experiment_schema;

  /** Schema supplied by the user; this contains a listing of all fields
   *  of a sample (i.e, as it appears on disk);
   *  may contain additional "metadata" nodes that contain processing
   *  directives, normalization values, etc.
   */
  conduit::Node m_data_schema;

  /** contains a sample's schema, as loaded from disk
   * (identical for all samples) 
   */
  conduit::Schema m_schema_from_dataset;

  /** Maps a node's pathname to the node for m_data_schema */
  std::unordered_map<std::string, conduit::Node*> m_data_map;

  //=========================================================================
  // methods follow
  //=========================================================================

  /** P_0 reads and bcasts the schema */
  void load_sample_schema(conduit::Schema &s);

  /** Fills in various data structures by parsing the m_data_schema and
   *  m_experiment_schema
   */
  void parse_schemas();

  /** get pointers to all nodes in the subtree rooted at the 'starting_node;'
   *  keys are the pathnames; recursive. However, ignores any nodes named s_metadata_node_name
   */
  void get_schema_ptrs(conduit::Node* starting_node, std::unordered_map<std::string, conduit::Node*> &schema_name_map);

  /** Returns, in leaves, the schemas for all leaf nodes in the tree rooted
   *  at 'node_in.' However, ignores any nodes named s_metadata_node_name
   */
  void get_leaves(conduit::Node* node_in, std::vector<conduit::Node*> &leaves_out);

  /** Functionality is similar to get_leaves(); this method differs in that
   *  two conduit::Node hierarchies are searched for leaves. The leaves from 
   *  the first are found, and are then treated as starting points for 
   *  continuing the search in the second hierarchy.
   *
   *  Applicability: a user's schema should be able to specify an "inputs"
   *  node, without having to enumerate all the "inputs" leaves names
   */
  void get_leaves_multi(conduit::Node* node_in, std::vector<conduit::Node*> &leaves_out);

  /** Fills in: m_packed_name_to_num_elts and m_field_name_to_num_elts */
  void do_preload_data_store() override;

  /** loads a sample from file to a conduit::Node;
   *  call normalize, coerce, pack, etc
   */
  void load_sample(conduit::Node &node, size_t index); 

  /** Performs packing, normalization, etc. Called by load_sample. */
  void pack_data(conduit::Node &node_in_out);

  /** Trainer master loads a conduit::Node, then pulls out the Schema 
   *  and bcasts to others
   */
  void load_schema_from_data(conduit::Schema &schema);

  /** loads a use-supplied schema */
  void load_schema(std::string filename, conduit::Node &schema); 


  /** pack the data; this is for all 'groups' in the node */
  void pack(conduit::Node &node);

  /** Merges the contents of the two input nodes, either of which may be
   *  a nullptr; if the input nodes contain a common fieldname, then the
   *  value from node_B are used, and the value from node_A discarded.
   */
  conduit::Node merge_metadata_nodes(const conduit::Node *node_A, const conduit::Node *node_B);

  /** on return, every Node will have a (possibly empty) child node named
   *  <s_metadata_node_name>. The rules: 1) a node inherits the metadata node 
   *  of its parent; 2) if the node already had a metadata child, the contents
   *  are preserved and, where applicable, over-rides the fields of the parent.
   *  Recursive.
   */
  void adjust_metadata(conduit::Node* root);

  void build_packing_map();

  /** repacks from HWC to CHW */
  void repack_image(conduit::Node& node, const std::string &path, const conduit::Node& metadata); 

  /** called from load_sample */
  void coerce(
    const conduit::Node& metadata, 
    hid_t file_handle, 
    const std::string & original_path, 
    const std::string &new_pathname, 
    conduit::Node &node); 

  void normalize(
    conduit::Node& node, 
    const std::string &path, 
    const conduit::Node& metadata);

  // check that all fields that are to be packed in a group have the same
  // data type
  void verify_packing_data_types();

  // constructs m_useme_node_map from m_useme_nodes
  void build_useme_node_map();

  struct {
    bool operator()(const PackingData& a, const PackingData& b) const { 
      return a.ordering < b.ordering; 
    }
  } less_oper;

  //=========================================================================
  // templates follow
  //=========================================================================

  template<typename T_from, typename T_to>
  void coerceme(const T_from* data_in, size_t n_bytes, std::vector<T_to> & data_out); 

  template<typename T>
  void normalizeme(T* data, double scale, double bias, size_t n_bytes); 

  template<typename T>
  void normalizeme(T* data, const double* scale, const double* bias, size_t n_bytes, size_t n_channels); 

  template<typename T>
  void repack_image(T* src_buf, size_t n_bytes, size_t n_rows, size_t n_cols, int n_channels); 

  /** all field assigned to 'group_name' (e.g, 'datum') into a 1D vector */
  template<typename T>
  void pack(std::string group_name, conduit::Node& node);

}; // class hdf5_data_reader

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
void hdf5_data_reader::pack(std::string group_name, conduit::Node& node) {
  std::vector<T> data;
  data.reserve(m_packing_num_elts[group_name]);
  size_t offset = 0;
  for (const auto &nd : m_packing_nodes[group_name]) {
    size_t n_bytes = nd->dtype().number_of_elements() * sizeof(T);
    T* field_data = reinterpret_cast<T*>(nd->data_ptr());
    memcpy(data.data()+offset, field_data, n_bytes);
    offset += n_bytes;
  }
}


} // namespace lbann 

#endif // __REVISED_LBANN_DATA_READER_HDF5_HPP__
