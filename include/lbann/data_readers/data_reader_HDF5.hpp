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

  /** Returns a raw pointer to the requested data field or group. 
   */
  const void* get_raw_data(const size_t sample_id, const std::string &field_name, size_t &num_bytes) const; 
  /** Returns a raw pointer to the data.  This is intended to eventually
   *  replace fetch_datum(), etc. Once those are gone, the reader no longer
   *  needs to know anyting about CPUMat
   */
  template<typename T>
  void get_data(
    const size_t sample_id_in, 
    std::string field_name_in, 
    size_t &num_elts_out,
    T*& data_out) const; 

  // TODO; should go away in future; implementation is only for
  //       backwards compatibility and testing during development
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;


  // TODO: can go away in future; use get_data(...) instead;
  //       overring for backwards compatibility
  int fetch_responses(CPUMat& Y) override {
    LBANN_ERROR("not implemented for this data reader; please use the form that takes an 'std::string name' as a parameter");
  }

  // TODO: can go away in future; use get_data(...) instead;
  //       overring for backwards compatibility
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

  /** Returns the dimensions of the requested 'field.' Here, 'field'
   *  can be either the name of a field from the original data
   *  (which retains backwards compatibility), or a user-defined field
   *  per the user-supplied schema, e.g, a packed field name.
   *  In the latter case the returned vector will contains a single
   *  entry, which is the linearized size of the combined data fields.
   */
  const std::vector<int> get_data_dims(std::string name="") const override; 
  const std::vector<int> get_data_dims() const override {
    //hack! TODO XX this will break everything!! sooner or later!!!
    return get_data_dims("datum");
    LBANN_ERROR("not implemented for this data reader; please use the form that takes an 'std::string name' as a parameter");
    return std::vector<int>(0);
  }

  int get_linearized_data_size(std::string name="") const override;
  int get_linearized_data_size() const override {
    //hack! TODO XX this will break everything!! sooner or later!!!
    return get_linearized_size("datum");
    LBANN_ERROR("not implemented for this data reader; please use the form that takes an 'std::string name' as a parameter");
    return 0;
  }

  /** Returns information on the dimensions and field names for the
   *  requested field name. Currently not implemented, as it may not be 
   *  needed or used
   */
  void get_packing_data(std::string group_name, std::vector<std::vector<int>> &sizes_out, std::vector<std::string> &field_names_out) const override;

  int get_linearized_response_size(std::string name) const override {
    return get_linearized_data_size(name);
  }
  // required for backwards compatibility
  int get_linearized_response_size() const override {
    LBANN_ERROR("not implemented for this data reader; please use the form that takes an 'std::string name' as a parameter");
    return 0;
  }

  int get_linearized_label_size(std::string name) const override {
    return get_linearized_data_size(name);
  }
  // required for backwards compatibility
  int get_linearized_label_size() const override {
    LBANN_ERROR("not implemented for this data reader; please use the form that takes an 'std::string name' as a parameter");
    return 0;
  }

  int get_num_labels() const override {
    return m_num_labels;
  }

  int get_num_responses() const override {
    return m_num_responses;
  }

private:

  /** num labels and responses can be specified in a metadata node attached to
   *  the root of the experiment schema
   */
  int m_num_labels = 0;
  int m_num_responses = 0;

  // filled in by construct_linearized_size_lookup_tables; used by get_linearized_data_size()
  std::unordered_map<std::string, std::vector<int>> m_data_dims_lookup_table;

  // filled in by construct_linearized_size_lookup_tables; used by get_linearized_data_size()
  std::unordered_map<std::string, int> m_linearized_size_lookup_table;

  // filled in by construct_linearized_size_lookup_tables; used by get_packing_data()
  std::unordered_map<std::string, std::vector<std::string>> m_field_names_lookup_table;

  std::string m_experiment_schema_filename;

  std::string m_data_schema_filename;

  // to set to false, use the cmd line flag: --keep_packed_fields
  // I don't know a use case for keeping an original (possibly coerced)
  // field if it's been packed, but someone else might ...
  // note that setting to 'false' invokes both a memory and communication
  // penalty
  //TODO: not yet implemented
  bool m_delete_packed_fields = true;

  struct PackingGroup {
    std::string group_name;
    conduit::index_t data_type;
    size_t n_elts;
    std::vector<std::string> names;
    std::vector<size_t> sizes;
    std::vector<conduit::index_t> data_types;
  };

  std::unordered_map<std::string, PackingGroup> m_packing_groups;

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
  void get_leaves(conduit::Node* node_in, std::vector<conduit::Node*> &leaves_out, bool ignore_metadata=false);

  /** Functionality is similar to get_leaves(); this method differs in that
   *  two conduit::Node hierarchies are searched for leaves. The leaves from 
   *  the first are found, and are then treated as starting points for 
   *  continuing the search in the second hierarchy.
   *
   *  Applicability: a user's schema should be able to specify an "inputs"
   *  node, without having to enumerate all the "inputs" leaves names
   */
  void get_leaves_multi(
    conduit::Node* node_in, 
    std::vector<conduit::Node*> &leaves_out, 
    bool ignore_metadata=false);

  /** Fills in: m_packed_name_to_num_elts and m_field_name_to_num_elts */
  void do_preload_data_store() override;

  /** loads a sample from file to a conduit::Node;
   *  call normalize, coerce, pack, etc
   */
  void load_sample(conduit::Node &node, size_t index); 

  /** Performs packing, normalization, etc. Called by load_sample. */
  void pack_data(conduit::Node &node_in_out);

  /** loads a schema from file */
  void load_schema(std::string filename, conduit::Node &schema); 

  /** pack the data; this is for all 'groups' in the node */
  void pack(conduit::Node &node, size_t index);

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

  void build_packing_map(conduit::Node &node);

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

  // constructs m_useme_node_map from m_useme_nodes
  void build_useme_node_map();

  void construct_linearized_size_lookup_tables();

  //=========================================================================
  // templates follow
  //=========================================================================

  template<typename T_from, typename T_to>
  void coerceme(const T_from* data_in, size_t n_bytes, std::vector<T_to> & data_out); 

  //normalization for scalars and 1D arrays
  template<typename T>
  void normalizeme(T* data, double scale, double bias, size_t n_bytes); 

  //normalization for images with multiple channels
  template<typename T>
  void normalizeme(T* data, const double* scale, const double* bias, size_t n_bytes, size_t n_channels); 

  template<typename T>
  void repack_image(T* src_buf, size_t n_bytes, size_t n_rows, size_t n_cols, int n_channels); 

  // packs all fields assigned to 'group_name' (e.g, 'datum') into a 1D vector 
  template<typename T>
  void pack(std::string group_name, conduit::Node& node, size_t index);

}; // class hdf5_data_reader

//============================================================================

template<typename T_from, typename T_to>
void hdf5_data_reader::coerceme(const T_from* data_in, size_t n_bytes, std::vector<T_to> & data_out) {
  size_t n_elts = n_bytes / sizeof(T_from);
  data_out.resize(0);
  data_out.reserve(n_elts);
  for (size_t j=0; j<n_elts; j++) {
    data_out.push_back(*data_in++);
  }
}

//normalization for scalars and 1D arrays
template<typename T>
void hdf5_data_reader::normalizeme(T* data, double scale, double bias, size_t n_bytes) {
  size_t n_elts = n_bytes / sizeof(T);
  for (size_t j=0; j<n_elts; j++) {
    data[j] = ( data[j]*scale+bias );
  }
}

//normalization for images with multiple channels
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
    const conduit::Node& leaf = node[ss.str()];
    memcpy(data.data()+idx, leaf.data_ptr(), n_elts*sizeof(T));
    idx += n_elts;
  }
  if (idx != g.n_elts) {
    LBANN_ERROR("idx != g.n_elts*sizeof(T): ", idx, " ", g.n_elts*sizeof(T));
  }
  std::stringstream ss;
  ss << '/' << LBANN_DATA_ID_STR(index) + '/' + group_name;
  node[ss.str()] = data;

  static bool add_to_map = true;
  if (add_to_map) {
    add_to_map = false;
    m_useme_node_map[group_name] = &(node[ss.str()]);
  }
}

template<typename T>
void hdf5_data_reader::get_data(
    const size_t sample_id_in, 
    std::string field_name_in, 
    size_t &num_elts_out,
    T*& data_out) const {
  const conduit::Node& node = m_data_store->get_conduit_node(sample_id_in);
  std::stringstream ss;
  ss << node.name() << node.child(0).name() + "/" << field_name_in;
  if (!node.has_path(ss.str())) {
    LBANN_ERROR("no path: ", ss.str());
  }
  num_elts_out = node[ss.str()].dtype().number_of_elements();
  const std::string& tp = node[ss.str()].dtype().name();
  T w = 42;
  std::string tp2 = conduit::DataType::id_to_name(w);
  if (tp != tp2) {
    LBANN_ERROR("requested type is incorrect; data type is ", tp, " but you requested type ", tp2);
  }
  void *d = const_cast<void*>(node[ss.str()].data_ptr());
  data_out = reinterpret_cast<T*>(d);
}

} // namespace lbann 

#endif // __REVISED_LBANN_DATA_READER_HDF5_HPP__
