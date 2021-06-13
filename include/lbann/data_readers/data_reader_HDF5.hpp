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
  ~hdf5_data_reader() override;

  std::string get_type() const override {
    return "hdf5_data_reader";
  }

  /** @brief Prints metadata and data-types for all field-names 
   *
   *  Note: if you change the "os" parameter to other than cout, some 
   *  information will be lost; this is because conduit print() methods
   *  do not take parameters; they only print to cout.
   *  Note: this method is called internally (I forget from exactly where),
   *  and can be disabled by the cmd line switch: --quiet
   */
  void print_metadata(std::ostream& os=std::cout);

  void load() override;

  /** @brief Called by fetch_data, fetch_label, fetch_response 
   *
   * Note that 'which' is not confined to the three commonly used
   * in lbann (datum, label, response); in general, it can be
   * any pack field in the experiment schema: pack: <string>
   */
  bool fetch(std::string which, CPUMat& Y, int data_id, int mb_idx);


  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override {
    return fetch("datum", X, data_id, mb_idx);
  }

  bool fetch_response(CPUMat& Y, int data_id, int mb_idx) override {
    return fetch("response", Y, data_id, mb_idx);
  }

  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override {
    return fetch("label", Y, data_id, mb_idx);
  }  

  /** @brief Sets the name of the yaml experiment file */
  void set_experiment_schema_filename(std::string fn) {
    m_experiment_schema_filename = fn;
  }

  /** @brief Returns the name of the yaml experiment file */
  const std::string& get_experiment_schema_filename() {
    return m_experiment_schema_filename;
  }

  /** @brief Sets the name of the yaml data file */
  void set_data_schema_filename(std::string fn) {
    m_data_schema_filename = fn;
  }

  /** @brief Returns the name of the yaml data file */
  const std::string& get_data_schema_filename() {
    return m_data_schema_filename;
  }

  const std::vector<int> get_data_dims() const override {
    return get_data_dims("datum");
  }

  int get_linearized_data_size() const override {
    return get_linearized_size("datum");
  }

  int get_linearized_response_size() const override {
    return get_linearized_size("response"); 
  }

  int get_linearized_label_size() const override {
    return get_linearized_size("label");
  }

  int get_num_labels() const override {
    return get_linearized_label_size();
  }

  int get_num_responses() const override {
    return get_linearized_response_size();
  }

  /** @brief this method is made public for testing */
  conduit::Node get_experiment_schema() const { return m_experiment_schema; }
  /** @brief this method is made public for testing */
  conduit::Node get_data_schema() const{ return m_data_schema; }
  /** @brief this method is made public for testing */
  void set_experiment_schema(const conduit::Node& s);
  /** @brief this method is made public for testing */
  void set_data_schema(const conduit::Node& s);
  /** @brief this method is made public for testing */
  std::unordered_map<std::string, conduit::Node> get_node_map() const { 
    return  m_useme_node_map; 
  }

  /** @brief this method is made public for testing
   *
   *  On return, every Node will have a (possibly empty) child node named
   *  <s_metadata_node_name>. The rules: 1) a node inherits the metadata node 
   *  of its parent; 2) if the node already has a metadata child, the contents
   *  are preserved; if both parent and child have the same named field,
   *  the child's takes precedence.
   *  Recursive.
   */
  void adjust_metadata(conduit::Node* root);

private:

  /** filled in by construct_linearized_size_lookup_tables; 
   *  used by get_data_dims()
   */
  std::unordered_map<std::string, std::vector<int>> m_data_dims_lookup_table;

  /** filled in by construct_linearized_size_lookup_tables; 
   *  used by get_linearized_size()
   */
  std::unordered_map<std::string, int> m_linearized_size_lookup_table;

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

  const std::string s_composite_node = "composite_node";

  const std::string s_coerce_name = "coerce";

  /** maps: Node's path -> the Node */
  std::unordered_map<std::string, conduit::Node*> m_useme_node_map_ptrs;
  /** maps: Node's path -> the Node */
  std::unordered_map<std::string, conduit::Node> m_useme_node_map;

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

  /** only used in pack() **/
  std::unordered_set<std::string> m_add_to_map;

  //=========================================================================
  // methods follow
  //=========================================================================

  /** Returns a pointer to the requested data field.
   *
   *  The caller must cast to the appropriate type, as specified by 'dtype_out,'
   *  which is one of: float32, float64, int32, int64, uint64, uint32
   */
  const void* get_data(
    const size_t sample_id_in, 
    std::string field_name_in, 
    size_t &num_elts_out, 
    std::string& dtype_out) const;

  const std::vector<int> get_data_dims(std::string name="") const;

  /** Returns the size of the requested field (datum, label, response, etc) */
  int get_linearized_size(std::string name) const;

  /** P_0 reads and bcasts the schema */
  void load_sample_schema(conduit::Schema &s);

  /** Fills in various data structures by parsing the schemas 
   *  (i.e, km_data_schema and m_experiment_schema
   */
  void parse_schemas();

  /** get pointers to all nodes in the subtree rooted at the 'starting_node;'
   *  keys are the pathnames; recursive. However, ignores any nodes named 
   *  "metadata" (or whatever 's_metadata_node_name' is set to). 
   */
  void get_schema_ptrs(conduit::Node* starting_node, std::unordered_map<std::string, conduit::Node*> &schema_name_map);

  /**  Returns, in leaves, the schemas for leaf nodes in the tree rooted
   *  at 'node_in.'  Optionally ignores nodes named "metadata" (or whatever
   *  's_metadata_node_name' is set to).
   *  Keys in the filled-in map are the pathnames to the leaf nodes.
   */
  void get_leaves(conduit::Node* node_in, std::unordered_map<std::string, conduit::Node*> &leaves_out);

  /** Functionality is similar to get_leaves(). This method differs in that
   *  two conduit::Node trees are searched for leaves. The leaves from 
   *  the first are found, and are then treated as starting points for 
   *  continuing the search in the second tree. In practice, the first tree
   *  is defined by the experiment_schema, and the second by the data_schema.
   */
  void get_leaves_multi(
    conduit::Node* node_in, 
    std::unordered_map<std::string, conduit::Node*> &leaves_out);

  void do_preload_data_store() override;

  /** Loads a sample from file to a conduit::Node; call normalize,
   *  coerce, pack, etc. "ignore_failure" is only used for
   *  by the call to print_metadata().
   */
  void load_sample(conduit::Node &node, size_t index, bool ignore_failure = false); 

  /** Performs packing, normalization, etc. Called by load_sample. */
  void pack_data(conduit::Node &node_in_out);

  /** loads a schema from file */
  void load_schema(std::string filename, conduit::Node &schema); 

  /** pack the data; this is for all 'groups' in the node */
  void pack(conduit::Node &node, size_t index);

  /** Merges the contents of the two input nodes, either of which may be
   *  a nullptr. If the input nodes contain a common field-name, then the
   *  value from node_B are used, and the value from node_A discarded.
   */
  conduit::Node merge_metadata_nodes(const conduit::Node *node_A, const conduit::Node *node_B);

  /** Fills in m_packing_groups data structure */
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

  /** Constructs m_data_dims_lookup_table and m_linearized_size_lookup_table */
  void construct_linearized_size_lookup_tables();

  /** sanity check; call after adjust_metadata */
  void test_that_all_nodes_contain_metadata(conduit::Node& node); 

  //=========================================================================
  // template declarations follow
  //=========================================================================

  template<typename T_from, typename T_to>
  void coerceme(const T_from* data_in, size_t n_bytes, std::vector<T_to> & data_out); 

  /** Performs normalization for scalars and 1D arrays */
  template<typename T>
  void normalizeme(T* data, double scale, double bias, size_t n_bytes); 

  /** Performs normalization for images with multiple channels */
  template<typename T>
  void normalizeme(T* data, const double* scale, const double* bias, size_t n_bytes, size_t n_channels); 

  template<typename T>
  void repack_image(T* src_buf, size_t n_bytes, size_t n_rows, size_t n_cols, int n_channels); 

  /** Packs all fields assigned to 'group_name' (datum, label, response) 
   *  into a 1D vector; the packed field is then inserted in a conduit
   *  node, that is passed to the data_store
   */
  template<typename T>
  void pack(std::string group_name, conduit::Node& node, size_t index);

  /** Returns true if this is a node that was constructed from one or more
   * original data fields
   */
  bool is_composite_node(const conduit::Node& node);
  bool is_composite_node(const conduit::Node* node) {
    return is_composite_node(*node);
  }

}; // END: class hdf5_data_reader

//=========================================================================
// templates definitions follow (from here to end of file)
//=========================================================================

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

#endif // __REVISED_LBANN_DATA_READER_HDF5_HPP__
