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

  // should go away in future?
  int fetch_data(CPUMat& X, El::Matrix<El::Int>& indices_fetched) override;

  // should go away in future?
  int fetch_responses(CPUMat& Y) override {
    LBANN_ERROR("fetch_response() is not implemented");
  }

  // TODO, perhaps
  // should go away in future?
  int fetch_labels(CPUMat& Y) override {
    LBANN_ERROR("fetch_labels() is not implemented");
  }

  /** returns true if the field_name is a float or float* */
  bool is_float(const std::string &field_name) const {
    conduit::DataType dtype = m_data_schema[field_name].dtype();
    return dtype.is_float();
  }  

  /** returns true if the field_name is a double or double* */
  bool is_double(const std::string &field_name) const {
    conduit::DataType dtype = m_data_schema[field_name].dtype();
    return dtype.is_double();
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

  /** Name of nodes in schemas that contain instructions 
   * on normalizing, packing, and casting data, etc.
   */
  const std::string s_metadata_node_name = "metadata";

  /** Leaf nodes whose fields are to be used in the current experiment */
  std::vector<conduit::Node*> m_useme_nodes;

  /** Schema supplied by the user; this contains a listing of the fields
   *  that will be used in an experiment; additionally may contain processing
   *  directives related to type coercion, packing, etc. */
  conduit::Node m_experiment_schema;

  /** Schema specifying the data set as it resides, e.g, on disk.
   *  May contain additional "metadata" nodes that contain processing
   *  directives, normalization values, etc.
   */
  conduit::Node m_data_schema;

  /** Maps a node's pathname to the node for m_data_schema */
  std::unordered_map<std::string, conduit::Node*> m_data_map;

  /** Maps a node's pathname to the node for m_experiment_schema */
//XX  std::unordered_map<std::string, conduit::Node*> m_experiment_map;

  //=========================================================================
  // methods follow
  //=========================================================================

  /** Fills in various data structures by parsing the m_data_schema and
   *  m_experiment_schema
   */
  void parse_schemas();

  /** P_0 reads and bcasts the schema */
  void load_schema(std::string fn, conduit::Node &schema);

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
   *  Applicability: a user's schema should be able to specify "inputs,"
   *  without specifying all the "inputs" leaf names
   */
  void get_leaves_multi(conduit::Node* node_in, std::vector<conduit::Node*> &leaves_out);

#if 0
  /** Next few are used for "packing" data. 
   *  'name' would be datum, label, response, or other (the user can choose
   *  any names they like; I'm using datum, etc for backwards compatibility)
   */
  std::unordered_map<std::string, std::vector<std::string>> m_packed_to_field_names_map;
  std::unordered_map<std::string, std::string> m_field_name_to_packed_map;

  std::unordered_map<std::string, size_t> m_field_name_to_num_elts;
  std::unordered_map<std::string, size_t> m_packed_name_to_num_elts;
#endif


#if 0
TODO
  /** next few are for caching normalization values, so we don't have 
   * to parse them from the conduit::Nodes each time load_sample is called
   */
  std:unordered_map<std::string, double> m_scale;
  std:unordered_map<std::string, double> m_scale;
#endif


  /** Fills in: m_packed_name_to_num_elts and m_field_name_to_num_elts */
  void tabulate_packing_memory_requirements();



  void do_preload_data_store() override;

  /** loads a sample from file to a conduit::Node
   */
  void load_sample(conduit::Node &node, size_t index); 

  /** Performs packing, normalization, etc. Called by load_sample. */
  void pack_data(conduit::Node &node_in_out);

  /** Loads a conduit::Node, then pulls out the Schema */
  void load_schema_from_data(conduit::Schema &schema);

  /** run transform pipelines */
  void transform(conduit::Node& node); 

  /** pack the data */
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

  void coerce();

//  const std::string strip_sample_id(const std::string &s);
};

} // namespace lbann 

#endif // __REVISED_LBANN_DATA_READER_HDF5_HPP__
