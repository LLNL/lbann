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
 *
 * Assumptions:
 *   1. All field names are unique. A "file name" is the final
 *      '/'-deliniated name in a pathname
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

private:

///  std::unordered_map<std::string, std::unordered_map<std::string, lbann::DataType>> m_normailzation;

  /** Contains pointers to Nodes that contain the complete schemas
   *  for the data on disk, and additionally contain normalization data.
   */
  std::unordered_map<std::string, conduit::Node*> m_data_schema_nodes;

  /** Contains pointers to nodes that specify which data fields are
   * to be used for the current experiment
   */
  std::unordered_map<std::string, conduit::Node*> m_experiment_schema_nodes;

  /** Name of a (possibly empty) top-level branch in the useme schema that 
   *  contains instructions on normalizing and packing data, etc.
   */
  const std::string m_metadata_field_name = "metadata";

  /** Schema supplied by the user; this should contain the smae information
   *  (sans data) as is on disk, and may additionally contain normalization
   *  data, etc; supplied by the user
   */
  conduit::Node m_data_schema;

  /** The Schema specifying the data fields to use in an experiment */
  conduit::Node m_experiment_schema;

  /** P_0 reads and bcasts the schema */
  void load_schema(std::string fn, conduit::Node &schema);


  /** Next few are used for "packing" data. 
   *  'name' would be datum, label, response, or other (the user can choose
   *  any names they like; I'm using datum, etc for backwards compatibility)
   */
  std::unordered_map<std::string, std::vector<std::string>> m_packed_to_field_names_map;
  std::unordered_map<std::string, std::string> m_field_name_to_packed_map;

  std::unordered_map<std::string, size_t> m_field_name_to_num_elts;
  std::unordered_map<std::string, size_t> m_packed_name_to_num_elts;


  std::unordered_set<const conduit::Node*> m_all_exp_leaves;

  /** Fills in various data structures (m_packed_to_field_names_map, 
   *  m_packed_name_to_num_elts, etc) using data from the schema for
   *  the actual data from disk, and the metadata schemas supplied by
   *  the user
   */
  void parse_schemas();

  /** Fills in: m_packed_name_to_num_elts and m_field_name_to_num_elts */
  void tabulate_packing_memory_requirements();

  /** get pointers to all nodes in the subtree rooted at the 'starting_node;'
   *  keys are the pathnames; recursive.
   */
  void get_schema_ptrs(conduit::Node* starting_node, std::unordered_map<std::string, conduit::Node*> &schema_name_map);

  /** Returns, in leaves, the schemas for all leaf noodes in the tree 
   *  rooted at 'schema_in'
   */
  void get_leaves(const conduit::Node* node_in, std::vector<const conduit::Node*> &leaves, std::string ignore_child_branch="metadata");

  void do_preload_data_store() override;

  /** loads a sample from file to a conduit::Node
   */
  void load_sample(conduit::Node &node, size_t index); 

  /** Performs packing, normalization, etc. Called by load_sample. */
  void munge_data(conduit::Node &node_in_out);

  /** Loads a conduit::Node, then pulls out the Schema */
  void load_schema_from_data(conduit::Schema &schema);
};

} // namespace lbann 

#endif // __REVISED_LBANN_DATA_READER_HDF5_HPP__
