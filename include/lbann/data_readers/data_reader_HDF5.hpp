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

  /** Set the pathname to the schema specifying which fields from the data
   * to use.  This schema should be a subset of the complete schema,
   * except that it may contain one additional branch, rooted at the top
   * level, and titled "lbann_metadata."
   *
   * Developer's note: you must call this method, and you must call it
   * before load() is called
   */
  void set_schema_filename(std::string fn) {
    m_useme_schema_filename = fn;
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

  /** Returns a raw pointer to the data. 
   * The data fields that are returned are determined by the "field_names" 
   * vector, which may contain names of leaf and/or internal nodes; for
   * internal nodes, the values returned come from the leaves of the subtree
   * rooted at the named node(s)
   */
//  const unsigned char* get_raw_data(const size_t sample_id, std::vector<std::string> &field_names, size_t &num_bytes) const { /* TODO */ }

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

  /** Returns the set of data fields that the user specified in their
   * supplied schema. Excludes fields than include the "lbann_metadata"
   * substring. This method is primarily implemented for testing purposes.
   */
  const std::unordered_set<std::string> &get_field_names() const {
    return m_useme_pathnames;
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

  /** Contains pointers to nodes in the Schema obtained from data (on disk) */
  std::unordered_map<std::string, conduit::Schema*> m_data_schema_nodes;

  /** Contains pointers to nodes in the Schema supplied by the user
   *  (excludes nodes from the lbann_metadata subtree)
   */
  std::unordered_map<std::string, conduit::Schema*> m_user_schema_nodes;

  /** Contains pointers to nodes in the user supplied Schema for the
   *  (possibly empty) lbann_metadata subtree
   */
  std::unordered_map<std::string, conduit::Schema*> m_metadata_schema_nodes;

  /** Name of a (possibly empty) top-level branch in the useme schema that 
   *  contains instructions on normalizing and packing data, etc.
   */
  const std::string m_metadata_field_name = "lbann_metadata";

  /** The complete Schema (the schema that is constructed and/or read by conduit) */
  conduit::Schema m_data_schema;

  /** The Schema specifying the data fields to use in an experiment;
   *  a subset of the complete schema, with optional additional metadata 
   */
  conduit::Schema m_useme_schema;

  /** A possibly empty subset of m_useme_schema */
  conduit::Schema m_metadata_schema;

  /** Pathnames to schema files */
  std::string m_useme_schema_filename = "";

  /** P_0 reads and bcasts the schema */
  void load_schema(std::string fn, conduit::Schema &schema);
  void load_schema_from_data();

  /** contains all pathnames, from root to leaves, for the
   *  schema that is supplied by the user; excludes branches
   *  that include m_metadata_field_name
   */
  std::unordered_set<std::string> m_useme_pathnames;

  /** contains all pathnames, from root to leaves, for the
   *  schema that is read from the data; excludes branches
   *  that include m_metadata_field_name
   */
  std::unordered_set<std::string> m_data_pathnames;

  /** Next few are for "packing" data. 
   *  'name' would be datum, label, or response
   */
  std::unordered_map<std::string, std::unordered_set<std::string>> m_packed_to_field_names_map;
  std::unordered_map<std::string, std::string> m_field_name_to_packed_map;

  std::unordered_map<std::string, size_t> m_field_name_to_bytes;
  std::unordered_map<std::string, size_t> m_packed_name_to_bytes;

  //=========================================================================
  // private methods follow
  //=========================================================================

  //-------------------------------------------------------------------------
  // --------- START of experimental prototype for parsing metadata ---------
  //-------------------------------------------------------------------------
  
  /** experimental; may decide to do something else. For now, here's
   * a loosely defined language, by example. Be aware that the parser,
   * for now, does no to minimal sanity checking.
   *
   * There should be a single branch beginning at a node lableled
   * m_metadata_field_name.
   *
   * subnodes should be labled per the following:
   *
   *   fetch - data
   *         - label
   *         - response
   *
   *   pack  - data
   *         - label
   *         - response
   *
   *   cast - field_name_j - data_type
   *
   *   normalize - field_name_j
   *
   * Comments: any pathname that contains the '#' character
   *           any place in the string is considered a comment
   */
  void parse_metadata();

  /** Fills in: m_packed_name_to_bytes and m_field_name_to_bytes */
  void tabulate_packing_memory_requirements();

  /** On return, schema_name_map contains the names of all nodes in the schema.
   *  WARNING: assumes each node's name is unique (name, not full path);
   *           this should be relooked and relaxed in the future
   */
  void get_schema_ptrs(conduit::Schema* schema, std::unordered_map<std::string, conduit::Schema*> &schema_name_map);

  /** Returns, in leaves, the schemas for all leaf noodes in the tree 
   *  rooted at 'schema_in'
   *  WARNING: assumes each node's name is unique (name, not full path);
   *           this should be relooked and relaxed in the future
   */
  void get_leaves(const conduit::Schema* schema_in, std::vector<const conduit::Schema*> &leaves);

  std::vector<conduit::Schema*> get_children(conduit::Schema *schema);
  std::vector<conduit::Schema*> get_grand_children(conduit::Schema *schema);

  //-------------------------------------------------------------------------
  // ---------- END of experimental prototype for parsing metadata ----------
  // ------------------------------------------------------------------------

  void do_preload_data_store() override;

  /** loads a sample from file to a conduit::Node
   */
  void load_sample(conduit::Node &node, size_t index); 

  /** Performs packing, normalization, etc. Called by load_sample. */
  void munge_data(conduit::Node &node_in_out);

  /** Verify the useme schema is a subset of the complete schema */
  void validate_useme_schema();

  /** Returns, in "output," the pathnames to all leaves in the schema */
  void get_datum_pathnames(
    const conduit::Schema &schema, 
    std::unordered_set<std::string> &output,
    int n = 0,
    std::string path = "");
};

} // namespace lbann 

#endif // __REVISED_LBANN_DATA_READER_HDF5_HPP__
