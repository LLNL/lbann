////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_DATA_READER_HDF5_REVISED_HPP
#define LBANN_DATA_READER_HDF5_REVISED_HPP

#include "lbann/data_ingestion/readers/data_reader.hpp"
#include "lbann/data_ingestion/readers/data_reader_sample_list.hpp"
#include "lbann/data_ingestion/readers/sample_list_hdf5.hpp"
#include "lbann/data_store/data_store_conduit.hpp"

#include <set>

// Forward declaration
class DataReaderHDF5WhiteboxTester;

/** Valid keys in a metadata file */
#define HDF5_METADATA_KEY_DIMS "dims"
#define HDF5_METADATA_KEY_CHANNELS "channels"
#define HDF5_METADATA_KEY_ORDERING "ordering"
#define HDF5_METADATA_KEY_SCALE "scale"
#define HDF5_METADATA_KEY_BIAS "bias"
#define HDF5_METADATA_KEY_LAYOUT "layout"
#define HDF5_METADATA_KEY_TRANSPOSE "transpose"
#define HDF5_METADATA_KEY_COERCE "coerce"
#define HDF5_METADATA_KEY_PACK "pack"
/** Valid string values for a metadata file */
#define HDF5_METADATA_VALUE_COERCE_FLOAT "float"
#define HDF5_METADATA_VALUE_COERCE_DOUBLE "double"
#define HDF5_METADATA_VALUE_COERCE_FLOAT64 "float64"
#define HDF5_METADATA_VALUE_COERCE_FLOAT16 "float16"
#define HDF5_METADATA_VALUE_LAYOUT_CHW "chw"
#define HDF5_METADATA_VALUE_LAYOUT_HWC "hwc"
#define HDF5_METADATA_VALUE_LAYOUT_CDHW "cdhw"
#define HDF5_METADATA_VALUE_LAYOUT_DHWC "dhwc"

namespace lbann {

bool is_hdf5_metadata_key_valid(std::string const& key);
bool is_hdf5_field_channels_last(conduit::Node const& field);
bool does_hdf5_field_require_repack_to_channels_first(
  conduit::Node const& metadata);
/**
 * For some reason conduit includes quotes around the string,
 * even when they're not in the json file -- so need to strip them off
 */
std::string conduit_to_string(conduit::Node const& field);

static std::set<std::string> const hdf5_metadata_valid_keys = {
  HDF5_METADATA_KEY_DIMS,
  HDF5_METADATA_KEY_CHANNELS,
  HDF5_METADATA_KEY_ORDERING,
  HDF5_METADATA_KEY_SCALE,
  HDF5_METADATA_KEY_BIAS,
  HDF5_METADATA_KEY_LAYOUT,
  HDF5_METADATA_KEY_TRANSPOSE,
  HDF5_METADATA_KEY_COERCE,
  HDF5_METADATA_KEY_PACK,
};

/**
 * A generalized data reader for data stored in HDF5 files.
 */
class hdf5_data_reader
  : public data_reader_sample_list<sample_list_hdf5<std::string>>
{
public:
  hdf5_data_reader(bool shuffle = true);
  hdf5_data_reader(const hdf5_data_reader&);
  hdf5_data_reader& operator=(const hdf5_data_reader&);
  hdf5_data_reader* copy() const override
  {
    return new hdf5_data_reader(*this);
  }
  void copy_members(const hdf5_data_reader& rhs);
  ~hdf5_data_reader() override;

  bool has_conduit_output() override { return true; }

  std::string get_type() const override { return "hdf5_data_reader"; }

  /** @brief Prints metadata and data-types for all field-names
   *
   *  Note: if you change the "os" parameter to other than cout, some
   *  information will be lost; this is because conduit print() methods
   *  do not take parameters; they only print to cout.
   *  Note: this method is called internally (I forget from exactly where),
   *  and can be disabled by the cmd line switch: --quiet
   */
  void print_metadata(std::ostream& os = std::cout);

  void load() override;

  bool fetch_conduit_node(conduit::Node& sample, uint64_t data_id) override;

  /** @brief Sets the name of the yaml experiment file */
  void set_experiment_schema_filename(std::string fn)
  {
    m_experiment_schema_filename = fn;
  }

  /** @brief Returns the name of the yaml experiment file */
  const std::string& get_experiment_schema_filename()
  {
    return m_experiment_schema_filename;
  }

  /** @brief Sets the name of the yaml data file */
  void set_data_schema_filename(std::string fn) { m_data_schema_filename = fn; }

  /** @brief Returns the name of the yaml data file */
  const std::string& get_data_schema_filename()
  {
    return m_data_schema_filename;
  }

  const std::vector<El::Int> get_data_dims() const override
  {
    return get_data_dims(INPUT_DATA_TYPE_SAMPLES);
  }

  int get_linearized_data_size() const override
  {
    return get_linearized_size(INPUT_DATA_TYPE_SAMPLES);
  }

  int get_linearized_response_size() const override
  {
    return get_linearized_size(INPUT_DATA_TYPE_RESPONSES);
  }

  int get_linearized_label_size() const override
  {
    return get_linearized_size(INPUT_DATA_TYPE_LABELS);
  }

  int get_num_labels() const override { return get_linearized_label_size(); }

  int get_num_responses() const override
  {
    return get_linearized_response_size();
  }

  /** @brief this method is made public for testing */
  conduit::Node get_experiment_schema() const { return m_experiment_schema; }
  /** @brief this method is made public for testing */
  conduit::Node get_data_schema() const { return m_data_schema; }
  /** @brief this method is made public for testing */
  void set_experiment_schema(const conduit::Node& s);
  /** @brief this method is made public for testing */
  void set_data_schema(const conduit::Node& s);
  /** @brief this method is made public for testing */
  std::unordered_map<std::string, conduit::Node> get_node_map() const
  {
    return m_useme_node_map;
  }

  /** @brief this method is made public for testing
   *
   *  On return, every Node will have a (possibly empty) child node named
   *  \<s_metadata_node_name\>. The rules: 1) a node inherits the metadata node
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
  std::unordered_map<std::string, std::vector<El::Int>>
    m_data_dims_lookup_table;

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
  bool m_delete_packed_fields = true;

  struct PackingGroup
  {
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

  /** Refers to data that will be used for the experiment.
   *  Combination of hte data & experimental schema.
   *
   *  DAH (7/20/21) m_useme_node_map_ptr should be completely replaced
   *  by m_useme_node_map (background: m_useme_node_map_ptrs was what
   *  I 1st coded, then realized that you shouldn't copy pointers in
   *  copy_members)
   *
   *  BVE @todo - cleanup node map pointers  */
  /** maps: Node's path -> the Node */
  std::unordered_map<std::string, conduit::Node*> m_useme_node_map_ptrs;
  /** maps: Node's path -> the Node */
  std::unordered_map<std::string, conduit::Node> m_useme_node_map;

  /** Schema supplied by the user; this contains a listing of the fields
   *  that will be used in an experiment; additionally may contain processing
   *  directives related to type coercion, packing, etc.
   *  Takes precidences over m_data_schema and inherits from m_data_schema */
  conduit::Node m_experiment_schema;

  /** Schema supplied by the user; this contains a listing of all fields
   *  of a sample (i.e, as it appears on disk);
   *  may contain additional "metadata" nodes that contain processing
   *  directives, normalization values, etc.
   */
  conduit::Node m_data_schema;

  /** Used internally in the construction of the other node maps - refers
   *  to nodes that don't contain data */
  /** Maps a node's pathname to the node for m_data_schema */
  std::unordered_map<std::string, conduit::Node*> m_data_map;

  /** only used in pack() **/
  std::unordered_set<std::string> m_add_to_map;

  //=========================================================================
  // methods follow
  //=========================================================================

  const std::vector<El::Int> get_data_dims(std::string name = "") const;

  /** Returns the size of the requested field (datum, label, response, etc) */
  int get_linearized_size(data_field_type const& data_field) const override;

  /** P_0 reads and bcasts the schema */
  void load_sample_schema(conduit::Schema& s);

  /** Fills in various data structures by parsing the schemas
   *  (i.e, m_data_schema and m_experiment_schema
   */
  void parse_schemas();

  /** get pointers to all nodes in the subtree rooted at the 'starting_node;'
   *  keys are the pathnames; recursive. However, ignores any nodes named
   *  "metadata" (or whatever 's_metadata_node_name' is set to).
   */
  void get_schema_ptrs(
    conduit::Node* starting_node,
    std::unordered_map<std::string, conduit::Node*>& schema_name_map);

  /**  Returns, in leaves, the schemas for leaf nodes in the tree rooted
   *  at 'node_in.'  Optionally ignores nodes named "metadata" (or whatever
   *  's_metadata_node_name' is set to).
   *  Keys in the filled-in map are the pathnames to the leaf nodes.
   */
  void get_leaves(conduit::Node* node_in,
                  std::unordered_map<std::string, conduit::Node*>& leaves_out);

  /** Functionality is similar to get_leaves(). This method differs in that
   *  two conduit::Node trees are searched for leaves. The leaves from
   *  the first are found, and are then treated as starting points for
   *  continuing the search in the second tree. In practice, the first tree
   *  is defined by the experiment_schema, and the second by the data_schema.
   */
  void
  get_leaves_multi(conduit::Node* node_in,
                   std::unordered_map<std::string, conduit::Node*>& leaves_out);

  void do_preload_data_store() override;

  /** Loads a sample from file to a conduit::Node; call normalize,
   *  coerce, pack, etc. "ignore_failure" is only used for
   *  by the call to print_metadata().
   */
  void load_sample(conduit::Node& node,
                   hid_t file_handle,
                   const std::string& sample_name,
                   bool ignore_failure = false);

  /** Finds a sample in the sample list by index and then loads it.
   */
  void load_sample_from_sample_list(conduit::Node& node,
                                    size_t index,
                                    bool ignore_failure = false);

  /** Performs packing, normalization, etc. Called by load_sample. */
  void pack_data(conduit::Node& node_in_out);

  /** loads a schema from file */
  void load_schema(std::string filename, conduit::Node& schema);

  /** pack the data; this is for all 'groups' in the node */
  void pack(conduit::Node& node, size_t index);

  /** Merges the contents of the two input nodes, either of which may be
   *  a nullptr. If the input nodes contain a common field-name, then the
   *  value from node_B are used, and the value from node_A discarded.
   */
  conduit::Node merge_metadata_nodes(const conduit::Node* node_A,
                                     const conduit::Node* node_B);

  /** Fills in m_packing_groups data structure */
  void build_packing_map(conduit::Node& node);

  /** repacks from HWC to CHW */
  void repack_image(conduit::Node& node,
                    const std::string& path,
                    const conduit::Node& metadata);

  /** called from load_sample */
  void coerce(const conduit::Node& metadata,
              hid_t file_handle,
              const std::string& original_path,
              const std::string& new_pathname,
              conduit::Node& node);

  void normalize(conduit::Node& node,
                 const std::string& path,
                 const conduit::Node& metadata);

  /** Constructs m_data_dims_lookup_table and m_linearized_size_lookup_table */
  void construct_linearized_size_lookup_tables();
  void construct_linearized_size_lookup_tables(conduit::Node& node);

  /** sanity check; call after adjust_metadata */
  void test_that_all_nodes_contain_metadata(conduit::Node& node);

  bool get_delete_packed_fields() { return m_delete_packed_fields; }
  void set_delete_packed_fields(bool flag) { m_delete_packed_fields = flag; }

  //=========================================================================
  // template declarations follow
  //=========================================================================

  /** Packs all fields assigned to 'group_name' (datum, label, response)
   *  into a 1D vector; the packed field is then inserted in a conduit
   *  node, that is passed to the data_store
   */
  template <typename T>
  void pack(std::string const& group_name, conduit::Node& node, size_t index);

  /** Returns true if this is a node that was constructed from one or more
   * original data fields
   */
  bool is_composite_node(const conduit::Node& node) const;

  // Designate a whitebox testing friend
  friend class ::DataReaderHDF5WhiteboxTester;

}; // END: class hdf5_data_reader

} // namespace lbann

#endif // LBANN_DATA_READER_HDF5_REVISED_HPP
