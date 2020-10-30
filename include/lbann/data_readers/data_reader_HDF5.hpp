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
#include "conduit/conduit.hpp"

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

  /** sets the level at which the passed useme schema begins wrt the
   * complete schema. The intention here is to discard top level(s) 
   * that contain unique sample IDs. (Default is 2 for JAG data)
   * (this will need to be re-thought)
   */
  void set_schema_starting_level(int level) { m_schema_starting_level = level; }

private:

  /** the level at which the passed useme schema begins wrt the complete schema */
  int m_schema_starting_level = 2;

  /** Name of a special branch in the schemas that may exist, in addition
   * to metada for the actual data; if present, contains information for
   * parsing the data.
   */
  const std::string m_metadata_field_name = "lbann_metadata";

  /** The complete Schema (the schema that is constructed and/or read by conduit) */
  conduit::Schema m_data_schema;

  /** The Schema specifying the data fields to use in an experiment;
   * a subset of the complete schema, with optional additional metadata 
   */
  conduit::Schema m_useme_schema;

  /** Pathnames to schema files */
  std::string m_useme_schema_filename = "";

  /** P_0 reads and bcasts the schema */
  void load_useme_schema();

  std::unordered_set<std::string> m_useme_pathnames;
  std::unordered_set<std::string> m_data_pathnames;

  //=========================================================================
  // private methods follow
  //=========================================================================

  /** Verify the useme schema is a subset of the complete schema */
  void validate_useme_schema();

  /** Returns, in "output," the pathnames to all leaves in the schema */
  void get_datum_pathnames(
    const conduit::Schema &schema, 
    std::unordered_set<std::string> &output,
    int n = 0,
    std::string path = "");

#if 0
  int get_linearized_data_size() const override {
    return m_num_features;
  }
  int get_linearized_label_size() const override {
    if(!m_has_labels) {
      return generic_data_reader::get_linearized_label_size();
    }
    // This data reader currently assumes that the shape of the label
    // tensor is the same to the data tensor.
    return m_num_features;
  }
  int get_linearized_response_size() const override {
    if(!m_has_responses) {
      return generic_data_reader::get_linearized_response_size();
    }
    return m_all_responses.size();
  }
  const std::vector<int> get_data_dims() const override {
    return m_data_dims;
  }
#endif
};

}
#endif // __REVISED_LBANN_DATA_READER_HDF5_HPP__
