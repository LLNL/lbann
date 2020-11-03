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

  int get_linearized_size(const std::string &key) const override;

  int fetch_data(CPUMat& X, El::Matrix<El::Int>& indices_fetched) override;

  int fetch_responses(CPUMat& Y) override {
    LBANN_ERROR("fetch_response() is not implemented");
  }

  int fetch_labels(CPUMat& Y) override {
    LBANN_ERROR("fetch_labels() is not implemented");
  }

private:

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
  void load_schema(std::string fn, conduit::Schema &schema);
  void load_schema_from_data();

  std::unordered_set<std::string> m_useme_pathnames;
  std::unordered_set<std::string> m_data_pathnames;

  //=========================================================================
  // private methods follow
  //=========================================================================

  void do_preload_data_store() override;


  void load_sample(conduit::Node &node, size_t index); 

  // may go away soon!
  void preload_helper(
    const hid_t& h, 
    const std::string &sample_name, 
    const std::string &field_name, 
    int data_id, conduit::Node &node);

  /** Verify the useme schema is a subset of the complete schema */
  void validate_useme_schema();

  /** Returns, in "output," the pathnames to all leaves in the schema */
  void get_datum_pathnames(
    const conduit::Schema &schema, 
    std::unordered_set<std::string> &output,
    int n = 0,
    std::string path = "");
};

}
#endif // __REVISED_LBANN_DATA_READER_HDF5_HPP__
