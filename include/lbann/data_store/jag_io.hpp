////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#ifndef __JAG_IO_HPP__
#define __JAG_IO_HPP__

#include "lbann_config.hpp"

#ifdef LBANN_HAS_CONDUIT
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace lbann {

class jag_io {
 public:

 /**
  * NOTE: some methods below take variables "string node_name" while others
  *       take "string key." My convention is that "node_name" indicated
  *       a fully qualified name, i.e, it begins with the sample id:
  *          0/field_1/field_2
  *       on the other hand, "key" does not contain the sample id:
  *          field_1/field_2
  */

/// WARNING! CAUTION! BE ADVISED!
/// cut-n-paste from data_reader_jag_conduit; this is
/// fragile -- probably best to but these in a small
/// file that both reader and store can then include
/// @todo
using ch_t = double; ///< jag output image channel type
using scalar_t = double; ///< jag scalar output type
using input_t = double; ///< jag input parameter type

  //! ctor
  jag_io();

  //! copy ctor
  jag_io(const jag_io&) = default;

  //! operator=
  jag_io& operator=(const jag_io&) = default;

  //@todo: this causes a compile error ... why?
  //jag_io * copy() const { return new jag_io(*this); }

  //! dtor
  ~jag_io();

  /// converts conduit data to our format and saves to disk
  void convert(std::string conduit_pathname, std::string base_dir);

  /// load our format from disk
  void load(std::string base_dir);

  /// returns the set of the child nodes of the parent node
  /// @todo not currently used; may not be needed
  const std::unordered_set<std::string> &get_children(std::string parent) const;

  //const std::vector<jag_io::scalar_t> & get_scalars(size_t sample_id) const; 

  /// Returns size and data type information for the requested node. 
  /// 'total_bytes_out' is num_elts_out * bytes_per_elt_out; 
  void get_metadata(std::string node_name, size_t &num_elts_out, size_t &bytes_per_elt_out, size_t &total_bytes_out, std::string &type_out);

  /// Reads the requested data from file and returns it in 'data_out.'
  /// The caller is responsible for allocating sufficient memory,
  /// i.e, they should previously have called get_metadata(...), then
  /// allocated memory, i.e, std::vector<char> d(total_bytes_out);
  void get_data(std::string node_name, int tid, char * data_out, size_t num_bytes);

  /// returns true if the key exists in the metadata map
  bool has_key(std::string key) const;

  const std::vector<std::string>& get_scalar_choices() const;

  const std::vector<std::string>& get_input_choices() const;

  size_t get_num_samples() const {
    return m_num_samples;
  }

  /// this method is provided for testing and debugging
  size_t get_offset(std::string node_name);

  /// this method is provided for testing and debugging
  const std::vector<std::string> &get_keys() const {
    return m_keys;
  }

  /// this method is provided for testing and debugging
  void print_metadata();

protected :

  struct MetaData {
    MetaData() {}
    MetaData(std::string tp, int elts, int bytes, size_t _offset = 0)
      : dType(tp), num_elts(elts), num_bytes(bytes), offset(_offset) {}

    std::string dType; //float64, int64, etc.
    int         num_elts;  //number of elements in this field
    int         num_bytes; //number of bytes for a single element
    size_t      offset;  //offset wrt m_data: where this resides on disk
  };

  size_t m_num_samples;

  /// used when reading converted data from file; 
  /// each thread gets a separate stream
  std::vector<std::ifstream> m_data_streams;

  /// recursive function invoked by convert();
  /// fills in m_keys and m_parent_to_children
  void get_hierarchy(
      conduit::Node &head,
      std::string parent_name);

  /// maps parent node_named to child node_names
  std::unordered_map<std::string, std::unordered_set<std::string>> m_parent_to_children;

  /// contains the same keys that appear in m_metadata; saving them 
  /// separately so we can iterate through in the order they appeared
  std::vector<std::string> m_keys;

  
  ///@todo this may go away ...
  //std::unordered_map<std::string, std::string> m_data_reader;

  std::unordered_map<std::string, MetaData> m_metadata;

  /// number of bytes required to store each sample on disk in our format
  size_t m_sample_offset;

  //std::vector<std::string> m_scalar_keys;

  std::vector<std::string> m_input_keys;

  /// some conduit keys contain white space, which is annoying to parse,
  /// so internally we convert them to underscores
  void white_space_to_underscore(std::string &s) {
    for (size_t j=0; j<s.size(); j++) {
      if (s[j] == ' ') {
        s[j] = '_';
      }
    }
  }

  /// returns the node_name with the sample_id removed;
  /// also checks that the key exists in m_metadata
  std::string get_metadata_key(std::string node_name) const;

  /// checks that 'key' is a valid key in the m_metadata map;
  /// if not, throws an exception
  void key_exists(std::string key) const;

  /// returns the sample ID
  size_t get_sample_id(std::string node_name) const;
};

}  // namespace lbann

#endif //ifndef LBANN_HAS_CONDUIT ... else

#endif  // __JAG_IO_HPP__
