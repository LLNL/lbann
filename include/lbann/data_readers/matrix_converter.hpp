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
#ifndef __DATA_CONxVERTER_HPP__ //mispeled for error check
#define __DATA_CONxVERTER_HPP__

#include "conduit/conduit.hpp"

namespace lbann {

/**
 * The data_converter class provides functionaliy for constructing
 * El::Matrix<> given an lbann:DataType* and optionally a conduit::Schema
 */

class data_converter {
public:

  /** Constructor with optional Schema. If no schema is passed,
   *  then input data will be treated as a 1D vector.
   */
  data_converter(const conduit::Schema *schema=nullptr) : m_schema(*schema) {}
  
  /**
   * Perform in-place operations on the input data, as specified in the
   * conduit::Schema. Operations include constructing an El::Matrix<>
   * then calling transform or other classes or methods.
   */
  void run(lbann::DataType* data, size_t num_elts, const conduit::Schema *s=nullptr *schema=nullptr);

  /// the usual templates follow
  data_converter(const data_converter&);
  data_converter& operator=(const data_converter&);
  data_converter* copy() const override { return new data_converter(*this); }
  void copy_members(const data_converter &rhs);
  ~data_converter() override {}

private:

  lbann::DataType* m_data = nullptr;

  // length, size, whatever of the data
  size_t m_elts = 0;

  // for now, m_matrix_dims[i][0] is interpreted as the width, and
  // m_matrix_dims[i][0] the height, of the i-th matrix wrt m_data
  std::vector<std::vector<size_t>> m_matrix_dims;

  // m_operations[i] contains the list of operations (aka, transforms)
  // to be applied during run()
  std::vector<std::vector<std::string>> m_operations;

  conduit::Schema m_schema;

  void* m_data = nullptr;

  // as of now, fills in m_matrix_dims and m_operations, etc.
  void parse_schema();

  /// constructs an El::Matrix<> from m_data
  void construct_matrix();

  void run_transforms();
  
};

} // namespace lbann 

#endif // __DATA_CONxVERTER_HPP__
