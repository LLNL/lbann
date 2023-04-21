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

#include "Catch2BasicSupport.hpp"

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"

#include "./data_reader_common_HDF5_test_utils.hpp"

void check_node_fields(conduit::Node const& ref_node,
                       conduit::Node const& test_node,
                       conduit::Node const& data_schema,
                       std::vector<std::string> fields,
                       const std::string original_path,
                       const std::string new_pathname)
{
  for (auto f : fields) {
    const std::string ref_pathname(original_path + "/" + f);
    const std::string test_pathname(new_pathname + "/" + f);
    // Select the metadata for a field and check any transformations performed
    // when loading the sample
    const std::string metadata_path = f + "/metadata";
    conduit::Node metadata = data_schema[metadata_path];
    // Check to make sure that each element in the transformed field are
    // properly normalized
    size_t num_elements = ref_node[ref_pathname].dtype().number_of_elements();
    // Always coerce the reference sample into double
    conduit::Node tmp;
    ref_node[ref_pathname].to_data_type(conduit::DataType::FLOAT64_ID,
                                        tmp[ref_pathname]);
    auto scale =
      metadata.has_child("scale") ? metadata["scale"].as_double() : 1.0;
    auto bias = metadata.has_child("bias") ? metadata["bias"].as_double() : 0.0;
    if (num_elements > 1) {
      for (size_t i = 0; i < num_elements; i++) {
        // Native data type of the real fields are double
        double check = tmp[ref_pathname].as_double_array()[i] * scale + bias;
        if (test_node[test_pathname].dtype().is_float32()) {
          CHECK(test_node[test_pathname].as_float32_array()[i] ==
                Approx(float(check)));
        }
        else if (test_node[test_pathname].dtype().is_float64()) {
          CHECK(test_node[test_pathname].as_double_array()[i] == Approx(check));
        }
      }
    }
    else {
      // Native data type of the real fields are double
      double check = tmp[ref_pathname].as_double() * scale + bias;
      if (test_node[test_pathname].dtype().is_float32()) {
        CHECK(test_node[test_pathname].as_float32() == Approx(float(check)));
      }
      else if (test_node[test_pathname].dtype().is_float64()) {
        CHECK(test_node[test_pathname].as_double() == Approx(check));
      }
    }
  }
}
