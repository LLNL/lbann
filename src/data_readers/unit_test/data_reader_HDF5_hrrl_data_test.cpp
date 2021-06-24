////////////////////////////////////////////////////////////////////////////////
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
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch.hpp>

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"
#include "lbann/proto/proto_common.hpp"
#include <google/protobuf/text_format.h>
#include <lbann.pb.h>

#include <conduit/conduit.hpp>
#include <cstdlib>
#include <errno.h>
#include <string.h>

//#include "./data_reader_common_catch2.hpp"
#include "lbann/data_readers/data_reader_HDF5.hpp"

// input data; each of these contain a single variable: "const std::string"
#include "./test_data/hdf5_hrrl_data_schema.yaml"
#include "./test_data/hdf5_hrrl_experiment_schema.yaml"
#include "./test_data/hdf5_hrrl_reader.prototext"
#include "./test_data/hdf5_hrrl_test.sample_list"
#include "./test_data/hdf5_hrrl_train.sample_list"
#include "./test_data/hdf5_hrrl_validate.sample_list"

namespace pb = ::google::protobuf;

//std::string test_field_name("alpha");

const std::string hdf5_hrrl_data_sample =R"FOO(RUN_ID:
  000000334:
    Epmax: 15.2486634101312
    Etot: 0.0426354341969429
    Image: [456.288777930614, 231.340700217946, 113.528447010204, 115.115911382861, 116.716861149023, 118.331222098325, 120.52874207647, 122.175220756304, 123.834871115725, 125.507597035081, 126.011234474661, 123.587537036166]
    N: 64037572840.4818
    T: 5.34505173275895
    alpha: 32.6826031770453
)FOO";

const std::string hdf5_hrrl_data_schema_test = R"AurthurDent(
# Re, the "ordering" fields: ordering is relative and need not be unique;
# it specifies, e.g, the order in which a set of scalars
# would be appended to a vector.
#
# metadata values in the below schema can be over-ridden by values in
# the experiment_schema.yaml
#
# For reference: the metadata nodes may contain additional info,
# e.g, scale and bias for normalization.
#
# The intent is that the the schema and metadata values below should
# be reasonably static, while the experiment_schema species the
# subset of values to use in an experiment
#
#
Image:
  metadata:
    dims: [4,3]
    channels: 1
    ordering: 0
    scale: [1.5259021896696422e-05]
    bias: [-1.5259021896696422e-05]
Epmax:
  metadata:
    ordering: 10
    scale: 0.1
    bias: -1.0
Etot:
  metadata:
    ordering: 20
    scale: 0.3916485873519399
    bias: -0.00039973613068075743
T:
  metadata:
    ordering: 50
    scale: 0.125
    bias: -0.25
alpha:
  metadata:
    ordering: 60
    scale: 0.1
    bias: -2.5

N:
  metadata:
    ordering: 40
    scale: 3.1662826662374707e-13
    bias: -0.001001267234978943
Xshift:
  metadata:
    ordering: 70
Yshift:
  metadata:
    ordering: 80
)AurthurDent";



TEST_CASE("hdf5 data reader transform tests",
          "[mpi][data_reader][hdf5][hrrl][.filesystem]")
{
  // initialize stuff (boilerplate)
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);

  conduit::Node node;
  node.parse(hdf5_hrrl_data_sample, "yaml");

  lbann::hdf5_data_reader* hdf5_dr = new lbann::hdf5_data_reader();
  conduit::Node schema;
  schema.parse(hdf5_hrrl_data_schema_test, "yaml");

  SECTION("HRRL conduit node normalize")
  {
    // Check to see if each field of the HRRL data set can be properly normalized and
    // avoid clobbering other fields
    std::vector<std::string>fields = {"Epmax", "Etot", "Image", "N", "T", "alpha"};
    for (auto f : fields) {
      const std::string test_pathname("RUN_ID/000000334/" + f);
      // Instantiate a fresh copy of the sample
      conduit::Node test_node;
      test_node.parse(hdf5_hrrl_data_sample, "yaml");
      // Select the metadata for a field and transform the sample
      const std::string metadata_path = f + "/metadata";
      conduit::Node metadata = schema[metadata_path];
      if (metadata.has_child("scale")) {
        hdf5_dr->normalize(test_node, test_pathname, metadata);
      }
      // Check to make sure that each element in the transformed field are properly normalized
      size_t num_elements = node[test_pathname].dtype().number_of_elements();
      if(num_elements > 1) {
        for(size_t i = 0; i < num_elements; i++) {
          double check = node[test_pathname].as_double_array()[i] * metadata["scale"].as_double() + metadata["bias"].as_double();
          CHECK(test_node[test_pathname].as_double_array()[i] == Approx(check));
        }
      }else {
        double check = node[test_pathname].as_double() * metadata["scale"].as_double() + metadata["bias"].as_double();
        CHECK(test_node[test_pathname].as_double() == Approx(check));
      }
      // Check to make sure that none of the other fields have accidentally changed
      for(auto nf : fields) {
        if(nf != f) {
          const std::string ref_pathname("RUN_ID/000000334/" + nf);
          size_t ref_num_elements = node[ref_pathname].dtype().number_of_elements();
          if(ref_num_elements > 1) {
            for(size_t i = 0; i < ref_num_elements; i++) {
              CHECK(test_node[ref_pathname].as_double_array()[i] == node[ref_pathname].as_double_array()[i]);
            }
          }
          else {
            CHECK(test_node[ref_pathname].as_double() == node[ref_pathname].as_double());
          }
        }
      }
    }

  }
}
