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

#include "TestHelpers.hpp"
#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/proto_common.hpp"
#include <google/protobuf/text_format.h>

#include <conduit/conduit.hpp>
#include <cstdlib>
#include <errno.h>
#include <string.h>

#include "./data_reader_common_HDF5_test_utils.hpp"

#include "./test_data/hdf5_hrrl_test_data_and_schemas.yaml"
#include "lbann/data_readers/data_reader_HDF5.hpp"

// Use a different schema to create a different packing
const std::string packed_hdf5_hrrl_data_sample_id_foobar = R"FOO(000000334:
    samples: [456.288777930614, 231.340700217946, 113.528447010204, 115.115911382861, 116.716861149023, 118.331222098325, 120.52874207647, 122.175220756304, 123.834871115725, 125.507597035081, 126.011234474661, 123.587537036166]
    foo: [15.2486634101312, 0.0426354341969429]
    bar: [64037572840.4818, 5.34505173275895]
    baz: [32.6826031770453]
)FOO";

// Now change the ordering fields in the experiment schema to change the field
// order
const std::string packed_hdf5_hrrl_data_sample_id_foobar_permute =
  R"FOO(000000334:
    samples: [456.288777930614, 231.340700217946, 113.528447010204, 115.115911382861, 116.716861149023, 118.331222098325, 120.52874207647, 122.175220756304, 123.834871115725, 125.507597035081, 126.011234474661, 123.587537036166]
    foo: [0.0426354341969429, 15.2486634101312]
    bar: [5.34505173275895, 64037572840.4818]
    baz: [32.6826031770453]
)FOO";

const std::string hdf5_hrrl_experiment_schema_test_foobar = R"AurthurDent(
Image:
  metadata:
    pack: "samples"
    coerce: "float"
Epmax:
  metadata:
    pack: "foo"
Etot:
  metadata:
    pack: "foo"
N:
  metadata:
    pack: "bar"
T:
  metadata:
    pack: "bar"
alpha:
  metadata:
    pack: "baz"
)AurthurDent";

// Change how the experiment data should be packed and ordered within each field
const std::string hdf5_hrrl_experiment_schema_test_foobar_permute =
  R"AurthurDent(
Image:
  metadata:
    pack: "samples"
    coerce: "float"
Epmax:
  metadata:
    pack: "foo"
    ordering: 5
Etot:
  metadata:
    pack: "foo"
    ordering: 4
N:
  metadata:
    pack: "bar"
    ordering: 3
T:
  metadata:
    pack: "bar"
    ordering: 2
alpha:
  metadata:
    pack: "baz"
    ordering: 1
)AurthurDent";

TEST_CASE("hdf5 data reader transform tests", "[data_reader][hdf5][hrrl]")
{
  // initialize stuff (boilerplate)
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);

  conduit::Node node;
  node.parse(hdf5_hrrl_data_sample, "yaml");

  auto hdf5_dr = std::make_unique<lbann::hdf5_data_reader>();
  DataReaderHDF5WhiteboxTester white_box_tester;

  conduit::Node schema;
  schema.parse(hdf5_hrrl_data_schema, "yaml");

  SECTION("HRRL conduit node normalize")
  {
    // Check to see if each field of the HRRL data set can be properly
    // normalized and avoid clobbering other fields
    std::vector<std::string> fields =
      {"Epmax", "Etot", "Image", "N", "T", "alpha"};
    for (auto f : fields) {
      const std::string test_pathname("RUN_ID/000000334/" + f);
      // Instantiate a fresh copy of the sample
      conduit::Node test_node;
      test_node.parse(hdf5_hrrl_data_sample, "yaml");
      // Select the metadata for a field and transform the sample
      const std::string metadata_path = f + "/metadata";
      conduit::Node metadata = schema[metadata_path];
      if (metadata.has_child("scale")) {
        REQUIRE_NOTHROW(white_box_tester.normalize(*hdf5_dr,
                                                   test_node,
                                                   test_pathname,
                                                   metadata));
      }
      // Check to make sure that each element in the transformed field are
      // properly normalized
      size_t num_elements = node[test_pathname].dtype().number_of_elements();
      if (num_elements > 1) {
        for (size_t i = 0; i < num_elements; i++) {
          double check = node[test_pathname].as_double_array()[i] *
                           metadata["scale"].as_double() +
                         metadata["bias"].as_double();
          CHECK(test_node[test_pathname].as_double_array()[i] == Approx(check));
        }
      }
      else {
        double check =
          node[test_pathname].as_double() * metadata["scale"].as_double() +
          metadata["bias"].as_double();
        CHECK(test_node[test_pathname].as_double() == Approx(check));
      }
      // Check to make sure that none of the other fields have accidentally
      // changed
      for (auto nf : fields) {
        if (nf != f) {
          const std::string ref_pathname("RUN_ID/000000334/" + nf);
          size_t ref_num_elements =
            node[ref_pathname].dtype().number_of_elements();
          if (ref_num_elements > 1) {
            for (size_t i = 0; i < ref_num_elements; i++) {
              CHECK(test_node[ref_pathname].as_double_array()[i] ==
                    node[ref_pathname].as_double_array()[i]);
            }
          }
          else {
            CHECK(test_node[ref_pathname].as_double() ==
                  node[ref_pathname].as_double());
          }
        }
      }
    }
  }

  SECTION("HRRL conduit node repack_image")
  {
    // Check to make sure that the repack_image function properly fails on the
    // HRRL data
    const std::string f = "Image";
    const std::string test_pathname("RUN_ID/000000334/" + f);
    // Instantiate a fresh copy of the sample
    conduit::Node test_node;
    test_node.parse(hdf5_hrrl_data_sample, "yaml");
    // Select the metadata for a field and transform the sample
    const std::string metadata_path = f + "/metadata";
    conduit::Node metadata = schema[metadata_path];
    if (metadata.has_child("channels")) {
      REQUIRE_THROWS(white_box_tester.repack_image(*hdf5_dr,
                                                   test_node,
                                                   test_pathname,
                                                   metadata));
    }
  }
}

TEST_CASE("hdf5 data reader pack test", "[data_reader][hdf5][hrrl][pack]")
{
  // initialize stuff (boilerplate)
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);

  auto hdf5_dr = std::make_unique<lbann::hdf5_data_reader>();
  DataReaderHDF5WhiteboxTester white_box_tester;

  hdf5_dr->set_role("train");

  // Setup the data schema for this HRRL data set
  conduit::Node& data_schema = white_box_tester.get_data_schema(*hdf5_dr);
  data_schema.parse(hdf5_hrrl_data_schema, "yaml");

  // For some reason this approach does not properly setup the data reader
  // conduit::Node data_schema;
  // conduit::Node experiment_schema;
  // data_schema.parse(hdf5_hrrl_data_schema, "yaml");
  // experiment_schema.parse(hdf5_hrrl_experiment_schema_test, "yaml");
  // white_box_tester.set_data_schema(*hdf5_dr, data_schema);
  // white_box_tester.set_experiment_schema(*hdf5_dr, experiment_schema);

  SECTION("HRRL conduit pack node")
  {
    // Read in the experiment schema and setup the data reader
    conduit::Node& experiment_schema =
      white_box_tester.get_experiment_schema(*hdf5_dr);
    experiment_schema.parse(hdf5_hrrl_experiment_schema, "yaml");
    // experiment_schema.print();
    white_box_tester.parse_schemas(*hdf5_dr);

    size_t index = 334;
    // Instantiate a fresh copy of the sample
    conduit::Node test_node;
    test_node.parse(hdf5_hrrl_data_sample_id, "yaml");
    // white_box_tester.print_metadata(*hdf5_dr);
    white_box_tester.pack(*hdf5_dr, test_node, index);

    // Get the reference packed node
    conduit::Node ref_node;
    ref_node.parse(packed_hdf5_hrrl_data_sample_id, "yaml");

    // Check each of the fields to ensure that the packing worked
    std::vector<std::string> fields = {"samples", "responses"};
    for (auto f : fields) {
      const std::string ref_pathname("000000334/" + f);
      size_t ref_num_elements =
        ref_node[ref_pathname].dtype().number_of_elements();
      if (ref_num_elements > 1) {
        for (size_t i = 0; i < ref_num_elements; i++) {
          CHECK(test_node[ref_pathname].as_double_array()[i] ==
                ref_node[ref_pathname].as_double_array()[i]);
        }
      }
      else {
        CHECK(test_node[ref_pathname].as_double() ==
              ref_node[ref_pathname].as_double());
      }
    }
  }
  SECTION("Alternate packings HRRL conduit node")
  {
    // Read in the experiment schema and setup the data reader
    conduit::Node& experiment_schema =
      white_box_tester.get_experiment_schema(*hdf5_dr);
    experiment_schema.parse(hdf5_hrrl_experiment_schema_test_foobar, "yaml");
    white_box_tester.parse_schemas(*hdf5_dr);

    size_t index = 334;
    // Instantiate a fresh copy of the sample
    conduit::Node test_node;
    test_node.parse(hdf5_hrrl_data_sample_id, "yaml");
    white_box_tester.pack(*hdf5_dr, test_node, index);

    // Get the reference packed node
    conduit::Node ref_node;
    ref_node.parse(packed_hdf5_hrrl_data_sample_id_foobar, "yaml");

    // Check each of the fields to ensure that the packing worked
    std::vector<std::string> fields = {"samples", "foo", "bar", "baz"};
    for (auto f : fields) {
      const std::string ref_pathname("000000334/" + f);
      size_t ref_num_elements =
        ref_node[ref_pathname].dtype().number_of_elements();
      if (ref_num_elements > 1) {
        for (size_t i = 0; i < ref_num_elements; i++) {
          CHECK(test_node[ref_pathname].as_double_array()[i] ==
                ref_node[ref_pathname].as_double_array()[i]);
        }
      }
      else {
        CHECK(test_node[ref_pathname].as_double() ==
              ref_node[ref_pathname].as_double());
      }
    }
  }

  SECTION("Alternate packings HRRL conduit node - Permuted Ordering")
  {
    // Read in the experiment schema and setup the data reader
    conduit::Node& experiment_schema =
      white_box_tester.get_experiment_schema(*hdf5_dr);
    experiment_schema.parse(hdf5_hrrl_experiment_schema_test_foobar_permute,
                            "yaml");
    white_box_tester.parse_schemas(*hdf5_dr);

    size_t index = 334;
    // Instantiate a fresh copy of the sample
    conduit::Node test_node;
    test_node.parse(hdf5_hrrl_data_sample_id, "yaml");
    white_box_tester.pack(*hdf5_dr, test_node, index);

    // Get the reference packed node
    conduit::Node ref_node;
    ref_node.parse(packed_hdf5_hrrl_data_sample_id_foobar_permute, "yaml");

    // Check each of the fields to ensure that the packing worked
    std::vector<std::string> fields = {"samples", "foo", "bar", "baz"};
    for (auto f : fields) {
      const std::string ref_pathname("000000334/" + f);
      size_t ref_num_elements =
        ref_node[ref_pathname].dtype().number_of_elements();
      if (ref_num_elements > 1) {
        for (size_t i = 0; i < ref_num_elements; i++) {
          CHECK(test_node[ref_pathname].as_double_array()[i] ==
                ref_node[ref_pathname].as_double_array()[i]);
        }
      }
      else {
        CHECK(test_node[ref_pathname].as_double() ==
              ref_node[ref_pathname].as_double());
      }
    }
  }
}
