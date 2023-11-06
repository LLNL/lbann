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
#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/proto_common.hpp"
#include <google/protobuf/text_format.h>

#include <cstdlib>
#include <errno.h>
#include <string.h>

#include "./data_reader_common_catch2.hpp"
#include "lbann/data_ingestion/readers/data_reader_HDF5.hpp"
#include "lbann/utils/argument_parser.hpp"
#include "lbann/utils/options.hpp"

#include "./data_reader_common_HDF5_test_utils.hpp"

#include "./test_data/hdf5_hrrl_reader.prototext"
#include "./test_data/hdf5_hrrl_test.sample_list"
#include "./test_data/hdf5_hrrl_test_data_and_schemas.yaml"
#include "./test_data/hdf5_hrrl_train.sample_list"
#include "./test_data/hdf5_hrrl_validate.sample_list"
#include "./test_data/hdf5_repack_data_and_schemas.yaml"

namespace pb = ::google::protobuf;

double get_bias_from_node_map(const lbann::hdf5_data_reader* reader,
                              const std::string field_name);

void test_change_bias_value(lbann::hdf5_data_reader* reader,
                            const std::string field_name);

void new_metadata_field(conduit::Node schema, lbann::hdf5_data_reader* reader);

std::string test_field_name("alpha");

TEST_CASE("hdf5 data reader schema tests",
          "[mpi][data_reader][hdf5][.filesystem]")
{
  // initialize stuff (boilerplate)
  auto& comm = unit_test::utilities::current_world_comm();
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);
  auto& arg_parser = lbann::global_argument_parser();
  arg_parser.clear();             // Clear the argument parser.
  lbann::construct_all_options(); // Reset to the default state.

  // create working directory
  std::string work_dir = create_test_directory("hdf5_reader");

  // adjust directory names in the prototext
  std::string prototext(hdf5_hrrl_data_reader_prototext); // non-const copy
  size_t j1;
  while ((j1 = prototext.find("WORK_DIR")) != std::string::npos) {
    prototext.replace(j1, 8, work_dir);
  };

  // write input files to work directory
  write_file(prototext, work_dir, "hdf5_hrrl_reader.prototext");
  write_file(hdf5_hrrl_data_schema, work_dir, "hdf5_hrrl_data_schema.yaml");
  write_file(hdf5_hrrl_experiment_schema,
             work_dir,
             "hdf5_hrrl_experiment_schema.yaml");
  write_file(hdf5_hrrl_train_sample_list,
             work_dir,
             "hdf5_hrrl_train.sample_list");
  write_file(hdf5_hrrl_validate_sample_list,
             work_dir,
             "hdf5_hrrl_validate.sample_list");
  write_file(hdf5_hrrl_test_sample_list,
             work_dir,
             "hdf5_hrrl_test.sample_list");

  // set up the options that the reader expects
  char const* argv[] = {"smiles_functional_black_box.exe",
                        "--use_data_store",
                        "--preload_data_store"};
  int const argc = sizeof(argv) / sizeof(argv[0]);
  REQUIRE_NOTHROW(arg_parser.parse(argc, argv));

  // instantiate the data readers
  lbann::generic_data_reader* train_ptr = nullptr;
  lbann::generic_data_reader* validate_ptr = nullptr;
  lbann::generic_data_reader* tournament_ptr = nullptr;
  lbann::generic_data_reader* test_ptr = nullptr;
  auto all_readers = instantiate_data_readers(prototext,
                                              comm,
                                              train_ptr,
                                              validate_ptr,
                                              test_ptr,
                                              tournament_ptr);

  REQUIRE(train_ptr != nullptr);
  REQUIRE(validate_ptr != nullptr);
  REQUIRE(tournament_ptr != nullptr);
  REQUIRE(test_ptr != nullptr);
  lbann::hdf5_data_reader* train_reader =
    dynamic_cast<lbann::hdf5_data_reader*>(train_ptr);
  lbann::hdf5_data_reader* validate_reader =
    dynamic_cast<lbann::hdf5_data_reader*>(validate_ptr);
  lbann::hdf5_data_reader* tournament_reader =
    dynamic_cast<lbann::hdf5_data_reader*>(tournament_ptr);
  lbann::hdf5_data_reader* test_reader =
    dynamic_cast<lbann::hdf5_data_reader*>(test_ptr);
  REQUIRE(train_reader != nullptr);
  REQUIRE(validate_reader != nullptr);
  REQUIRE(tournament_reader != nullptr);
  REQUIRE(test_reader != nullptr);

  SECTION("hdf5_reader: metadata inheritance")
  {
    // Code in this section tests the inheritance mechanism in the
    // hdf5_data_reader::adjust_metadata() method.  Here, "inheritance"
    // refers to the data and experiment schemas. The inheritance policy
    // is: nodes inherit metadata from parents, disallowing overrides.
    // adjust_metadata() works on each schema independently, so we
    // only need test the data schema.

    // setup: get copies of data_schema, bias, and scale for the Epmax field
    conduit::Node data_schema = train_reader->get_data_schema();
    REQUIRE(data_schema.has_child("Epmax"));
    REQUIRE(data_schema["Epmax"].has_child("metadata"));
    REQUIRE(data_schema["Epmax"]["metadata"].has_child("bias"));
    REQUIRE(data_schema["Epmax"]["metadata"].has_child("scale"));
    double parent_bias = data_schema["Epmax"]["metadata"]["bias"].value();
    double parent_scale = data_schema["Epmax"]["metadata"]["scale"].value();

    SECTION("hdf5_reader: inheritance missing metadata")
    {
      // tests: nodes without metadata inherit from parent
      REQUIRE(data_schema["Epmax"].has_child("horse") == false);
      data_schema["Epmax"]["horse"];
      data_schema["Epmax"]["horse"]["carrot"] = 1.1;
      REQUIRE(data_schema["Epmax"]["horse"].has_child("metadata") == false);
      REQUIRE(data_schema["Epmax"]["horse"]["carrot"].has_child("metadata") ==
              false);
      train_reader->adjust_metadata(&data_schema);

      REQUIRE(data_schema["Epmax"]["horse"].has_child("metadata") == true);
      REQUIRE(data_schema["Epmax"]["horse"]["carrot"].has_child("metadata") ==
              true);

      REQUIRE(data_schema["Epmax"]["horse"]["metadata"].has_child("bias"));
      double carrot_bias =
        data_schema["Epmax"]["horse"]["carrot"]["metadata"]["bias"].value();
      REQUIRE(carrot_bias == parent_bias);
    }

    SECTION("hdf5_reader: inheritance partial metadata #1")
    {
      // tests: nodes with an empty metadata node inherit from parent
      REQUIRE(data_schema["Epmax"].has_child("horse") == false);
      data_schema["Epmax"]["horse"];
      data_schema["Epmax"]["horse"]["metadata"];
      REQUIRE(data_schema["Epmax"]["horse"]["metadata"].has_child("bias") ==
              false);

      train_reader->adjust_metadata(&data_schema);

      REQUIRE(data_schema["Epmax"]["horse"]["metadata"].has_child("bias") ==
              true);
      double horse_bias =
        data_schema["Epmax"]["horse"]["metadata"]["bias"].value();
      REQUIRE(horse_bias == parent_bias);
    }

    SECTION("hdf5_reader: inheritance partial metadata #2")
    {
      // tests: nodes with non-empty metadata node partially inherit from parent
      REQUIRE(data_schema["Epmax"].has_child("horse") == false);
      data_schema["Epmax"]["horse"];
      data_schema["Epmax"]["horse"]["metadata"];
      data_schema["Epmax"]["horse"]["metadata"]["bias"] = 1234.5;

      train_reader->adjust_metadata(&data_schema);

      double horse_bias =
        data_schema["Epmax"]["horse"]["metadata"]["bias"].value();
      double horse_scale =
        data_schema["Epmax"]["horse"]["metadata"]["scale"].value();
      REQUIRE(horse_bias == 1234.5);
      REQUIRE(horse_bias != parent_bias);
      REQUIRE(horse_scale == parent_scale);
    }

  } // SECTION("hdf5_reader: metadata inheritance")

  // What the following sectional tests do (mostly): modify values in input
  // schemas, which result in modifications to entries in the "node_map."
  //
  // The "node_map" contains the names of the fields that will be loaded from
  // disk, along with metadata information, when load_sample() is called.
  //
  // These tests exercise these pathways:
  //   parse_schemas(), get_schema_ptrs(), get_leaves_multi(), get_leaves(),
  //   adjust_metadata(), load_sample(), and possibly others.
  //
  // Note: the reader set() and get() methods operate with copies, not
  // references

  SECTION("hdf5_reader: node_map_bias_existence")
  {
    double bias_1 = get_bias_from_node_map(train_reader, test_field_name);
    double bias_2 = get_bias_from_node_map(validate_reader, test_field_name);
    double bias_3 = get_bias_from_node_map(tournament_reader, test_field_name);
    double bias_4 = get_bias_from_node_map(test_reader, test_field_name);
    REQUIRE(bias_1 == -2.5);
    REQUIRE(bias_2 == -2.5);
    REQUIRE(bias_3 == -2.5);
    REQUIRE(bias_4 == -2.5);
  }

  SECTION("hdf5_reader: modify_existing")
  {
    // change field value in data_schema, test for change in node_map
    test_change_bias_value(train_reader, test_field_name);
    test_change_bias_value(validate_reader, test_field_name);
    test_change_bias_value(tournament_reader, test_field_name);
    test_change_bias_value(test_reader, test_field_name);
  }

  SECTION("hdf5_reader: node_map")
  {
    // test prior condition (xyz == false)
    auto node_map = train_reader->get_node_map();
    REQUIRE(node_map.find(test_field_name) != node_map.end());
    const conduit::Node& nd = node_map[test_field_name];
    REQUIRE(nd.has_child("metadata"));
    REQUIRE(nd["metadata"].has_child("xyz") == false);
    REQUIRE(nd["metadata"].has_child("bias") == true);

    SECTION("hdf5_reader: inheritance_plus")
    {
      // add metadata field to existing nodes
      //  add xyz to data_schema
      conduit::Node data_schema = train_reader->get_data_schema();
      data_schema[test_field_name]["metadata"]["xyz"] = 1234.5678;
      train_reader->set_data_schema(data_schema);

      // test if xyz is now in the node map
      auto node_map_TEST = train_reader->get_node_map();
      REQUIRE(node_map_TEST.find(test_field_name) != node_map.end());
      const conduit::Node& nd2 = node_map_TEST[test_field_name];
      REQUIRE(nd2.has_child("metadata"));
      REQUIRE(nd2["metadata"].has_child("xyz"));
      double test_val = nd2["metadata"]["xyz"].value();
      REQUIRE(test_val == 1234.5678);
    }

    SECTION("hdf5_reader: override")
    {
      // values in the experiment schema should override those in the data
      // schema
      //  metadata fields in the experiment_schema take precedence over fields
      //  in the data_schema, during construction of the node_map.

      conduit::Node data_schema = train_reader->get_data_schema();
      const double old_val =
        data_schema[test_field_name]["metadata"]["bias"].value();
      const double new_val = old_val * 1.23;

      conduit::Node experiment_schema = train_reader->get_experiment_schema();
      REQUIRE(experiment_schema[test_field_name]["metadata"].has_child(
                "bias") == false);
      experiment_schema[test_field_name]["metadata"]["bias"] = new_val;

      train_reader->set_data_schema(data_schema);
      train_reader->set_experiment_schema(experiment_schema);

      auto node_map_TEST = train_reader->get_node_map();
      REQUIRE(node_map_TEST.find(test_field_name) != node_map.end());
      const conduit::Node& nd3 = node_map_TEST[test_field_name];
      REQUIRE(nd3.has_child("metadata"));
      REQUIRE(nd3["metadata"].has_child("bias"));
      double test_val = nd3["metadata"]["bias"].value();
      REQUIRE(test_val == new_val);
    }
  } // SECTION("hdf5_reader: node_map")

  // Cleanup the data readers
  for (auto t : all_readers) {
    delete t.second;
  }
}

//==========================================================================

double get_bias_from_node_map(const lbann::hdf5_data_reader* reader,
                              const std::string field_name)
{
  auto node_map = reader->get_node_map();
  REQUIRE(node_map.find(field_name) != node_map.end());
  conduit::Node a1(node_map[field_name]);
  REQUIRE(a1.has_child("metadata"));
  REQUIRE(a1["metadata"].has_child("bias"));
  double bias = a1["metadata"]["bias"].value();
  return bias;
}

void test_change_bias_value(lbann::hdf5_data_reader* reader,
                            const std::string field_name)
{
  // change the value in the schema
  conduit::Node schema = reader->get_data_schema();
  schema[field_name]["metadata"]["bias"] = 77.2;
  reader->set_data_schema(schema);

  // check that the change appears in the node_map
  double bias = get_bias_from_node_map(reader, field_name);
  REQUIRE(bias == 77.2);
}

void alter(conduit::Node schema, lbann::hdf5_data_reader* reader)
{
  REQUIRE(schema.has_child(test_field_name));
  REQUIRE(schema[test_field_name].has_child("metadata"));
  REQUIRE(schema[test_field_name]["metadata"].has_child("phoo") == false);

  schema[test_field_name]["metadata"]["phoo"] = 42;
  reader->set_data_schema(schema);
  conduit::Node schema_TEST = reader->get_data_schema();
  REQUIRE(schema_TEST.has_child(test_field_name));
  REQUIRE(schema_TEST[test_field_name].has_child("metadata"));
  REQUIRE(schema_TEST[test_field_name]["metadata"].has_child("phoo"));
  int v = schema_TEST[test_field_name]["metadata"]["phoo"].value();
  REQUIRE(v == 42);
}

TEST_CASE("hdf5 data reader repack tests", "[data_reader][hdf5][repack]")
{
  // initialize stuff (boilerplate)
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);

  conduit::Node channels_first_node;
  channels_first_node.parse(hdf5_channels_first_3x4x4_data_sample, "yaml");

  auto hdf5_dr = std::make_unique<lbann::hdf5_data_reader>();
  DataReaderHDF5WhiteboxTester white_box_tester;

  conduit::Node schema;
  schema.parse(hdf5_channels_last_4x4x3_data_schema, "yaml");

  SECTION("HDF5 conduit node repack volume")
  {
    // Check to make sure that the repack_image function properly fails on the
    // a 3D volume of data
    const std::string pathname("000000001");
    const std::string f = "volume";
    const std::string test_pathname(pathname + "/" + f);
    // Instantiate a fresh copy of the sample
    conduit::Node test_node;
    test_node.parse(hdf5_channels_last_4x4x3_data_sample, "yaml");
    // Select the metadata for a field and transform the sample
    const std::string metadata_path = f + "/metadata";
    conduit::Node metadata = schema[metadata_path];
    if (metadata.has_child("channels")) {
      white_box_tester.repack_image(*hdf5_dr,
                                    test_node,
                                    test_pathname,
                                    metadata);
    }
    std::vector<std::string> fields = {"volume"};
    check_node_fields(channels_first_node,
                      test_node,
                      schema,
                      fields,
                      pathname,
                      pathname);
  }
}
