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

#include <conduit/conduit.hpp>
#include <conduit/conduit_relay_mpi.hpp>
#include <cstdlib>
#include <errno.h>
#include <string.h>

#include "./data_reader_common_HDF5_test_utils.hpp"
#include "./data_reader_common_catch2.hpp"

#include "./test_data/hdf5_3D_UNet_test_data_and_schemas.yaml"
#include "lbann/data_readers/data_reader_HDF5.hpp"

TEST_CASE("HDF5 3D UNet data reader file ingest tests",
          "[.filesystem][data_reader][hdf5][3DUNet][file_ingest]")
{
  // initialize stuff (boilerplate)
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);

  auto hdf5_dr = std::make_unique<lbann::hdf5_data_reader>();
  DataReaderHDF5WhiteboxTester white_box_tester;

  const std::string original_path = "000000001";
  const std::string new_pathname = "000000001";

  conduit::Node& data_schema = white_box_tester.get_data_schema(*hdf5_dr);

  // Read in the experiment schema and setup the data reader
  conduit::Node& experiment_schema =
    white_box_tester.get_experiment_schema(*hdf5_dr);
  experiment_schema.parse(hdf5_3dunet_experiment_schema, "yaml");
  experiment_schema.print();

  // create working directory
  std::string work_dir = create_test_directory("hdf5_reader");

  SECTION("HDF5 3D UNet 2x2x2 write and then read to HDF5 file")
  {
    conduit::Node node;
    node.parse(hdf5_3dunet_2x2x2_data_sample, "yaml");

    // open hdf5 file and obtain a handle
    hid_t h5_id = conduit::relay::io::hdf5_create_file(
      work_dir + "/3DUNet_2x2x2_test_sample.hdf5");
    // write data
    conduit::relay::io::hdf5_write(node, h5_id);
    // close our file
    conduit::relay::io::hdf5_close_file(h5_id);

    hid_t h5_fid = conduit::relay::io::hdf5_open_file_for_read(
      work_dir + "/3DUNet_2x2x2_test_sample.hdf5");

    // Setup the data schema for this 3d UNet data set
    data_schema.parse(hdf5_3dunet_2x2x2_data_schema, "yaml");

    // Parse all of the schemas
    white_box_tester.parse_schemas(*hdf5_dr);

    white_box_tester.print_metadata(*hdf5_dr);

    conduit::Node test_node;
    white_box_tester.load_sample(*hdf5_dr,
                                 test_node[new_pathname],
                                 h5_fid,
                                 original_path);

    // Check to see if the 3D UNet sample can be read via the data
    // reader's load_sample method.  Note that this will coerce and
    // normalize all data fields as specified in the data set and
    // experiment schemas.
    std::vector<std::string> fields = {"segmentation", "volume"};
    check_node_fields(node,
                      test_node,
                      data_schema,
                      fields,
                      original_path,
                      new_pathname);
  }

  SECTION("HDF5 3D UNet 4x4x4 write and then read to HDF5 file")
  {
    conduit::Node node;
    node.parse(hdf5_3dunet_4x4x4_data_sample, "yaml");

    // open hdf5 file and obtain a handle
    hid_t h5_id = conduit::relay::io::hdf5_create_file(
      work_dir + "/3DUNet_4x4x4_test_sample.hdf5");
    // write data
    conduit::relay::io::hdf5_write(node, h5_id);
    // close our file
    conduit::relay::io::hdf5_close_file(h5_id);

    hid_t h5_fid = conduit::relay::io::hdf5_open_file_for_read(
      work_dir + "/3DUNet_4x4x4_test_sample.hdf5");

    // Setup the data schema for this 3d UNet data set
    data_schema.parse(hdf5_3dunet_2x2x2_data_schema, "yaml");

    // Parse all of the schemas
    white_box_tester.parse_schemas(*hdf5_dr);

    white_box_tester.print_metadata(*hdf5_dr);

    conduit::Node test_node;
    white_box_tester.load_sample(*hdf5_dr,
                                 test_node[new_pathname],
                                 h5_fid,
                                 original_path);

    // Check to see if the 3D UNet sample can be read via the data
    // reader's load_sample method.  Note that this will coerce and
    // normalize all data fields as specified in the data set and
    // experiment schemas.
    std::vector<std::string> fields = {"segmentation", "volume"};
    check_node_fields(node,
                      test_node,
                      data_schema,
                      fields,
                      original_path,
                      new_pathname);
  }
}
