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

#include "TestHelpers.hpp"
#include "MPITestHelpers.hpp"
#include "lbann/proto/proto_common.hpp"
#include <lbann.pb.h>
#include <google/protobuf/text_format.h>

#include <errno.h>
#include <string.h>
#include <cstdlib>

#include "data_reader_common_catch2.hpp"
#include "lbann/data_readers/data_reader_HDF5.hpp"

// input data; each of these files contains a single: const std::string;
#include "test_data/hdf5_hrrl_reader.prototext"
#include "test_data/hdf5_hrrl_data_schema.yaml"
#include "test_data/hdf5_hrrl_experiment_schema.yaml"
#include "test_data/hdf5_hrrl_train.sample_list"
#include "test_data/hdf5_hrrl_validate.sample_list"
#include "test_data/hdf5_hrrl_test.sample_list"

namespace pb = ::google::protobuf;

TEST_CASE("hdf5 data reader schema tests", "[mpi][reader][hdf5][.filesystem]")
{
  auto& comm = unit_test::utilities::current_world_comm();
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);

  // create and test working directory 
  std::string work_dir = create_test_directory("hdf5_reader");

  // adjust directory names in the prototext string
  std::string r_proto(hdf5_hrrl_data_reader_prototext); //non-const copy
  size_t j1;
  while ((j1 = r_proto.find("WORK_DIR")) != std::string::npos) {
    r_proto.replace(j1, 8, work_dir);
  };

  // write input files to work directory
  write_file(r_proto, work_dir, "hdf5_hrrl_reader.prototext");
  write_file(hdf5_hrrl_data_schema, work_dir, "hdf5_hrrl_data_schema.yaml");
  write_file(hdf5_hrrl_experiment_schema, work_dir, "hdf5_hrrl_experiment_schema.yaml");
  write_file(hdf5_hrrl_train_sample_list, work_dir, "hdf5_hrrl_train.sample_list");
  write_file(hdf5_hrrl_validate_sample_list, work_dir, "hdf5_hrrl_validate.sample_list");
  write_file(hdf5_hrrl_test_sample_list, work_dir, "hdf5_hrrl_test.sample_list");

  // instantiate the data readers
  lbann::generic_data_reader* train_ptr = nullptr; 
  lbann::generic_data_reader* validate_ptr = nullptr;
  lbann::generic_data_reader* tournament_ptr = nullptr;
  lbann::generic_data_reader* test_ptr = nullptr;
  auto all_readers = instantiate_data_readers(r_proto, comm, train_ptr, validate_ptr, test_ptr, tournament_ptr);

  REQUIRE(train_ptr != nullptr);
  REQUIRE(validate_ptr != nullptr);
  REQUIRE(tournament_ptr != nullptr);
  REQUIRE(test_ptr != nullptr);

  SECTION("experiment schema")
  {
    //This is a stub, as I'm unsure what to test, specifically, to what
    //extent is it the user's responsibility to ensure correct inputs,
    //and to what extent should we error check

    lbann::hdf5_data_reader* reader = dynamic_cast<lbann::hdf5_data_reader*>(train_ptr);
    REQUIRE(reader != nullptr);
    conduit::Node experiment_schema =  reader->get_experiment_schema();
    conduit::Node data_schema =  reader->get_data_schema();

    experiment_schema.print();

    //do things to the experiment and/or data schema, then call:
    reader->set_experiment_schema(experiment_schema);
    reader->set_data_schema(data_schema);

    //note: the calls to set_XX_schema() calls parse_schemas(), 
    //      which in turn calls the following:
    //          adjust_metadata(data_schema)
    //          adjust_metadata(experiment_schema)
    //          get_schema_ptrs(experiment_schema)
    //          get_leaves_multi
    //            calls: get_leaves
    //          construct_linearized_size_lookup_tables
    //            calls: load_sample, get_leaves
    //
    //      At this point, each object in "m_useme_node_map_ptrs:"
    //         1. is a leaf node whose values will be used in the experiment
    //         2. has a "metatdata" child node that contains instructions for
    //            munging the data, i.e: scale, bias, ordering, coersion, packing, etc.
    //         3. munging, etc, takes place during calls to load_sample(),
    //            which is called by construct_linearized_size_lookup_tables (above),
    //            and also for each sample during preloading (below)
  }


}
