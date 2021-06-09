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
#if 0
#include "lbann/data_readers/data_reader_HDF5.hpp"


// input data; each of these files contains a single const std::string
#include "test_data/hdf5_reader_hrrl.prototext" //experiment_prototext
#include "test_data/hdf5_data_schema_hrrl.yaml" //data_schema
#include "test_data/hdf5_experiment_schema_hrrl.yaml" //experiment_schema
#include "test_data/hdf5_hrrl_train.sample_list" //sample_list_train
#include "test_data/hdf5_hrrl_validate.sample_list" //sample_list_validate
#include "test_data/hdf5_hrrl_test.sample_list" //sample_list_test

namespace pb = ::google::protobuf;

TEST_CASE("hdf5 data reader schema tests", "[mpi][reader][hdf5][.filesystem]")
{
  auto& comm = unit_test::utilities::current_world_comm();
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);

  // create and test working directory 
  std::string work_dir = create_test_directory("hdf5_reader");

  // fix directory names in the prototext
  std::string r_proto(experiment_prototext); //non-const copy
  size_t j1;
  while ((j1 = r_proto.find("WORK_DIR")) != std::string::npos) {
    r_proto.replace(j1, 8, work_dir);
  };


  // write input files to work directory
  write_file(r_proto, work_dir, "hdf5_hrrl_reader.prototext")
  write_file(data_schema, work_dir, "hdf5_hrrl_data_schema.yaml");
  write_file(experiment_schema, work_dir, "hdf5_hrrl_experiment_schema.yaml");
  write_file(sample_list_train, work_dir, "train_hrrl.sample_list")
  write_file(sample_list_validate, work_dir, "validate_hrrl.sample_list")
  write_file(sample_list_test, work_dir, "test.sample_list")

  // instantiate the data readers
  lbann::options *opts = lbann::options::get();
  lbann::generic_data_reader* train_ptr; 
  lbann::generic_data_reader* validate_ptr;
  lbann::generic_data_reader* tournament_ptr;
  lbann::generic_data_reader* test_ptr;
  auto all_readers = instantiate_data_readers(r_proto, comm, train_ptr, validate_ptr, test_ptr, tournament_ptr);
  REQUIRE(train_ptr != nullptr);
  REQUIRE(validate_ptr != nullptr);
  REQUIRE(tournament_ptr != nullptr);
  REQUIRE(test_ptr != nullptr);


  SECTION("experiment_schema_1")
  {
  #if 0
    const conduit::Node& s1 = validate_ptr->get_experiment_schema();
    //make non-const copy
    conduit::Node s2(s1);

    //TODO: alter the schema "s2" in some way

    //set_XX_schema() will cause hdf5_data_reader::parse_schemas() to be
    //called, and that should detect errors in the schemas
    validate_ptr->set_experiment_schema(s2);
    #endif
  }
}
#endif
