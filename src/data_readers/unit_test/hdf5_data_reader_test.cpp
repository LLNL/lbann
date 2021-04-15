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

#if 0
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

// input data
#include "test_data/hrrl_hdf5.prototext" //string experiment_prototext
#include "test_data/hdf5_data_schema.yaml" //string data_schema
#include "test_data/hdf5_experiment_schema.yaml" //string experiment_schema

namespace pb = ::google::protobuf;

void write_file(std::string data, std::string dir, std::string fn); 

TEST_CASE("hdf5 data reader schema tests", "[mpi][reader][hdf5][.filesystem]")
{
  auto& comm = unit_test::utilities::current_world_comm();
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);
  //lbann::construct_std_options();

  // create working directory (/tmp)
  std::string work_dir = create_test_directory("hdf5_reader");

  //=====================================================================
  // instantiate the data readers
  // (note: experiment_prototext contains a specification for a complete
  // experiment, though here we are only using the data_reader portion)
  lbann::options *opts = lbann::options::get();
  opts->set_option("keep_packed_fields", false);
  opts->set_option("preload_data_store", true);
  lbann::generic_data_reader* train_ptr; 
  lbann::generic_data_reader* validate_ptr;
  lbann::generic_data_reader* test_ptr;

  // fix filenames in the prototext
  std::string r_proto(experiment_prototext); //non-const copy
  size_t j1;
  while ((j1 = r_proto.find("WORK_DIR")) != std::string::npos) {
    r_proto.replace(j1, 8, work_dir);
  };
  auto all_readers = instantiate_data_readers(r_proto, comm, train_ptr, validate_ptr, test_ptr);
  REQUIRE(train_ptr != nullptr);
  REQUIRE(validate_ptr != nullptr);
  // (end) instantiate the data readers
  //=====================================================================

  std::stringstream sample_list; //TODO; 


  write_file(data_schema, work_dir, "hdf5_data_schema.yaml");
  write_file(experiment_schema, work_dir, "hdf5_experiment_schema.yaml");
  write_file(sample_list.str(), work_dir, "sample_list_fake.txt");

  SECTION("experiment_schema_1")
  {
    const conduit::Node& s1 = validate_ptr->get_experiment_schema();
    //make non-const copy
    conduit::Node s2(s1);

    //TODO: alter the schema "s2" in some way

    //set_XX_schema() will cause hdf5_data_reader::parse_schemas() to be
    //called, and that should detect errors in the schemas
    validate_ptr->set_experiment_schema(s2);
  }
}

void write_file(std::string data, std::string dir, std::string fn) {
  std::stringstream s;
  s << dir << "/" << fn;
  std::ofstream out(s.str().c_str());
  REQUIRE(out);
  out << data;
  out.close();
}
#endif
