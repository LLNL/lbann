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

// The code being tested
#include "lbann/data_ingestion/readers/data_reader_smiles.hpp"
#include "lbann/data_ingestion/readers/sample_list_impl.hpp"
#include "lbann/data_ingestion/readers/sample_list_open_files_impl.hpp"

namespace pb = ::google::protobuf;

namespace {

std::string const sample_list_one_range = R"ptext(CONDUIT_HDF5_INCLUSION
20 0 1
/foo/bar/
baz.txt 20 0 0 ... 19
)ptext";

std::string const sample_list_many_ranges = R"ptext(CONDUIT_HDF5_INCLUSION
18 2 1
/foo/bar/
baz.txt 18 2 0 ... 5 7 ... 10 12 ... 19
)ptext";

std::string const multi_sample_inclusion_v2_list =
  R"ptext(MULTI-SAMPLE_INCLUSION_V2
18 1
/foo/bar/
blah.txt 18 0 ... 5 7 ... 10 12 ... 19
)ptext";

std::string const multi_sample_inclusion_v2_list_multi_files =
  R"ptext(MULTI-SAMPLE_INCLUSION_V2
36 2
/foo/bar/
baz.txt 18 0 ... 5 7 ... 10 12 ... 19
blah.txt 18 0 ... 4 6 ... 11 13 ... 19
)ptext";

std::string const multi_sample_inclusion_v2_list_many_files =
  R"ptext(MULTI-SAMPLE_INCLUSION_V2
72 4
/foo/bar/
baz.txt 18 0 ... 5 7 ... 10 12 ... 19
blah.txt 18 0 ... 4 6 ... 11 13 ... 19
caffe.txt 18 0 ... 3 5 ... 12 14 ... 19
babe.txt 18 0 ... 6 8 ... 14 16 ... 19
)ptext";

// std::vector<size_t,size_t> SH30M600_offsets = {{0,100}, {101, 75}}
// std::vector<size_t,size_t> SH30M600_offsets_off_by_one_under = {{0,99}, {101,
// 75}} std::vector<size_t,size_t> SH30M600_offsets_off_by_one_over = {{0,101},
// {101, 75}} std::vector<size_t,size_t> SH30M600_offsets_overlapping =
// {{0,100}, {100, 75}}

} // namespace

using unit_test::utilities::IsValidPtr;
TEST_CASE("Sample list", "[mpi][data_reader][smiles]")
{
  // using DataType = float; commented out to silence compiler

  auto& comm = unit_test::utilities::current_world_comm();
  lbann::init_random(0, 2);
  lbann::init_data_seq_random(42);
  auto& arg_parser = lbann::global_argument_parser();
  arg_parser.clear(); // Clear the argument parser.
  lbann::construct_all_options();

  auto const& g = comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);

  auto smiles = std::make_unique<lbann::smiles_data_reader>(/*shuffle=*/true);
  // Avoid the sample list code checking that the files really exist
  // in the file system
  smiles->get_sample_list().unset_data_file_check();

  SECTION("CONDUIT_HDF5_INCLUSION - one range")
  {
    std::string const sample_list = sample_list_one_range;
    std::istringstream iss(sample_list);
    smiles->get_sample_list().load(iss, comm, true);
    smiles->get_sample_list().all_gather_packed_lists(comm);
    std::string buf;
    smiles->get_sample_list().to_string(buf);
    CHECK(sample_list == buf);
  }
  SECTION("CONDUIT_HDF5_INCLUSION - many ranges")
  {
    std::string const sample_list = sample_list_many_ranges;
    std::istringstream iss(sample_list);
    smiles->get_sample_list().load(iss, comm, true);
    smiles->get_sample_list().all_gather_packed_lists(comm);
    std::string buf;
    smiles->get_sample_list().to_string(buf);
    CHECK(sample_list == buf);
  }
  SECTION("CONDUIT_HDF5_INCLUSION - load from string many ranges")
  {
    std::string const sample_list = sample_list_many_ranges;
    smiles->get_sample_list().load_from_string(sample_list, comm, true);
    smiles->get_sample_list().all_gather_packed_lists(comm);
    std::string buf;
    smiles->get_sample_list().to_string(buf);
    CHECK(sample_list == buf);
  }
  SECTION("MULTI-SAMPLE_INCLUSION_V2")
  {
    std::string const sample_list = multi_sample_inclusion_v2_list;
    std::istringstream iss(sample_list);
    smiles->get_sample_list().load(iss, comm, true);
    smiles->get_sample_list().all_gather_packed_lists(comm);
    std::string buf;
    smiles->get_sample_list().to_string(buf);
    CHECK(sample_list == buf);
  }
  SECTION("MULTI-SAMPLE_INCLUSION_V2 - multiple files")
  {
    std::string const sample_list = multi_sample_inclusion_v2_list_multi_files;
    std::istringstream iss(sample_list);
    smiles->get_sample_list().load(iss, comm, true);
    smiles->get_sample_list().all_gather_packed_lists(comm);
    std::string buf;
    smiles->get_sample_list().to_string(buf);
    CHECK(sample_list == buf);
  }
  SECTION("MULTI-SAMPLE_INCLUSION_V2 - many files")
  {
    std::string const sample_list = multi_sample_inclusion_v2_list_many_files;
    std::istringstream iss(sample_list);
    smiles->get_sample_list().load(iss, comm, true);
    smiles->get_sample_list().all_gather_packed_lists(comm);
    std::string buf;
    smiles->get_sample_list().to_string(buf);
    CHECK(sample_list == buf);
  }
}
