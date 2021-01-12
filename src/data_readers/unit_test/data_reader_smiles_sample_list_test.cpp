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

// The code being tested
#include "lbann/data_readers/data_reader_smiles.hpp"
#include "lbann/data_readers/sample_list_impl.hpp"
#include "lbann/data_readers/sample_list_open_files_impl.hpp"

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

std::string const multi_sample_inclusion_v2_list = R"ptext(MULTI-SAMPLE_INCLUSION_V2
18 1
/foo/bar/
blah.txt 18 0 ... 5 7 ... 10 12 ... 19
)ptext";

std::string const multi_sample_inclusion_v2_list_multi_files = R"ptext(MULTI-SAMPLE_INCLUSION_V2
36 2
/foo/bar/
baz.txt 18 0 ... 5 7 ... 10 12 ... 19
blah.txt 18 0 ... 4 6 ... 11 13 ... 19
)ptext";

/// /p/vast1/atom/arthor_dbs/REAL/S/H30/SH30M600.smi
std::string const SH30M600_smi =
R"smile(CC(=O)N[C@@H]1[C@@H](O)C[C@@](O)(C(=O)N2CCNC(=O)C2C(N)=O)O[C@H]1[C@H](O)[C@H](O)CO s_22____17553516____16656074
CC(=O)N[C@@H]1[C@@H](O)C[C@@](O)(C(=O)N2CC(=O)NCC2C(N)=O)O[C@H]1[C@H](O)[C@H](O)CO s_22____14789470____16656074)smile";

/// /p/vast1/atom/arthor_dbs/REAL/S/H30/SH30M600.smi - malformed with
/// a % and a ' ' inserted into each string
// std::string const SH30M600_smi_malformed =
// R"smile(CC(=O)N[C@@H]1[C@@H](O)C[C@@](O)%(C(=O)N2CCNC(=O)C2C(N)=O)O[C@H]1[C@H](O)[C@H](O)CO s_22____17553516____16656074
// CC(=O)N[C@@H]1[C@@H](O)C[C@@](O)(C(=O)N2CC (=O)NCC2C(N)=O)O[C@H]1[C@H](O)[C@H](O)CO s_22____14789470____16656074)smile";

// std::vector<size_t,size_t> SH30M600_offsets = {{0,100}, {101, 75}}
// std::vector<size_t,size_t> SH30M600_offsets_off_by_one_under = {{0,99}, {101, 75}}
// std::vector<size_t,size_t> SH30M600_offsets_off_by_one_over = {{0,101}, {101, 75}}
// std::vector<size_t,size_t> SH30M600_offsets_overlapping = {{0,100}, {100, 75}}

}// namespace <anon>

using unit_test::utilities::IsValidPtr;
TEST_CASE("Sample list", "[mpi][data reader]")
{
  //using DataType = float; commented out to silence compiler

  auto& comm = unit_test::utilities::current_world_comm();

  auto const& g = comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);

  lbann::smiles_data_reader *smiles = new lbann::smiles_data_reader(true /*shuffle*/);
  // Avoid the sample list code checking that the files really exist
  // in the file system
  smiles->get_sample_list().unset_data_file_check();


  SECTION("CONDUIT_HDF5_INCLUSION - one range")
  {
    std::string const sample_list = sample_list_one_range;
    std::istringstream iss(sample_list);
    smiles->get_sample_list().load(iss, comm, true);
    std::string buf;
    smiles->get_sample_list().to_string(buf);
    CHECK(sample_list == buf);
  }
  SECTION("CONDUIT_HDF5_INCLUSION - many ranges")
  {
    std::string const sample_list = sample_list_many_ranges;
    std::istringstream iss(sample_list);
    smiles->get_sample_list().load(iss, comm, true);
    std::string buf;
    smiles->get_sample_list().to_string(buf);
    CHECK(sample_list == buf);
  }
  SECTION("CONDUIT_HDF5_INCLUSION - load from string many ranges")
  {
    std::string const sample_list = sample_list_many_ranges;
    smiles->get_sample_list().load_from_string(sample_list);
    std::string buf;
    smiles->get_sample_list().to_string(buf);
    CHECK(sample_list == buf);

  }
  SECTION("MULTI-SAMPLE_INCLUSION_V2")
  {
    std::string const sample_list = multi_sample_inclusion_v2_list;
    std::istringstream iss(sample_list);
    smiles->get_sample_list().load(iss, comm, true);
    std::string buf;
    smiles->get_sample_list().to_string(buf);
    CHECK(sample_list == buf);
  }
  SECTION("MULTI-SAMPLE_INCLUSION_V2 - multiple files")
  {
    std::string const sample_list = multi_sample_inclusion_v2_list_multi_files;
    std::istringstream iss(sample_list);
    smiles->get_sample_list().load(iss, comm, true);
    std::string buf;
    smiles->get_sample_list().to_string(buf);
    CHECK(sample_list == buf);
  }
  SECTION("Test SMILES ingestion")
  {
    // std::string const sample_list = multi_sample_inclusion_v2_list_multi_files;
    // std::istringstream iss(sample_list);
    // smiles->get_sample_list().load(iss, comm, true);
    // std::string buf;
    // smiles->get_sample_list().to_string(buf);
    // CHECK(sample_list == buf);
    std::string const smiles_str = SH30M600_smi;
    std::istringstream iss(smiles_str);
    //    smiles->get_raw_sample(iss, 0);
  }
  // SECTION("Test SMILES ingestion - malformed SMILE")
  // {
  //   // std::string const sample_list = multi_sample_inclusion_v2_list_multi_files;
  //   // std::istringstream iss(sample_list);
  //   // smiles->get_sample_list().load(iss, comm, true);
  //   // std::string buf;
  //   // smiles->get_sample_list().to_string(buf);
  //   // CHECK(sample_list == buf);
  //   std::string const smiles_str = SH30M600_smi_malformed;
  //   std::istringstream iss(smiles_str);
  //   //    smiles->get_raw_sample(iss, 0);
  // }
}
