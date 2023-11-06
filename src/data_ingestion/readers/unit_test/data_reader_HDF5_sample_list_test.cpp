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
#include "lbann/data_ingestion/readers/data_reader_HDF5.hpp"
#include "lbann/data_ingestion/readers/sample_list_impl.hpp"
#include "lbann/data_ingestion/readers/sample_list_open_files_impl.hpp"
#include "lbann/proto/lbann.pb.h"
#include <google/protobuf/text_format.h>
#include <lbann/base.hpp>

namespace {

std::string const probies_hdf5_legacy_sample_list =
  R"ptext(CONDUIT_HDF5_INCLUSION
12 0 3
/p/vast1/lbann/datasets/PROBIES/h5_data/
h5out_30.h5 4 0 RUN_ID/000000138 RUN_ID/000000139 RUN_ID/000000140 RUN_ID/000000141
h5out_31.h5 4 0 RUN_ID/000000000 RUN_ID/000000001 RUN_ID/000000002 RUN_ID/000000003
h5out_32.h5 4 0 RUN_ID/000000004 RUN_ID/000000005 RUN_ID/000000006 RUN_ID/000000007
)ptext";

std::string const probies_hdf5_multi_sample_inclusion_v2_sample_list =
  R"ptext(MULTI-SAMPLE_INCLUSION_V2
12 3
/p/vast1/lbann/datasets/PROBIES/h5_data/
h5out_30.h5 4 RUN_ID/000000138 RUN_ID/000000139 RUN_ID/000000140 RUN_ID/000000141
h5out_31.h5 4 RUN_ID/000000000 RUN_ID/000000001 RUN_ID/000000002 RUN_ID/000000003
h5out_32.h5 4 RUN_ID/000000004 RUN_ID/000000005 RUN_ID/000000006 RUN_ID/000000007
)ptext";

} // namespace

TEST_CASE("hdf5 data reader",
          "[mpi][data_reader][sample_list][hdf5][.filesystem]")
{
  auto& comm = unit_test::utilities::current_world_comm();

  auto hdf5_dr = std::make_unique<lbann::hdf5_data_reader>();
  // Avoid the sample list code checking that the files really exist
  // in the file system
  hdf5_dr->get_sample_list().unset_data_file_check();

  SECTION("CONDUIT_HDF5_INCLUSION")
  {
    std::string const sample_list = probies_hdf5_legacy_sample_list;
    std::istringstream iss(sample_list);
    hdf5_dr->get_sample_list().load(iss, comm, true);
    hdf5_dr->get_sample_list().all_gather_packed_lists(comm);
    std::string buf;
    hdf5_dr->get_sample_list().to_string(buf);
    CHECK(sample_list == buf);
  }

  SECTION("MULTI-SAMPLE_INCLUSION_V2")
  {
    std::string const sample_list =
      probies_hdf5_multi_sample_inclusion_v2_sample_list;
    std::istringstream iss(sample_list);
    hdf5_dr->get_sample_list().load(iss, comm, true);
    hdf5_dr->get_sample_list().all_gather_packed_lists(comm);
    std::string buf;
    hdf5_dr->get_sample_list().to_string(buf);
    CHECK(sample_list == buf);
  }
}
