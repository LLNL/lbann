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
#include <lbann/base.hpp>
#include <lbann.pb.h>
#include <google/protobuf/text_format.h>


namespace pb = ::google::protobuf;

namespace {


// "hdf5_reader" string is defined here as a "const std::string".
#include "hdf5_reader.prototext.inc"

auto make_data_reader(lbann::lbann_comm& comm)
{
std::cout << " starting make_data_reader\n";
  lbann_data::LbannPB reader;
  if (!pb::TextFormat::ParseFromString(hdf5_reader, &reader))
    throw "Parsing protobuf failed.";
  return reader;
}


TEST_CASE("hdf5 data reader", "[mpi][reader][hdf5]")
{
  auto& world_comm = unit_test::utilities::current_world_comm();
  //  int rank_in_world = world_comm.get_rank_in_world();
  WARN("in TEST_CASE");

  auto reader = make_data_reader(world_comm);

  SECTION("a test section")
  {
    WARN("in a test section");
  }
}

} // namespace {
