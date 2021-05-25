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

#include <lbann/comm_impl.hpp>

#include "MPITestHelpers.hpp"

TEST_CASE("Example: test of Broadcast", "[mpi][example]")
{
  auto& world_comm = unit_test::utilities::current_world_comm();
  int rank_in_world = world_comm.get_rank_in_world();

  SECTION("Scalar broadcast")
  {
    int value = (rank_in_world == 0 ? 13 : -1);
    world_comm.world_broadcast(0, value);

    REQUIRE(value == 13);
  }

  SECTION("Vector broadcast")
  {
    std::vector<float> true_values = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> values =
      (rank_in_world == 0
       ? true_values
       : std::vector<float>(4, -1.f));

    world_comm.world_broadcast(0, values.data(), values.size());

    REQUIRE(values == true_values);
  }
}
