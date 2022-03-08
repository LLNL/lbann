////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#include <lbann/base.hpp>
#include <lbann/proto/proto_common.hpp>

#include <string>
#include <vector>

TEST_CASE("Testing parse_list", "[proto][utilities]")
{
  SECTION("execution_mode")
  {
    const std::vector<lbann::execution_mode> expected =
      { lbann::execution_mode::training,
        lbann::execution_mode::validation,
        lbann::execution_mode::testing };

    auto const answer =
      lbann::parse_list<lbann::execution_mode>("train validate test");
    CHECK(answer == expected);
    CHECK(
      lbann::parse_list<lbann::execution_mode>("")
      == std::vector<lbann::execution_mode>{});
    CHECK(
      lbann::parse_list<lbann::execution_mode>(" ")
      == std::vector<lbann::execution_mode>{});

    CHECK_THROWS(
      lbann::parse_list<lbann::execution_mode>("banana tuna salad"));
  }

  SECTION("std::string")
  {
    const std::vector<std::string> expected = { "this", "is", "a", "test" };
    auto const answer =
      lbann::parse_list<std::string>("this is a test");
    CHECK(answer == expected);
    CHECK(
      lbann::parse_list<std::string>("") == std::vector<std::string>{});

  }

  SECTION("int")
  {
    const std::vector<int> expected = { 1, 2, 3, 4, 5 };
    auto const answer =
      lbann::parse_list<int>("1 2 3 4 5");
    CHECK(answer == expected);
    CHECK(lbann::parse_list<int>("") == std::vector<int>{});
    CHECK(lbann::parse_list<int>(" ") == std::vector<int>{});
  }
}
