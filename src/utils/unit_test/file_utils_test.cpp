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
// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/utils/file_utils.hpp>

#include <memory>
#include <numeric>
#include <vector>

TEST_CASE ("Testing \"file_join\" function", "[seq][file][utilities]")
{
  REQUIRE(lbann::file::join_path("a") == "a");
  REQUIRE(lbann::file::join_path("a", "b") == "a/b");
  REQUIRE(lbann::file::join_path("a/", "b") == "a//b");
  REQUIRE(lbann::file::join_path(
            "/a", "b", std::string("c"), "d") == "/a/b/c/d");
}
