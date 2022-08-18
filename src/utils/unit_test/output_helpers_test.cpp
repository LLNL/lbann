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

#include "Catch2BasicSupport.hpp"

#include <lbann/utils/output_helpers.hpp>

#include <sstream>

TEST_CASE("ASCI color codes", "[utils][output]")
{
  using namespace lbann;
  std::ostringstream oss;
  SECTION("OSS::Clear()")
  {
    oss << "hello";
    std::ostringstream{"bananas", std::ios_base::ate}.swap(oss);
    oss << "world";
    REQUIRE(oss.str() == "bananasworld");
  }
  SECTION("Colorized string")
  {
    oss << blue << "hello" << nocolor;
    REQUIRE(oss.str() == "\x1b[34mhello\x1b[0m");
    REQUIRE(strip_ansi_csis(oss.str()) == "hello");
  }

  SECTION("Remove ANSI CSIs")
  {
    REQUIRE(strip_ansi_csis("\x1b[34mhello\x1b[0m") == "hello");

    oss << blue << "he" << cyan << "ll" << red << "o " << white << "wo"
        << magenta << "rl" << green << "d!" << nocolor << clearline;
    REQUIRE(strip_ansi_csis(oss.str()) == "hello world!");
  }
}

TEST_CASE("Truncate to width", "[utils][output][string]")
{
  using lbann::truncate_to_width;
  CHECK(truncate_to_width("bananas and apples", 10) == "bananas...");
  CHECK(truncate_to_width("bananas", 10) == "bananas");
  CHECK(truncate_to_width("whatchamacallit", 10) == "whatcha...");
}
