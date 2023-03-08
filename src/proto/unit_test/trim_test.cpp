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

#include <lbann/proto/proto_common.hpp>

#include <string>

TEST_CASE("Testing string trimming", "[proto][utilities]")
{
  SECTION("Leading spaces")
  {
    CHECK(lbann::trim(" my string") == "my string");
    CHECK(lbann::trim("\nmy string") == "my string");
    CHECK(lbann::trim("\tmy string") == "my string");
    CHECK(lbann::trim(" \n\tmy string") == "my string");
    CHECK(lbann::trim("      my string") == "my string");
  }
  SECTION("Trailing spaces")
  {
    CHECK(lbann::trim("my string ") == "my string");
    CHECK(lbann::trim("my string\n") == "my string");
    CHECK(lbann::trim("my string\t") == "my string");
    CHECK(lbann::trim("my string \n\t") == "my string");
    CHECK(lbann::trim("my string    ") == "my string");
  }
  SECTION("Leading and trailing spaces")
  {
    CHECK(lbann::trim(" my string ") == "my string");
    CHECK(lbann::trim(" my string\n") == "my string");
    CHECK(lbann::trim(" my string\t") == "my string");
    CHECK(lbann::trim(" my string \n\t") == "my string");
    CHECK(lbann::trim(" my string    ") == "my string");

    CHECK(lbann::trim("\nmy string ") == "my string");
    CHECK(lbann::trim("\nmy string\n") == "my string");
    CHECK(lbann::trim("\nmy string\t") == "my string");
    CHECK(lbann::trim("\nmy string \n\t") == "my string");
    CHECK(lbann::trim("\nmy string    ") == "my string");

    CHECK(lbann::trim("\tmy string ") == "my string");
    CHECK(lbann::trim("\tmy string\n") == "my string");
    CHECK(lbann::trim("\tmy string\t") == "my string");
    CHECK(lbann::trim("\tmy string \n\t") == "my string");
    CHECK(lbann::trim("\tmy string    ") == "my string");

    CHECK(lbann::trim(" \n\tmy string ") == "my string");
    CHECK(lbann::trim(" \n\tmy string\n") == "my string");
    CHECK(lbann::trim(" \n\tmy string\t") == "my string");
    CHECK(lbann::trim(" \n\tmy string \n\t") == "my string");
    CHECK(lbann::trim(" \n\tmy string    ") == "my string");

    CHECK(lbann::trim("  my string ") == "my string");
    CHECK(lbann::trim("   my string\n") == "my string");
    CHECK(lbann::trim("    my string\t") == "my string");
    CHECK(lbann::trim("     my string \n\t") == "my string");
    CHECK(lbann::trim("      my string    ") == "my string");
  }
  SECTION("Neither leading nor trailing spaces")
  {
    CHECK(lbann::trim("my string") == "my string");
    CHECK(lbann::trim("lbann") == "lbann");
  }
  SECTION("Only spaces")
  {
    CHECK(lbann::trim(" ") == "");
    CHECK(lbann::trim("\n") == "");
    CHECK(lbann::trim("\t") == "");
    CHECK(lbann::trim(" \n\t") == "");
    CHECK(lbann::trim("     \t\n\t") == "");
  }
  SECTION("Empty string") { CHECK(lbann::trim("") == ""); }
}
