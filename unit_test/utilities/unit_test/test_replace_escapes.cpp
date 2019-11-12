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

#include "ReplaceEscapes.hpp"

#include <lbann/utils/system_info.hpp>

#include <string>

// Stub the system info
class TestSystemInfo : public lbann::utils::SystemInfo
{
public:
  std::string pid() const override { return "321"; }
  std::string host_name() const override { return "test.host.name"; }
  int mpi_rank() const override { return 123; }
  int mpi_size() const override { return 432; }
  std::string env_variable_value(std::string const& var_name) const override
  {
    if (var_name == "PUMPKIN")
      return "pie";
    if (var_name == "CRANBERRY")
      return "sauce";
    return "";
  }
}; // class TestSystemInfo

// Bring the function under test into scope.
using unit_test::utilities::replace_escapes;
using unit_test::utilities::BadSubstitutionPattern;

TEST_CASE("Subtitution of patterns in strings", "[seq][utils][testing]")
{
  TestSystemInfo sys_info;

  SECTION("No patterns leaves the string unchanged.")
  {
    std::string test_string = "I am a string";
    CHECK(replace_escapes(test_string, sys_info) == test_string);
  }

  SECTION("Subtitute %p for process ID")
  {
    auto pid = sys_info.pid();
    CHECK(replace_escapes("%p", sys_info) == pid);
    CHECK(replace_escapes("%p_apple", sys_info) == pid+"_apple");
    CHECK(replace_escapes("%p%p", sys_info) == pid + pid);
    CHECK(replace_escapes("%pap%pple_%p", sys_info)
          == pid+"ap"+pid+"ple_"+pid);
  }

  SECTION("Substitute %h for hostname")
  {
    auto host = sys_info.host_name();
    CHECK(replace_escapes("%h", sys_info) == host);
    CHECK(replace_escapes("Tahitian %h farm", sys_info) == "Tahitian "+host+" farm");
    CHECK(replace_escapes("%h%h", sys_info) == host + host);
    CHECK(replace_escapes("G%hs%hsss", sys_info) == "G"+host+"s"+host+"sss");
  }

  SECTION("Substitute %r for MPI rank")
  {
    auto rank = std::to_string(sys_info.mpi_rank());
    CHECK(replace_escapes("%r", sys_info) == rank);
    CHECK(replace_escapes("I have %r cats", sys_info)
          == "I have "+rank+" cats");
    CHECK(replace_escapes("%r%r", sys_info) == rank + rank);
    CHECK(replace_escapes("G%rs%rhss", sys_info)
          == "G"+rank+"s"+rank+"hss");
  }

  SECTION("Substitute %s for MPI size")
  {
    auto size = std::to_string(sys_info.mpi_size());
    CHECK(replace_escapes("%s", sys_info) == size);
    CHECK(replace_escapes("I have %s puppies", sys_info)
          == "I have "+size+" puppies");
    CHECK(replace_escapes("%s%s", sys_info) == size + size);
    CHECK(replace_escapes("G%ss%shss", sys_info)
          == "G"+size+"s"+size+"hss");
  }

  SECTION("Substitute %% for a literal %")
  {
    CHECK(replace_escapes("%%", sys_info) == "%");
    CHECK(replace_escapes("110%% is a lie", sys_info)
          == "110% is a lie");
    CHECK(replace_escapes("%%%%", sys_info) == "%%");
    CHECK(replace_escapes("100%%", sys_info) == "100%");
    CHECK(replace_escapes("%%hope", sys_info) == "%hope");
    CHECK(replace_escapes("%%query", sys_info) == "%query");
  }

  SECTION("Substitute %env{<NAME>} for $<NAME> in the current environment")
  {
    auto pumpkin = sys_info.env_variable_value("PUMPKIN");
    auto cranberry = sys_info.env_variable_value("CRANBERRY");
    auto pid = sys_info.pid();
    auto host = sys_info.host_name();
    CHECK(replace_escapes("%env{PUMPKIN}", sys_info) == pumpkin);
    CHECK(replace_escapes("%env{PUMPKIN}%env{PUMPKIN}", sys_info)
          == pumpkin+pumpkin);
    CHECK(replace_escapes("%env{PUMPKIN}%env{CRANBERRY}", sys_info)
          == pumpkin+cranberry);
    CHECK(replace_escapes("%%%env{THIS_IS_UNDEFINED}", sys_info) == "%");
    CHECK(replace_escapes("eat_%env{PUMPKIN}_%h_%p.txt", sys_info)
          == "eat_"+pumpkin+"_"+host+"_"+pid + ".txt");
    CHECK(replace_escapes("%env{THIS_IS_UNDEFINED}", sys_info) == "");
  }

  SECTION("Bad patterns are rejected")
  {
    CHECK_THROWS_AS(replace_escapes("%env", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%a", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%b", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%c", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%d", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%e", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%f", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%g", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%i", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%j", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%k", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%l", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%m", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%n", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%o", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%q", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%t", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%u", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%v", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%w", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%x", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%y", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%z", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%A", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%B", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%C", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%D", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%E", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%F", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%G", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%H", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%I", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%J", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%K", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%L", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%M", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%N", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%O", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%P", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%Q", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%R", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%S", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%T", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%U", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%V", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%W", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%X", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%Y", sys_info), BadSubstitutionPattern);
    CHECK_THROWS_AS(replace_escapes("%Z", sys_info), BadSubstitutionPattern);
  }
}
