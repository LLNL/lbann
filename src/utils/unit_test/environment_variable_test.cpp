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

#include "lbann/utils/environment_variable.hpp"

#include "stubs/preset_env_accessor.hpp"

using namespace lbann::utils;

TEST_CASE("Environment variable wrapper", "[utilities][parser]")
{
  using TestENV = EnvVariable<stubs::PresetEnvAccessor>;

  SECTION("A floating point variable")
  {
    TestENV apple("APPLE");

    CHECK(apple.exists());
    CHECK(apple.name() == "APPLE");
    CHECK(apple.raw_value() == "3.14");

    // This class is (purposefully) not as rigorously typed as, say,
    // the type-erased "any". Since conversion is done on-the-fly from
    // a string, there's less need for strong typing.
    CHECK(apple.value<float>() == 3.14f);
    CHECK(apple.value<double>() == 3.14);
    CHECK(apple.value<int>() == 3);

    // Environment variables should always be convertible to strings
    CHECK(apple.value<std::string>() == apple.raw_value());
  }

  SECTION("An integer variable")
  {
    TestENV scoops("ICE_CREAM_SCOOPS");

    CHECK(scoops.exists());
    CHECK(scoops.name() == "ICE_CREAM_SCOOPS");
    CHECK(scoops.raw_value() == "3");

    CHECK(scoops.value<float>() == 3.f);
    CHECK(scoops.value<double>() == 3.);
    CHECK(scoops.value<int>() == 3);
    CHECK(scoops.value<std::string>() == scoops.raw_value());
  }

  SECTION("A string variable")
  {
    TestENV pizza("PIZZA");
    CHECK(pizza.exists());
    CHECK(pizza.name() == "PIZZA");
    CHECK(pizza.raw_value() == "pepperoni");
    CHECK(pizza.value<std::string>() == pizza.raw_value());

    CHECK_THROWS_AS(pizza.value<float>() == 123.f, std::invalid_argument);
    CHECK_THROWS_AS(pizza.value<double>() == 321., std::invalid_argument);
    CHECK_THROWS_AS(pizza.value<int>() == 42, std::invalid_argument);
  }

  SECTION("A variable that doesn't exist")
  {
    TestENV bad("DOESNT_EXIST");

    CHECK_FALSE(bad.exists());

    CHECK(bad.name() == "DOESNT_EXIST");
    CHECK(bad.raw_value() == "");
    CHECK(bad.value<std::string>() == bad.raw_value());

    CHECK_THROWS_AS(bad.value<float>() == 123.f, std::invalid_argument);
    CHECK_THROWS_AS(bad.value<double>() == 321., std::invalid_argument);
    CHECK_THROWS_AS(bad.value<int>() == 42, std::invalid_argument);
  }
}
