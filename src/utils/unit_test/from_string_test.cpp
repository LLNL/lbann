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

// MUST include this
#include <catch2/catch.hpp>

// File being tested
#include <lbann/utils/from_string.hpp>

namespace
{

template <typename T> T PositiveAnswer() noexcept;
template <typename T> T NegativeAnswer() noexcept;

template <> int PositiveAnswer() noexcept { return 123; }
template <> int NegativeAnswer() noexcept { return -456; }
template <> long PositiveAnswer() noexcept { return 123L; }
template <> long NegativeAnswer() noexcept { return -456L; }
template <> long long PositiveAnswer() noexcept { return 123LL; }
template <> long long NegativeAnswer() noexcept { return -456LL; }

template <> unsigned long PositiveAnswer() noexcept
{
  return 9876543210UL;
}
template <> unsigned long NegativeAnswer() noexcept
{
  return static_cast<unsigned long>(-1);
}
template <> unsigned long long PositiveAnswer() noexcept
{
  return 9876543210ULL;
}
template <> unsigned long long NegativeAnswer() noexcept
{
  return static_cast<unsigned long long>(-1);
}

template <> float PositiveAnswer() noexcept { return 9.87f; }
template <> float NegativeAnswer() noexcept { return -6.54f; }
template <> double PositiveAnswer() noexcept { return 9.87; }
template <> double NegativeAnswer() noexcept { return -6.54; }
template <> long double PositiveAnswer() noexcept { return 9.87l; }
template <> long double NegativeAnswer() noexcept { return -6.54l; }

}// namespace <anon>

using lbann::utils::from_string;

TEST_CASE("From string corner cases","[utilities][string]")
{
  SECTION("Boolean strings")
  {
    CHECK(from_string<bool>("true"));
    CHECK(from_string<bool>("TRUE"));
    CHECK(from_string<bool>("tRuE"));
    CHECK(from_string<bool>("TrUe"));
    CHECK(from_string<bool>("1"));
    CHECK(from_string<bool>("431"));
    CHECK(from_string<bool>("3.14"));

    CHECK_FALSE(from_string<bool>("false"));
    CHECK_FALSE(from_string<bool>("FALSE"));
    CHECK_FALSE(from_string<bool>("FaLsE"));
    CHECK_FALSE(from_string<bool>("0"));
    CHECK_FALSE(from_string<bool>("0.0"));

    // FIXME: This should be true:
    //CHECK(from_string<bool>("0.2"));

    CHECK_THROWS_AS(from_string<bool>("not a bool"), std::invalid_argument);
  }

  SECTION("From lvalue string to string")
  {
    std::string input("I am a string");
    REQUIRE(from_string<std::string>(input) == input);
    REQUIRE(from_string<std::string>(input) == "I am a string");
  }

  SECTION("From rvalue string to string")
  {
    REQUIRE(from_string("I'm a string") == "I'm a string");
  }

  SECTION("Exceptional cases")
  {
    REQUIRE_THROWS_AS(from_string<int>("9876543210"), std::out_of_range);
  }
}

TEMPLATE_TEST_CASE("From string to floating point type",
                   "[utilities][string]",
                   float, double, long double)
{
  REQUIRE_THROWS_AS(from_string<TestType>("pineapple"), std::invalid_argument);
  REQUIRE(from_string<TestType>("9.87") == PositiveAnswer<TestType>());
  REQUIRE(from_string<TestType>("-6.54") == NegativeAnswer<TestType>());
}

TEMPLATE_TEST_CASE("From string to signed integer type",
                   "[utilities][string]",
                   int, long, long long)
{
  REQUIRE_THROWS_AS(from_string<TestType>("pineapple"), std::invalid_argument);
  REQUIRE(from_string<TestType>("123") == PositiveAnswer<TestType>());
  REQUIRE(from_string<TestType>("-456") == NegativeAnswer<TestType>());
}

TEMPLATE_TEST_CASE("From string to unsigned integer type",
                   "[utilities][string]",
                   unsigned long, unsigned long long)
{
  REQUIRE_THROWS_AS(from_string<TestType>("pineapple"), std::invalid_argument);
  REQUIRE(from_string<TestType>("9876543210") == PositiveAnswer<TestType>());
  REQUIRE(from_string<TestType>("-1") == NegativeAnswer<TestType>());
}
