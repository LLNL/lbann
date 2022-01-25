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
#include <lbann/utils/any.hpp>

#include <memory>
#include <numeric>
#include <vector>

namespace
{
struct base { virtual ~base() = default; };
struct derived : base {};
}// namespace <anon>

TEST_CASE ("Testing the type-erased \"any\" class", "[any][utilities]")
{
  SECTION ("Default-constructing an \"any\" object")
  {
    lbann::utils::any empty_any;
    lbann::utils::any* null_any_ptr = nullptr;
    REQUIRE_FALSE(empty_any.has_value());
    REQUIRE(lbann::utils::any_cast<int>(&empty_any) == nullptr);
    REQUIRE(lbann::utils::any_cast<int>(null_any_ptr) == nullptr);
  }

  SECTION ("Storing a double in an \"any\" object")
  {
    lbann::utils::any eight_as_double(8.0);
    REQUIRE(eight_as_double.has_value());
    REQUIRE_NOTHROW(lbann::utils::any_cast<double>(eight_as_double));
    REQUIRE(lbann::utils::any_cast<double>(eight_as_double) == 8.0);
    REQUIRE_THROWS_AS(lbann::utils::any_cast<int>(eight_as_double),
                      lbann::utils::bad_any_cast);

    REQUIRE(eight_as_double.type() == typeid(double));
    REQUIRE_FALSE(eight_as_double.type() == typeid(int));

    eight_as_double.reset();
    REQUIRE(eight_as_double.type() == typeid(void));
    REQUIRE_FALSE(eight_as_double.has_value());
  }

  SECTION ("Storing a vector of ints in an \"any\" object")
  {
    lbann::utils::any int_vec_as_any(std::vector<int>(10));

    REQUIRE(int_vec_as_any.has_value());
    REQUIRE_NOTHROW(lbann::utils::any_cast<std::vector<int>&>(int_vec_as_any));

    auto& vec = lbann::utils::any_cast<std::vector<int>&>(int_vec_as_any);
    std::iota(vec.begin(),vec.end(),0);
    REQUIRE(lbann::utils::any_cast<std::vector<int>&>(int_vec_as_any)[5] == 5);

    REQUIRE_THROWS_AS(lbann::utils::any_cast<std::vector<double>>(int_vec_as_any),
                      lbann::utils::bad_any_cast);

    REQUIRE(int_vec_as_any.type() == typeid(std::vector<int>));
    REQUIRE_FALSE(int_vec_as_any.type() == typeid(int[]));

    int_vec_as_any.reset();
    REQUIRE_FALSE(int_vec_as_any.has_value());
  }

  SECTION ("Storing a derived type as pointer-to-base in \"any\" object")
  {
    lbann::utils::any derived_as_base_any(std::shared_ptr<base>{new derived});

    REQUIRE(derived_as_base_any.has_value());
    REQUIRE_NOTHROW(
      lbann::utils::any_cast<std::shared_ptr<base>&>(derived_as_base_any));

    REQUIRE_THROWS_AS(
      lbann::utils::any_cast<std::shared_ptr<derived>&>(derived_as_base_any),
      lbann::utils::bad_any_cast);

    derived_as_base_any.reset();
    REQUIRE_FALSE(derived_as_base_any.has_value());
  }

  SECTION ("Storing a derived type in \"any\" object")
  {
    lbann::utils::any derived_as_any(std::make_shared<derived>());

    REQUIRE(derived_as_any.has_value());
    REQUIRE_NOTHROW(
      lbann::utils::any_cast<std::shared_ptr<derived>&>(derived_as_any));

    REQUIRE_THROWS_AS(
      lbann::utils::any_cast<std::shared_ptr<base>&>(derived_as_any),
      lbann::utils::bad_any_cast);

    derived_as_any.reset();
    REQUIRE_FALSE(derived_as_any.has_value());
  }

  SECTION ("Storing a \"shared_ptr<derived>\" and change to \"double\"")
  {
    lbann::utils::any my_any(std::make_shared<derived>());

    REQUIRE(my_any.has_value());
    REQUIRE_NOTHROW(
      lbann::utils::any_cast<std::shared_ptr<derived>&>(my_any));

    // Change to double
    REQUIRE(my_any.emplace<double>(10.0) == 10.0);
    REQUIRE(lbann::utils::any_cast<double>(&my_any) != nullptr);
    REQUIRE(
      lbann::utils::any_cast<std::shared_ptr<derived>>(&my_any) == nullptr);
    my_any.reset();
    REQUIRE_FALSE(my_any.has_value());
  }
}
