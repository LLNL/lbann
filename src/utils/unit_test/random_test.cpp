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
#include "Catch2BasicSupport.hpp"

// File being tested
#include <lbann/utils/random.hpp>

#include <limits>

constexpr size_t num_tests = 1000;

TEST_CASE("Testing random_uniform", "[random][utilities]") {

  SECTION("32-bit Mersenne Twister") {
    std::mt19937 gen;
    SECTION("floats") {
      for (size_t i = 0; i < num_tests; ++i) {
        const float val = lbann::random_uniform<float>(gen);
        REQUIRE(val >= 0.0f);
        REQUIRE(val < 1.0f);
      }
    }
    SECTION("doubles") {
      for (size_t i = 0; i < num_tests; ++i) {
        const double val = lbann::random_uniform<double>(gen);
        REQUIRE(val >= 0.0);
        REQUIRE(val < 1.0);
      }
    }
  }

  SECTION("64-bit Mersenne Twister") {
    std::mt19937_64 gen;
    SECTION("floats") {
      for (size_t i = 0; i < num_tests; ++i) {
        const float val = lbann::random_uniform<float>(gen);
        REQUIRE(val >= 0.0f);
        REQUIRE(val < 1.0f);
      }
    }
    SECTION("doubles") {
      for (size_t i = 0; i < num_tests; ++i) {
        const double val = lbann::random_uniform<double>(gen);
        REQUIRE(val >= 0.0);
        REQUIRE(val < 1.0);
      }
    }
  }

  SECTION("Bounds") {
    SECTION("float") {
      SECTION("Min") {
        auto gen = []() -> uint64_t { return 0ull; };
        const float val = lbann::random_uniform<float>(gen);
        REQUIRE(val == 0.0f);
      }
      SECTION("Max") {
        auto gen = []() -> uint64_t { return -1ull; };
        const float val = lbann::random_uniform<float>(gen);
        constexpr float eps = std::numeric_limits<float>::epsilon();
        REQUIRE(val == 1.0f - eps/2);
      }
    }
    SECTION("double") {
      SECTION("Min") {
        auto gen = []() -> uint64_t { return 0ull; };
        const double val = lbann::random_uniform<double>(gen);
        REQUIRE(val == 0.0);
      }
      SECTION("Max") {
        auto gen = []() -> uint64_t { return -1ull; };
        const double val = lbann::random_uniform<double>(gen);
        constexpr double eps = std::numeric_limits<double>::epsilon();
        REQUIRE(val == 1.0 - eps/2);
      }
    }
  }

}
