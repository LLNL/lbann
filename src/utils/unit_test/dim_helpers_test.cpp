////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include <lbann/utils/dim_helpers.hpp>

using namespace lbann;
TEST_CASE("Computing linear sizes", "[dims][utilities]")
{
  SECTION("Empty dims gives 0 size.")
  {
    std::vector<int> dims;
    CHECK(get_linear_size(dims) == 0);
  }

  SECTION("Higher dims")
  {
    std::vector<int> dims1 = {32}, dims2 = {3, 4}, dims3 = {5, 4, 6};
    CHECK(get_linear_size(dims1) == 32);
    CHECK(get_linear_size(dims2) == 12);
    CHECK(get_linear_size(dims3) == 120);
  }

  SECTION("Zero-sized dimension doesn't throw; returns zero.")
  {
    std::vector<int> dims = {5, 0, 4};
    CHECK_NOTHROW(get_linear_size(dims));
    CHECK(get_linear_size(dims) == 0);
  }
}

TEST_CASE("Computing packed strides", "[dims][utilities]")
{
  SECTION("Empty dims, empty strides")
  {
    std::vector<int> dims;
    auto strides = get_packed_strides(dims);
    CHECK(strides.size() == 0UL);
  }

  SECTION("Single dimension, single stride of 1.")
  {
    std::vector<int> dims = {32};
    auto strides = get_packed_strides(dims);
    CHECK(strides.size() == 1UL);
    CHECK(strides.front() == 1);
  }

  SECTION("Two dimensions")
  {
    std::vector<int> dims = {4, 8}, correct = {8, 1};
    auto strides = get_packed_strides(dims);
    CHECK(strides.size() == dims.size());
    CHECK(strides == correct);
  }

  SECTION("Higher dimensions")
  {
    std::vector<int> dims = {4, 3, 5, 2}, correct = {30, 10, 2, 1};
    auto strides = get_packed_strides(dims);
    CHECK(strides.size() == dims.size());
    CHECK(strides == correct);
  }

  SECTION("Zero-sized dimension throws")
  {
    std::vector<int> dims = {3, 2, 0, 7};
    CHECK_THROWS(get_packed_strides(dims));
  }
}

TEST_CASE("Dimension splicing", "[utilities][tensor]")
{
  CHECK(splice_dims() == std::vector<size_t>{});
  CHECK(splice_dims(1, 2, 3) == std::vector<size_t>{1, 2, 3});
  CHECK(splice_dims(1, std::vector<int>{2, 3}) == std::vector<size_t>{1, 2, 3});
  CHECK(splice_dims(std::vector<int>{1, 2}, 3) == std::vector<size_t>{1, 2, 3});
  CHECK(splice_dims(std::vector<int>{1},
                    std::vector<short>{},
                    2,
                    std::vector<char>{3},
                    std::vector<unsigned>{}) == std::vector<size_t>{1, 2, 3});
}
