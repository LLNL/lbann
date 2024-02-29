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

#include <lbann/utils/tensor_dims_utils.hpp>

#include <algorithm>
#include <sstream>

template <typename T>
static std::string stringify_vec(std::vector<T> const& vec)
{
  std::ostringstream oss;
  oss << "{";
  for (size_t ii = 0; ii < vec.size(); ++ii)
    oss << (ii == 0 ? " " : ", ") << vec[ii];
  oss << " };";
  return oss.str();
}

using namespace lbann;

TEST_CASE("Copy between RowMajorDims and ColMajorDims", "[permute][dim utils]")
{
  auto const rm_dims = RowMajor(std::vector<int>{4, 5, 6, 7});
  auto const cm_dims = ColMajor(rm_dims);

  CHECK(rm_dims.size() == 4UL);
  CHECK(cm_dims.size() == 4UL);

  CHECK(cm_dims.get() == std::vector<int>{7, 6, 5, 4});
  CHECK(RowMajor(cm_dims).get() == std::vector<int>{4, 5, 6, 7});
}

TEST_CASE("Convert dims between row- and column-major", "[permute][dim utils]")
{
  auto const rm_dims = RowMajor(std::vector<int>{4, 5, 6, 7, 8});
  ColMajorDims<int> cm_dims;

  CHECK(rm_dims.size() == 5UL);
  CHECK(cm_dims.size() == 0UL);

  convert(rm_dims, cm_dims);

  CHECK(cm_dims.size() == 5UL);
  CHECK(cm_dims.get() == std::vector<int>{8, 7, 6, 5, 4});
}

TEMPLATE_TEST_CASE("Checking permutation validity",
                   "[permute][dim utils]",
                   RowMajorPerm,
                   ColMajorPerm)
{
  using PermType = TestType;
  auto valid_perms = PermType({0, 1, 2, 3, 4});
  auto invalid_perms = PermType({1, 2, 3, 4, 4});

  auto& valid = valid_perms.get();
  do {
    INFO(stringify_vec(valid));
    CHECK(is_valid(valid_perms));
  } while (std::next_permutation(begin(valid), end(valid)));

  auto& invalid = invalid_perms.get();
  do {
    INFO(stringify_vec(invalid));
    CHECK_FALSE(is_valid(invalid_perms));
  } while (std::next_permutation(begin(invalid), end(invalid)));
}

// This checks that inv(inv(P)) = P.
TEMPLATE_TEST_CASE("Invert permutation",
                   "[permute][layer][gpu]",
                   RowMajorPerm,
                   ColMajorPerm)
{
  using PermType = TestType;
  SECTION("Third-order")
  {
    PermType perm({0, 1, 2});
    auto& perm_v = perm.get();
    size_t count = 0;
    do {
      INFO(stringify_vec(perm_v));
      auto const iperm = invert(perm);
      auto const iiperm = invert(iperm);
      CHECK(iiperm.get() == perm_v);
      ++count;
    } while (std::next_permutation(begin(perm_v), end(perm_v)));
    CHECK(count == 6);
  }
  SECTION("Fourth-order")
  {
    PermType perm({0, 1, 2, 3});
    auto& perm_v = perm.get();
    size_t count = 0;
    do {
      INFO(stringify_vec(perm_v));
      auto const iperm = invert(perm);
      auto const iiperm = invert(iperm);
      CHECK(iiperm.get() == perm_v);
      ++count;
    } while (std::next_permutation(begin(perm_v), end(perm_v)));
    CHECK(count == 24);
  }
  SECTION("Fifth-order")
  {
    PermType perm({0, 1, 2, 3, 4});
    auto& perm_v = perm.get();
    size_t count = 0;
    do {
      INFO(stringify_vec(perm_v));
      auto const iperm = invert(perm);
      auto const iiperm = invert(iperm);
      CHECK(iiperm.get() == perm_v);
      ++count;
    } while (std::next_permutation(begin(perm_v), end(perm_v)));
    CHECK(count == 120);
  }
}

TEST_CASE("Getting packed strides.", "[permute][dim utils]")
{
  // Currently, only column-major support is included. Row-major is
  // trivial but not needed at this time.

  auto const dims = ColMajor(std::vector<int>{3, 5, 7, 9});
  auto const strides = get_strides(dims);

  CHECK(strides.get() == std::vector<int>{1, 3, 15, 105});
}

TEST_CASE("Permuting dimensions", "[permute][dim utils]")
{
  auto const dims = ColMajor(std::vector<int>{3, 5, 7, 9});
  auto const perm = ColMajorPerm(std::vector<int>{3, 2, 1, 0});
  auto const pdims = permute_dims(dims, perm);
  CHECK(pdims.get() == std::vector<int>{9, 7, 5, 3});
}
