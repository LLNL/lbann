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
#include "helper.hpp"
#include <lbann/transforms/vision/grayscale.hpp>

TEST_CASE("Testing grayscale preprocessing", "[preproc]")
{
  lbann::utils::type_erased_matrix mat =
    lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());

  SECTION("matrix with one channel")
  {
    identity(mat.template get<uint8_t>(), 3, 3, 1);
    std::vector<size_t> dims = {1, 3, 3};
    auto gs = lbann::transform::grayscale();

    SECTION("applying grayscape")
    {
      REQUIRE_NOTHROW(gs.apply(mat, dims));

      SECTION("grayscale does not change dims")
      {
        REQUIRE(dims[0] == 1);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("grayscale does not change matrix type")
      {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
      SECTION("grayscale does not change values")
      {
        auto& real_mat = mat.template get<uint8_t>();
        apply_elementwise(real_mat,
                          3,
                          3,
                          1,
                          [](uint8_t& x, El::Int row, El::Int col, El::Int) {
                            if (row == col) {
                              REQUIRE(x == 1);
                            }
                            else {
                              REQUIRE(x == 0);
                            }
                          });
      }
    }
  }

  SECTION("matrix with three channels")
  {
    identity(mat.template get<uint8_t>(), 3, 3, 3);
    std::vector<size_t> dims = {3, 3, 3};
    auto gs = lbann::transform::grayscale();

    SECTION("applying grayscale")
    {
      REQUIRE_NOTHROW(gs.apply(mat, dims));

      SECTION("grayscale changes dims correctly")
      {
        REQUIRE(dims[0] == 1);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("grayscale does not change matrix type")
      {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
      SECTION("grayscale produces correct values")
      {
        auto& real_mat = mat.template get<uint8_t>();
        apply_elementwise(real_mat,
                          3,
                          3,
                          1,
                          [](uint8_t& x, El::Int row, El::Int col, El::Int) {
                            if (row == col) {
                              REQUIRE(x == 1);
                            }
                            else {
                              REQUIRE(x == 0);
                            }
                          });
      }
    }
  }
}
