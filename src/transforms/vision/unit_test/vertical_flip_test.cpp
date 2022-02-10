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
#include <lbann/transforms/vision/vertical_flip.hpp>
#include <lbann/utils/random_number_generators.hpp>
#include "helper.hpp"

TEST_CASE("Testing vertical flip preprocessing", "[preproc]") {
  lbann::utils::type_erased_matrix mat = lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());
  // Grab the necessary I/O RNG and lock it
  lbann::locked_io_rng_ref io_rng = lbann::set_io_generators_local_index(0);

  SECTION("matrix with one channel") {
    zeros(mat.template get<uint8_t>(), 3, 3, 1);
    apply_elementwise(mat.template get<uint8_t>(), 3, 3, 1,
                      [](uint8_t& x, El::Int row, El::Int col, El::Int) {
                        if (row == 0) { x = 1; }
                      });
    std::vector<size_t> dims = {1, 3, 3};
    auto flipper = lbann::transform::vertical_flip(1.0);

    SECTION("applying the flip") {
      REQUIRE_NOTHROW(flipper.apply(mat, dims));

      SECTION("flipping does not change dims") {
        REQUIRE(dims[0] == 1);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("flipping does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
      SECTION("flipping produces correct values") {
        auto& real_mat = mat.template get<uint8_t>();
        apply_elementwise(
          real_mat, 3, 3, 1,
          [](uint8_t& x, El::Int row, El::Int col, El::Int) {
            if (row == 2) {
              REQUIRE(x == 1);
            } else {
              REQUIRE(x == 0);
            }
          });
      }
    }
  }

  SECTION("matrix with three channels") {
    zeros(mat.template get<uint8_t>(), 3, 3, 3);
    apply_elementwise(mat.template get<uint8_t>(), 3, 3, 3,
                      [](uint8_t& x, El::Int row, El::Int col, El::Int) {
                        if (row == 0) { x = 1; }
                      });
    std::vector<size_t> dims = {3, 3, 3};
    auto flipper = lbann::transform::vertical_flip(1.0);

    SECTION("applying the flip") {
      REQUIRE_NOTHROW(flipper.apply(mat, dims));

      SECTION("flipping does not change dims") {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 3);
        REQUIRE(dims[2] == 3);
      }
      SECTION("flipping does not change matrix type") {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
      SECTION("flipping produces correct values") {
        auto& real_mat = mat.template get<uint8_t>();
        apply_elementwise(
          real_mat, 3, 3, 3,
          [](uint8_t& x, El::Int row, El::Int col, El::Int) {
            if (row == 2) {
              REQUIRE(x == 1);
            } else {
              REQUIRE(x == 0);
            }
          });
      }
    }
  }
}
