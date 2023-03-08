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
#include <lbann/transforms/vision/random_affine.hpp>
#include <lbann/utils/random_number_generators.hpp>

// Note: This is *random* so we only do basic checks.
TEST_CASE("Testing random affine preprocessing", "[preproc]")
{
  lbann::utils::type_erased_matrix mat =
    lbann::utils::type_erased_matrix(El::Matrix<uint8_t>());
  // Grab the necessary I/O RNG and lock it
  lbann::locked_io_rng_ref io_rng = lbann::set_io_generators_local_index(0);
  // For simplicity, we'll only use a 3-channel matrix here.
  identity(mat.template get<uint8_t>(), 10, 10, 3);
  std::vector<size_t> dims = {3, 10, 10};

  SECTION("rotation")
  {
    auto affiner = lbann::transform::random_affine(0.0, 90.0, 0, 0, 0, 0, 0, 0);

    SECTION("applying the transform")
    {
      REQUIRE_NOTHROW(affiner.apply(mat, dims));

      SECTION("transform does not change dims")
      {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 10);
        REQUIRE(dims[2] == 10);
      }
      SECTION("transform does not change matrix type")
      {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
    }
  }

  SECTION("translate")
  {
    auto affiner = lbann::transform::random_affine(0, 0, 0.1, 0.1, 0, 0, 0, 0);

    SECTION("applying the transform")
    {
      REQUIRE_NOTHROW(affiner.apply(mat, dims));

      SECTION("transform does not change dims")
      {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 10);
        REQUIRE(dims[2] == 10);
      }
      SECTION("transform does not change matrix type")
      {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
    }
  }

  SECTION("scale")
  {
    auto affiner = lbann::transform::random_affine(0, 0, 0, 0, 0.0, 2.0, 0, 0);

    SECTION("applying the transform")
    {
      REQUIRE_NOTHROW(affiner.apply(mat, dims));

      SECTION("transform does not change dims")
      {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 10);
        REQUIRE(dims[2] == 10);
      }
      SECTION("transform does not change matrix type")
      {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
    }
  }

  SECTION("shear")
  {
    auto affiner = lbann::transform::random_affine(0, 0, 0, 0, 0, 0, 0.0, 45.0);

    SECTION("applying the transform")
    {
      REQUIRE_NOTHROW(affiner.apply(mat, dims));

      SECTION("transform does not change dims")
      {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 10);
        REQUIRE(dims[2] == 10);
      }
      SECTION("transform does not change matrix type")
      {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
    }
  }

  SECTION("all")
  {
    auto affiner =
      lbann::transform::random_affine(0.0, 90.0, 0.1, 0.1, 0.0, 2.0, 0.0, 45.0);

    SECTION("applying the transform")
    {
      REQUIRE_NOTHROW(affiner.apply(mat, dims));

      SECTION("transform does not change dims")
      {
        REQUIRE(dims[0] == 3);
        REQUIRE(dims[1] == 10);
        REQUIRE(dims[2] == 10);
      }
      SECTION("transform does not change matrix type")
      {
        REQUIRE_NOTHROW(mat.template get<uint8_t>());
      }
    }
  }
}
