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

// File being tested
#include <lbann/transforms/sample_normalize.hpp>
#include <lbann/transforms/scale.hpp>
#include <lbann/transforms/transform_pipeline.hpp>
#include <lbann/utils/memory.hpp>

TEST_CASE("Testing transform pipeline", "[preproc]")
{
  lbann::transform::transform_pipeline p;
  p.add_transform(std::make_unique<lbann::transform::scale>(2.0f));
  p.add_transform(std::make_unique<lbann::transform::sample_normalize>());
  lbann::CPUMat mat;
  El::Identity(mat, 3, 3);
  std::vector<size_t> dims = {3, 3};

  SECTION("applying the pipeline")
  {
    REQUIRE_NOTHROW(p.apply(mat, dims));

    SECTION("pipeline does not change dims")
    {
      REQUIRE(dims[0] == 3);
      REQUIRE(dims[1] == 3);
    }

    SECTION("pipeline produces correct values")
    {
      for (El::Int col = 0; col < 3; ++col) {
        for (El::Int row = 0; row < 3; ++row) {
          if (row == col) {
            REQUIRE(mat(row, col) == Approx(1.41421356));
          }
          else {
            REQUIRE(mat(row, col) == Approx(-0.70710678));
          }
        }
      }
    }
  }
}
