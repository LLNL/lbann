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
#include <lbann/transforms/sample_normalize.hpp>

TEST_CASE("Testing sample normalize preprocessing", "[preproc]")
{
  lbann::utils::type_erased_matrix mat =
    lbann::utils::type_erased_matrix(lbann::CPUMat());
  El::Identity(mat.template get<lbann::DataType>(), 3, 3);
  El::Scale(2.0, mat.template get<lbann::DataType>());
  std::vector<size_t> dims = {3, 3};
  auto normalizer = lbann::transform::sample_normalize();
  SECTION("applying the normalizer")
  {
    REQUIRE_NOTHROW(normalizer.apply(mat, dims));

    SECTION("normalizing does not change dims")
    {
      REQUIRE(dims[0] == 3);
      REQUIRE(dims[1] == 3);
    }
    SECTION("normalizing does not change matrix type")
    {
      REQUIRE_NOTHROW(mat.template get<lbann::DataType>());
    }
    SECTION("normalizing produces correct values")
    {
      auto& real_mat = mat.template get<lbann::DataType>();
      for (El::Int col = 0; col < 3; ++col) {
        for (El::Int row = 0; row < 3; ++row) {
          if (row == col) {
            REQUIRE(real_mat(row, col) == Approx(1.41421356));
          }
          else {
            REQUIRE(real_mat(row, col) == Approx(-0.70710678));
          }
        }
      }
    }
  }
}
