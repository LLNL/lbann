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

#include "../permute/cutt_permuteimpl.hpp"
#include "lbann/utils/dim_helpers.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

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

TEST_CASE("Computing dims", "[permute][layer][cutt]")
{
  std::vector<int> const lbann_dims = {3, 4, 5, 6};

  SECTION("no permutation (0, 1, 2, 3)")
  {
    std::vector<int> const lbann_perm = {0, 1, 2, 3};

    cuTT_PermuteImpl permuter{RowMajorPerm{lbann_perm}};
    permuter.set_dims(RowMajor(lbann_dims));

    auto const input_dims = permuter.input_dims();
    REQUIRE(input_dims.get() == std::vector<int>{6, 5, 4, 3});

    auto const output_dims = permuter.output_dims();
    REQUIRE(output_dims.get() == std::vector<int>{6, 5, 4, 3});
  }

  SECTION("permutation (0, 2, 1, 3)")
  {
    std::vector<int> const lbann_perm = {0, 2, 1, 3};

    cuTT_PermuteImpl permuter{RowMajorPerm{lbann_perm}};
    permuter.set_dims(RowMajor(lbann_dims));

    auto const input_dims = permuter.input_dims();
    REQUIRE(input_dims.get() == std::vector<int>{6, 5, 4, 3});

    auto const output_dims = permuter.output_dims();
    REQUIRE(output_dims.get() == std::vector<int>{6, 4, 5, 3});
  }

  SECTION("permutation (1, 2, 3, 0)")
  {
    std::vector<int> const lbann_perm = {1, 2, 3, 0};

    cuTT_PermuteImpl permuter{RowMajorPerm{lbann_perm}};
    permuter.set_dims(RowMajor(lbann_dims));

    auto const input_dims = permuter.input_dims();
    REQUIRE(input_dims.get() == std::vector<int>{6, 5, 4, 3});

    auto const output_dims = permuter.output_dims();
    REQUIRE(output_dims.get() == std::vector<int>{3, 6, 5, 4});
  }

  SECTION("permutation (1, 0, 3, 2)")
  {
    std::vector<int> const lbann_perm = {1, 0, 3, 2};

    cuTT_PermuteImpl permuter{RowMajorPerm{lbann_perm}};
    permuter.set_dims(RowMajor(lbann_dims));

    auto const input_dims = permuter.input_dims();
    REQUIRE(input_dims.get() == std::vector<int>{6, 5, 4, 3});

    auto const output_dims = permuter.output_dims();
    REQUIRE(output_dims.get() == std::vector<int>{5, 6, 3, 4});
  }

  SECTION("permutation (2, 3, 1, 0)")
  {
    std::vector<int> const lbann_perm = {2, 3, 1, 0};

    cuTT_PermuteImpl permuter{RowMajorPerm{lbann_perm}};
    permuter.set_dims(RowMajor(lbann_dims));

    auto const input_dims = permuter.input_dims();
    REQUIRE(input_dims.get() == std::vector<int>{6, 5, 4, 3});

    auto const output_dims = permuter.output_dims();
    REQUIRE(output_dims.get() == std::vector<int>{3, 4, 6, 5});
  }
}

// Fills a tensor with 0, 1, 2, 3, ... , (size-1).
// Copied verbatim from cuTENSOR test.
template <typename T>
void iota_fill(El::Matrix<T, El::Device::CPU>& cpu_mat,
               std::vector<int> const& dims,
               std::vector<int> const& perm = {})
{
  auto const mat_height = cpu_mat.Height(), mat_width = cpu_mat.Width(),
             mat_ldim = cpu_mat.LDim();

  if (perm.size() == 0) {
    for (El::Int col = 0; col < mat_width; ++col) {
      auto const offset = col * mat_height;
      for (El::Int row = 0; row < mat_height; ++row)
        cpu_mat.Ref(row, col) = El::To<T>(row + offset);
    }
  }
  else {
    // Setup the data we need.
    size_t const ndims = dims.size();
    std::vector<int> pdims(ndims);
    for (size_t ii = 0; ii < ndims; ++ii)
      pdims[ii] = dims[perm[ii]];

    auto const strides = lbann::get_packed_strides(dims);
    auto const pstrides = lbann::get_packed_strides(pdims);

    auto index_from_pindex =
      [&perm, &strides, &pstrides, &ndims](size_t pindex) {
        size_t index = 0UL;
        for (size_t ii = 0; ii < ndims; ++ii) {
          index += strides[perm[ii]] * (pindex / pstrides[ii]);
          pindex = pindex % pstrides[ii];
        }
        return index;
      };

    for (El::Int col = 0; col < mat_width; ++col) {
      size_t const offset = col * mat_height;
      T* const buffer = cpu_mat.Buffer() + col * mat_ldim;
      for (size_t idx = 0; idx < static_cast<size_t>(mat_height); ++idx) {
        size_t const source_index = index_from_pindex(idx);
        buffer[idx] = static_cast<T>(offset + source_index);
      }
    }
  }
}

template <typename T>
void iota_fill(El::Matrix<T, El::Device::GPU>& mat,
               std::vector<int> const& dims,
               std::vector<int> const& perm = {})
{
  El::Matrix<T, El::Device::CPU> cpu_mat(mat.Height(), mat.Width());
  iota_fill(cpu_mat, dims, perm);
  El::Copy(cpu_mat, mat);
}

TEST_CASE("iota_fill", "[permute][layer][gpu][cutt]")
{
  std::vector<int> const lbann_dims = {3, 4, 5, 6};
  auto const matrix_height = lbann::get_linear_size_as<El::Int>(lbann_dims);
  El::Int const matrix_width = 7;

  El::Matrix<float> mat(matrix_height, matrix_width),
    mat_perm(matrix_height, matrix_width);
  iota_fill(mat, lbann_dims);

  SECTION("Trivial case")
  {
    iota_fill(mat_perm, lbann_dims, std::vector<int>{0, 1, 2, 3});

    for (El::Int cc = 0; cc < matrix_width; ++cc) {
      for (El::Int rr = 0; rr < matrix_height; ++rr) {
        INFO("(i,j) = (" << rr << "," << cc << ")");
        CHECK(mat.CRef(rr, cc) == mat_perm.CRef(rr, cc));
      }
    }
  }
  // SECTION("Nontrivial example")
  // {
  //   // TODO
  // }
}

TEST_CASE("cuTT tensor permutation", "[permute][layer][gpu][cutt]")
{
  std::vector<int> const lbann_dims = {3, 4, 5, 6};
  // strides = { 120, 30, 6, 1 }
  // perm = { 0, 2, 1, 3 }
  // pdims = { 3, 5, 4, 6 }
  // pstrides = { 120, 24, 6, 1 }
  auto const matrix_height = lbann::get_linear_size_as<El::Int>(lbann_dims);
  El::Int const matrix_width = GENERATE(1, 7);
  El::Int const in_matrix_ldim = matrix_height /*+ 4*/;
  El::Int const out_matrix_ldim = matrix_height /*+ 7*/;
  El::Matrix<float, El::Device::GPU> in_mat(matrix_height,
                                            matrix_width,
                                            in_matrix_ldim);
  El::Matrix<float, El::Device::GPU> out_mat(matrix_height,
                                             matrix_width,
                                             out_matrix_ldim);
  El::Matrix<float, El::Device::GPU> inv_out_mat(matrix_height, matrix_width);

  // Setup the matrix values.
  iota_fill(in_mat, lbann_dims);
  El::Matrix<float, El::Device::CPU> const in_mat_cpu{in_mat};

  std::vector<int> lbann_perm = {0, 1, 2, 3};
  do {
    INFO("perm = " << stringify_vec(lbann_perm));

    El::Zero(out_mat);

    // Create the permutation object.
    cuTT_PermuteImpl permuter(RowMajorPerm{lbann_perm});
    permuter.set_dims(RowMajor(lbann_dims));

    // Compute the forward direction.
    permuter.permute(in_mat, out_mat);

    // Compute the inverse direction.
    permuter.inverse_permute(out_mat, inv_out_mat);

    // Check the forward output.
    El::Matrix<float, El::Device::CPU> cpu_out_mat(out_mat);
    El::Matrix<float, El::Device::CPU> cpu_true_mat(cpu_out_mat.Height(),
                                                    cpu_out_mat.Width());
    iota_fill(cpu_true_mat, lbann_dims, lbann_perm);
    for (El::Int col = 0; col < matrix_width; ++col) {
      for (El::Int row = 0; row < matrix_height; ++row) {
        INFO("(i,j)=(" << row << "," << col << ")");
        REQUIRE(cpu_out_mat.CRef(row, col) == cpu_true_mat.CRef(row, col));
      }
    }

    // Check the inverse direction
    El::Copy(inv_out_mat, cpu_out_mat);
    for (El::Int col = 0; col < matrix_width; ++col) {
      for (El::Int row = 0; row < matrix_height; ++row) {
        INFO("(i,j)=(" << row << "," << col << ")");
        REQUIRE(cpu_out_mat.CRef(row, col) == in_mat_cpu.CRef(row, col));
      }
    }

  } while (std::next_permutation(begin(lbann_perm), end(lbann_perm)));
}
