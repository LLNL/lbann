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

#include <lbann/base.hpp>
#include <lbann/utils/exception.hpp>
#include <lbann/utils/fftw_wrapper.hpp>

#include <complex>
#include <iostream>
#include <vector>

#include <fftw3.h>

// These
namespace {
auto get_input_dims(size_t ndims)
{
  std::vector<int> dims(ndims + 1, 16);
  dims.front() = 4; // 4 channels of size 16^ndims
  return dims;
}

auto get_num_samples() { return 7; }

auto get_ldim_offset()
{
  return 13; // because PRIME!
}

// The real-to-complex case reveals a certain symmetry in the DFT, but
// it's an odd symmetry. The gist of it is:
//
//   A(i,j,k) = Conj(A(ni - i, nj - j, nk - k))
//
// where the indices are periodic. I've written it for 3d, but the
// pattern is the same for arbitrary dimensions. The next few
// functions check that symmetry and print out some info to stdout
// when things go haywire.
//
// LBANN does not use the R2C case, and therefore these are left here
// as useful information for the future, should we ever revisit the
// R2C case.
template <typename T>
bool assert_r2c_symmetry_1d(El::Matrix<T, El::Device::CPU> const& full_output,
                            std::vector<int> const& full_dims)
{
  if (full_dims.size() != 2UL)
    LBANN_ERROR("Only valid for 1-D feature maps.");
  auto const num_feat_maps = full_dims[0];
  auto const num_entries = full_dims[1];
  auto const num_samples = full_output.Width();

  auto const r2c_dims = lbann::fft::get_r2c_output_dims(full_dims);
  auto const num_entries_r2c = r2c_dims[1];

  bool all_good = true;
  for (El::Int sample = 0; sample < num_samples; ++sample) {
    for (int feat_map = 0; feat_map < num_feat_maps; ++feat_map) {
      T const* feat_map_mat = full_output.LockedBuffer() +
                              sample * full_output.LDim() +
                              feat_map * num_entries;
      for (int ent = num_entries_r2c; ent < num_entries; ++ent) {
        auto const conj_ent = (ent == 0 ? 0 : num_entries - ent);
        auto const val = feat_map_mat[ent];
        auto const conj_val = feat_map_mat[conj_ent];
        if (val != El::Conj(conj_val)) {
          std::cout << "Error at (S,F,E,E') = (" << sample << "," << feat_map
                    << "," << ent << "," << conj_ent << "): "
                    << "val = " << val << "; "
                    << "conj_val = " << conj_val << "\n";
          all_good = false;
        }
      }
    }
  }
  return all_good;
}

template <typename T>
bool assert_r2c_symmetry_2d(El::Matrix<T, El::Device::CPU> const& full_output,
                            std::vector<int> const& full_dims)
{
  if (full_dims.size() != 3UL)
    LBANN_ERROR("Only valid for 2-D feature maps.");
  auto const num_feat_maps = full_dims[0];
  auto const num_rows = full_dims[1];
  auto const num_cols = full_dims[2];
  auto const num_samples = full_output.Width();

  auto const r2c_dims = lbann::fft::get_r2c_output_dims(full_dims);
  auto const num_cols_r2c = r2c_dims[2];

  bool all_good = true;
  for (El::Int sample = 0; sample < num_samples; ++sample)
    for (int feat_map = 0; feat_map < num_feat_maps; ++feat_map) {
      T const* feat_map_mat = full_output.LockedBuffer() +
                              sample * full_output.LDim() +
                              feat_map * num_rows * num_cols;
      for (int row = 0; row < num_rows; ++row)
        for (int col = num_cols_r2c; col < num_cols; ++col) {
          auto const conj_row = (row == 0 ? 0 : num_rows - row);
          auto const conj_col = (col == 0 ? 0 : num_cols - col);
          auto const idx = col + row * num_cols;
          auto const conj_idx = conj_col + conj_row * num_cols;
          auto const val = feat_map_mat[idx];
          auto const conj_val = feat_map_mat[conj_idx];

          if ((Approx(El::RealPart(val)) != El::RealPart(conj_val)) ||
              (Approx(El::ImagPart(val)) != -El::ImagPart(conj_val))) {
            std::cout << "Error at (S,F,R,C,R',C') = (" << sample << ","
                      << feat_map << "," << row << "," << col << "," << conj_row
                      << "," << conj_col << "): "
                      << "val = " << val << "; "
                      << "conj_val = " << conj_val << "\n";
            all_good = false;
          }
        }
    }
  return all_good;
}

template <typename T>
bool assert_r2c_symmetry(El::Matrix<T, El::Device::CPU> const& full_output,
                         std::vector<int> const& full_dims)
{
  switch (full_dims.size()) {
  case 0:
  case 1:
    LBANN_ERROR("Invalid dimension size. Remember: "
                "The first entry in the dimension array MUST "
                "be the number of feature maps.");
    break;
  case 2:
    return assert_r2c_symmetry_1d(full_output, full_dims);
    break;
  case 3:
    return assert_r2c_symmetry_2d(full_output, full_dims);
    break;
  default:
    LBANN_ERROR("LBANN currently only supports 1D and 2D DFT algorithms. "
                "Please open an issue on GitHub describing the use-case "
                "for higher-dimensional DFTs.");
  }
  return false;
}
} // namespace

// This is an early test that I wrote to consider the possibility of
// doing a Real-to-Complex (forward) DFT. I think there are some
// useful bits in here, namely understanding the conjugate symmetry
// and the strange format that's used to store it. Thus, even though
// this is not the case that's considered in LBANN, I have left it
// here in case we ever go in this direction.
TEMPLATE_TEST_CASE("Testing FFTW wrapper (R2C)",
                   "[fft][fftw][utilities]",
                   float,
                   double)
{
  using DataT = TestType;

  auto const ndims = GENERATE(1, 2);
  auto const use_ldim = GENERATE(false, true);

  lbann::fftw::FFTWWrapper<DataT> fftw;
  auto input_dims = get_input_dims(ndims);
  auto output_dims = lbann::fft::get_r2c_output_dims(input_dims);
  int const num_samples = get_num_samples();

  auto const input_matrix_height = lbann::get_linear_size(input_dims);
  auto const input_matrix_width = num_samples;
  auto const input_matrix_ldim_offset = (use_ldim ? get_ldim_offset() : 0);
  auto const input_matrix_ldim = input_matrix_height + input_matrix_ldim_offset;

  auto const output_matrix_height = lbann::get_linear_size(output_dims);
  auto const output_matrix_width = num_samples;
  auto const output_matrix_ldim_offset = (use_ldim ? get_ldim_offset() : 0);
  auto const output_matrix_ldim =
    output_matrix_height + output_matrix_ldim_offset;

  El::Matrix<DataT, El::Device::CPU> input(input_matrix_height,
                                           input_matrix_width,
                                           input_matrix_ldim),
    input_bwd(input_matrix_height, input_matrix_width);
  El::Matrix<El::Complex<DataT>, El::Device::CPU> r2c_output(
    output_matrix_height,
    output_matrix_width,
    output_matrix_ldim);
  El::Matrix<DataT, El::Device::CPU> full_output(input_matrix_height,
                                                 input_matrix_width);

  // Do the forward/backward setups up front.
  REQUIRE_NOTHROW(fftw.setup_forward(input, r2c_output, input_dims));
  REQUIRE_NOTHROW(fftw.setup_backward(r2c_output, input_bwd, input_dims));

  // Do some initializations
  El::MakeUniform(input, DataT(-1.f), DataT(1.f));
  El::Fill(r2c_output, El::Complex<DataT>(10.19, 11.23));
  El::Fill(full_output, -DataT(4.13));

  // Compute the forward transformation
  REQUIRE_NOTHROW(fftw.compute_forward(input, r2c_output));

  // Copy into the full format needed for LBANN
  REQUIRE_NOTHROW(lbann::fft::r2c_to_full(r2c_output, full_output, input_dims));

  // Verify the results
  REQUIRE(assert_r2c_symmetry(full_output, input_dims));

  // Do the backward transformation
  El::Zero(input_bwd);
  REQUIRE_NOTHROW(fftw.compute_backward(r2c_output, input_bwd));

  // Assert that we've gotten back to the ballpark of the original
  // input, with appropriate scaling (FFTW transforms are *not*
  // normalized).
  auto const scale_factor =
    DataT(lbann::get_linear_size(input_dims.size() - 1, input_dims.data() + 1));
  for (auto col = decltype(input_matrix_width){0}; col < input_matrix_width;
       ++col) {
    for (auto row = decltype(input_matrix_height){0}; row < input_matrix_height;
         ++row) {
      CAPTURE(row, col);
      auto const& input_ij = input.CRef(row, col);
      auto const& input_bwd_ij = input_bwd.CRef(row, col);

      // Ehhhh this is fine for now...
      CHECK(input_bwd_ij == Approx(scale_factor * input_ij).epsilon(0.05));
    }
  }
}

TEMPLATE_TEST_CASE("Testing FFTW wrapper (C2C)",
                   "[fft][fftw][utilities]",
                   float,
                   double)
{
  using RealT = TestType;
  using DataT = El::Complex<RealT>;

  auto const ndims = GENERATE(1, 2);
  auto const use_ldim = GENERATE(false, true);

  lbann::fftw::FFTWWrapper<DataT> fftw;
  auto dims = get_input_dims(ndims);
  int const num_samples = get_num_samples();

  auto const input_matrix_height = lbann::get_linear_size(dims);
  auto const input_matrix_width = num_samples;
  auto const input_matrix_ldim_offset = (use_ldim ? get_ldim_offset() : 0);
  auto const input_matrix_ldim = input_matrix_height + input_matrix_ldim_offset;

  // The two matrices may have different LDim
  auto const output_matrix_height = input_matrix_height;
  auto const output_matrix_width = num_samples;
  auto const output_matrix_ldim_offset = (use_ldim ? get_ldim_offset() : 0);
  auto const output_matrix_ldim =
    output_matrix_height + output_matrix_ldim_offset;

  El::Matrix<DataT, El::Device::CPU> input(input_matrix_height,
                                           input_matrix_width,
                                           input_matrix_ldim),
    input_bwd(input_matrix_height, input_matrix_width);
  El::Matrix<DataT, El::Device::CPU> output(output_matrix_height,
                                            output_matrix_width,
                                            output_matrix_ldim);

  // Do the forward/backward setups up front.
  REQUIRE_NOTHROW(fftw.setup_forward(input, output, dims));
  REQUIRE_NOTHROW(fftw.setup_backward(output, input_bwd, dims));

  // Do some initializations
  El::MakeUniform(input, DataT(0.f), RealT(2.f));
  El::Zero(input_bwd);
  El::Fill(output, -DataT(4.13));

  // Compute the forward transformation
  REQUIRE_NOTHROW(fftw.compute_forward(input, output));

  // Do the backward transformation
  REQUIRE_NOTHROW(fftw.compute_backward(output, input_bwd));

  // Assert that we've gotten back to the ballpark of the original
  // input, with appropriate scaling (FFTW transforms are *not*
  // normalized).
  auto const scale_factor =
    RealT(lbann::get_linear_size(dims.size() - 1, dims.data() + 1));
  for (auto col = decltype(input_matrix_width){0}; col < input_matrix_width;
       ++col) {
    for (auto row = decltype(input_matrix_height){0}; row < input_matrix_height;
         ++row) {
      CAPTURE(row, col);
      auto const& input_ij = input.CRef(row, col);
      auto const& input_bwd_ij = input_bwd.CRef(row, col);

      // Ehhhh this is fine for now...
      CHECK(RealPart(input_bwd_ij) ==
            Approx(scale_factor * RealPart(input_ij)).epsilon(0.05));
      CHECK(ImagPart(input_bwd_ij) ==
            Approx(scale_factor * ImagPart(input_ij)).epsilon(0.05));
    }
  }
}

TEMPLATE_TEST_CASE("Testing FFTW wrapper (C2C-InPlace)",
                   "[fft][fftw][utilities]",
                   float,
                   double)
{
  using RealT = TestType;
  using DataT = El::Complex<RealT>;

  int const ndims = GENERATE(1, 2);
  bool const use_ldim = GENERATE(false, true);

  lbann::fftw::FFTWWrapper<DataT> fftw;
  auto dims = get_input_dims(ndims);
  int const num_samples = get_num_samples();

  auto const matrix_height = lbann::get_linear_size(dims);
  auto const matrix_width = num_samples;
  auto const matrix_ldim_offset = (use_ldim ? get_ldim_offset() : 0);
  auto const matrix_ldim = matrix_height + matrix_ldim_offset;

  El::Matrix<DataT, El::Device::CPU> mat(matrix_height,
                                         matrix_width,
                                         matrix_ldim),
    mat_orig(matrix_height, matrix_width);

  // Do the forward/backward setups up front.
  REQUIRE_NOTHROW(fftw.setup_forward(mat, dims));
  REQUIRE_NOTHROW(fftw.setup_backward(mat, dims));

  // Do some initializations
  El::MakeUniform(mat, DataT(0.f), RealT(2.f));
  El::Copy(mat, mat_orig);

  // Compute the forward transformation
  REQUIRE_NOTHROW(fftw.compute_forward(mat));

  // Do the backward transformation
  REQUIRE_NOTHROW(fftw.compute_backward(mat));

  // Assert that we've gotten back to the ballpark of the original
  // input, with appropriate scaling (FFTW transforms are *not*
  // normalized).
  auto const scale_factor =
    RealT(lbann::get_linear_size(dims.size() - 1, dims.data() + 1));
  for (auto col = decltype(matrix_width){0}; col < matrix_width; ++col) {
    for (auto row = decltype(matrix_height){0}; row < matrix_height; ++row) {
      CAPTURE(row, col);
      auto const& mat_ij = mat.CRef(row, col);
      auto const& mat_orig_ij = mat_orig.CRef(row, col);

      // Ehhhh this is fine for now...
      CHECK(Approx(RealPart(mat_ij)).epsilon(0.05) ==
            scale_factor * RealPart(mat_orig_ij));
      CHECK(Approx(ImagPart(mat_ij)).epsilon(0.05) ==
            scale_factor * ImagPart(mat_orig_ij));
    }
  }
}
