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

#include <lbann/base.hpp>
#include <lbann/utils/exception.hpp>
#include <lbann/utils/cufft_wrapper.hpp>

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

namespace
{

auto get_input_dims(size_t ndims)
{
  std::vector<int> dims(ndims+1, 16);
  dims.front() = 4;// 4 channels of size 16^ndims
  return dims;
}

auto get_num_samples()
{
  return 7;
}

auto get_ldim_offset()
{
  return 13;// because PRIME!
}

template <typename T>
void make_input_matrix(El::Matrix<El::Complex<T>, El::Device::CPU>& mat,
                       std::vector<int> const& full_dims)
{
  using RealT = T;
  using ComplexT = El::Complex<T>;

  // I don't care to be that accurate, and this doesn't rely on
  // finnicky preprocessor macros, etc.
  RealT const twopi = 4*std::asin(RealT(1));
  RealT const one = RealT(1);

  std::mt19937 gen(13);
  std::uniform_real_distribution<RealT> dis(RealT(1.9), RealT(2.1));

  auto const num_samples = mat.Width();
  auto const sample_size = lbann::get_linear_size(full_dims);
  for (El::Int sample_id=0; sample_id < num_samples; ++sample_id)
  {
    ComplexT* const sample = mat.Buffer() + sample_id * mat.LDim();
    RealT const phase = twopi * RealT(sample_id) / RealT(num_samples);
    for (int ii = 0; ii < sample_size; ++ii)
    {
      RealT const x = RealT(ii) / RealT(sample_size) + phase;
      RealT const A = dis(gen);
      sample[ii] = ComplexT(A * std::sin(x) * std::cos(x),
                            -A * std::sin(one-x) * std::cos(one-x));
    }
  }
}

}// namespace <anon>

TEMPLATE_TEST_CASE("Testing cuFFT wrapper (C2C)",
                   "[.cufft][fft][gpu][cuda][utilities][!nonportable]",
                   float, double)
{
  using RealT = TestType;
  using DataT = El::Complex<RealT>;

  auto const ndims = GENERATE(1, 2);
  auto const use_ldim = GENERATE(false, true);

  lbann::cufft::cuFFTWrapper<DataT> cufft;
  auto dims = get_input_dims(ndims);
  int const num_samples = get_num_samples();

  auto const input_matrix_height = lbann::get_linear_size(dims);
  auto const input_matrix_width = num_samples;
  auto const input_matrix_ldim_offset = (use_ldim ? get_ldim_offset() : 0);
  auto const input_matrix_ldim =
    input_matrix_height + input_matrix_ldim_offset;

  // The two matrices may have different LDim
  auto const output_matrix_height = input_matrix_height;
  auto const output_matrix_width = num_samples;
  auto const output_matrix_ldim_offset = (use_ldim ? get_ldim_offset() : 0);
  auto const output_matrix_ldim =
    output_matrix_height + output_matrix_ldim_offset;

  El::Matrix<DataT, El::Device::GPU>
    input(input_matrix_height, input_matrix_width, input_matrix_ldim),
    input_bwd(input_matrix_height, input_matrix_width);
  El::Matrix<DataT, El::Device::GPU>
    output(output_matrix_height, output_matrix_width, output_matrix_ldim);

  El::Matrix<DataT, El::Device::CPU>
    input_cpu(input_matrix_height, input_matrix_width),
    input_bwd_cpu(input_matrix_height, input_matrix_width);

  // Do the forward/backward setups up front.
  REQUIRE_NOTHROW(
    cufft.setup_forward(input, output, dims));
  REQUIRE_NOTHROW(
    cufft.setup_backward(output, input_bwd, dims));

  // Do some initializations
  make_input_matrix(input_cpu, dims);
  //El::MakeUniform(input_cpu, DataT(0.f), RealT(2.f));

  // Move data to GPU
  El::Copy(input_cpu, input);

  // Compute the forward transformation on GPU
  REQUIRE_NOTHROW(cufft.compute_forward(input, output));

  // Do the backward transformation on GPU
  REQUIRE_NOTHROW(cufft.compute_backward(output, input_bwd));

  // Move data back to the CPU
  El::Copy(input_bwd, input_bwd_cpu);
  El::gpu::SynchronizeDevice();

  // Assert that we've gotten back to the ballpark of the original
  // input, with appropriate scaling (CUFFT transforms are *not*
  // normalized).
  auto const scale_factor
    = RealT(lbann::get_linear_size(dims.size() - 1,
                                   dims.data() + 1));
  CAPTURE(scale_factor);
  for (auto col = decltype(input_matrix_width){0};
       col < input_matrix_width;
       ++col)
  {
    for (auto row = decltype(input_matrix_height){0};
         row < input_matrix_height;
         ++row)
    {
      CAPTURE(row, col);
      auto const& input_ij = input_cpu.CRef(row, col);
      auto const& input_bwd_ij = input_bwd_cpu.CRef(row, col);

      // Ehhhh this is fine for now...
      CHECK(RealPart(input_bwd_ij)
            == Approx(scale_factor*RealPart(input_ij))
            .margin(std::numeric_limits<RealT>::epsilon()*100));
      CHECK(ImagPart(input_bwd_ij)
            == Approx(scale_factor*ImagPart(input_ij))
            .epsilon(std::numeric_limits<RealT>::epsilon()*1000)
            .margin(std::numeric_limits<RealT>::epsilon()*100));
    }
  }
}

TEMPLATE_TEST_CASE("Testing cuFFT wrapper (C2C-InPlace)",
                   "[.cufft][fft][gpu][cuda][utilities][!nonportable]",
                   float, double)
{
  using RealT = TestType;
  using DataT = El::Complex<RealT>;

  int const ndims = GENERATE(1,2);
  bool const use_ldim = GENERATE(false, true);
  CAPTURE(ndims, use_ldim);

  lbann::cufft::cuFFTWrapper<DataT> cufft;
  auto dims = get_input_dims(ndims);
  int const num_samples = get_num_samples();

  auto const matrix_height = lbann::get_linear_size(dims);
  auto const matrix_width = num_samples;
  auto const matrix_ldim_offset = (use_ldim ? get_ldim_offset() : 0);
  auto const matrix_ldim =
    matrix_height + matrix_ldim_offset;

  El::Matrix<DataT, El::Device::GPU>
    mat(matrix_height, matrix_width, matrix_ldim), mat_view;
  El::Matrix<DataT, El::Device::CPU>
    mat_cpu(matrix_height, matrix_width),
    mat_orig_cpu(matrix_height, matrix_width);
  El::View(mat_view, mat);

  // Do the forward/backward setups up front.
  REQUIRE_NOTHROW(
    cufft.setup_forward(mat, dims));
  REQUIRE_NOTHROW(
    cufft.setup_backward(mat, dims));

  // Do some initializations; start on the CPU for simplicity.
  //El::MakeUniform(mat_orig_cpu, DataT(0.f), RealT(2.f));
  make_input_matrix(mat_orig_cpu, dims);

  // Copy data to the GPU
  El::Copy(mat_orig_cpu, mat_view);

  // Compute the forward transformation on the GPU
  REQUIRE_NOTHROW(cufft.compute_forward(mat));

  // Do the backward transformation on the GPU
  REQUIRE_NOTHROW(cufft.compute_backward(mat));

  // Copy data back to the CPU
  El::Copy(mat, mat_cpu);
  El::gpu::SynchronizeDevice();

  // Assert that we've gotten back to the ballpark of the original
  // input, with appropriate scaling (cuFFT transforms are *not*
  // normalized).
  auto const scale_factor =
    RealT(lbann::get_linear_size(dims.size() - 1, dims.data() + 1));
  CAPTURE(scale_factor);
  for (auto col = decltype(matrix_width){0}; col < matrix_width; ++col)
  {
    for (auto row = decltype(matrix_height){0}; row < matrix_height; ++row)
    {
      CAPTURE(row, col);
      auto const& mat_ij = mat_cpu.CRef(row, col);
      auto const& mat_orig_cpu_ij = mat_orig_cpu.CRef(row, col);

      // Ehhhh..........
      CHECK(Approx(RealPart(mat_ij))
            .epsilon(std::numeric_limits<RealT>::epsilon()*1000)
            .margin(std::numeric_limits<RealT>::epsilon()*100)
            == scale_factor*RealPart(mat_orig_cpu_ij));
      CHECK(Approx(ImagPart(mat_ij))
            .epsilon(std::numeric_limits<RealT>::epsilon()*1000)
            .margin(std::numeric_limits<RealT>::epsilon()*100)
            == scale_factor*ImagPart(mat_orig_cpu_ij));
    }
  }
}

static_assert(El::IsStorageType<El::Complex<float>, El::Device::CPU>::value,
              "Complex should be a valid storage type on CPU");
static_assert(El::IsStorageType<El::Complex<float>, El::Device::GPU>::value,
              "Complex should be a valid storage type on GPU");
static_assert(El::IsStorageType<El::Complex<double>, El::Device::CPU>::value,
              "Complex should be a valid storage type on CPU");
static_assert(El::IsStorageType<El::Complex<double>, El::Device::GPU>::value,
              "Complex should be a valid storage type on GPU");
static_assert(El::IsComputeType<El::Complex<float>, El::Device::CPU>::value,
              "Complex should be a valid compute type on CPU");
static_assert(El::IsComputeType<El::Complex<float>, El::Device::GPU>::value,
              "Complex should be a valid compute type on GPU");
static_assert(El::IsComputeType<El::Complex<double>, El::Device::CPU>::value,
              "Complex should be a valid compute type on CPU");
static_assert(El::IsComputeType<El::Complex<double>, El::Device::GPU>::value,
              "Complex should be a valid compute type on GPU");
