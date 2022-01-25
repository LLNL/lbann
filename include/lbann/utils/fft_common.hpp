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
#ifndef LBANN_UTILS_FFT_COMMON_HPP_
#define LBANN_UTILS_FFT_COMMON_HPP_

#include <lbann/base.hpp>
#include <lbann/utils/dim_helpers.hpp>
#include <lbann/utils/exception.hpp>

namespace lbann
{
// Some metaprogramming. This isn't specific to FFT and should
// probably move elsewhere sometime soon.

template <typename T>
struct ToRealT
{
  using type = T;
};

template <typename T>
struct ToRealT<El::Complex<T>>
{
  using type = T;
};

template <typename T>
using ToReal = typename ToRealT<T>::type;

template <typename T>
struct ToComplexT
{
  using type = El::Complex<T>;
};

template <typename T>
struct ToComplexT<El::Complex<T>>
{
  using type = El::Complex<T>;
};

template <typename T>
using ToComplex = typename ToComplexT<T>::type;

namespace fft
{
template <typename T>
auto get_r2c_output_dims(std::vector<T> const& dims)
{
  std::vector<T> r2c_dims(dims);
  r2c_dims.back() = r2c_dims.back()/2 + 1;
  return r2c_dims;
}

// D1=float, D2=Complex<float>
// input_dims<D1,D2>(full_dims) = full_dims
// output_dims<D1,D2>(full_dims) = r2c_dims
//
// D1=Complex<float>, D2=Complex<float>
// input_dims<D1,D2>(full_dims) = full_dims
// output_dims<D1,D2>(full_dims) = full_dims
//
// D1=Complex<float>, D2=float
// input_dims<D1,D2>(full_dims) = r2c_dims
// output_dims<D1,D2>(full_dims) = full_dims
template <typename InT, typename OutT>
struct DimsHelper;

template <typename InOutT>
struct DimsHelper<InOutT, InOutT>
{
  static auto input_dims(std::vector<int> const& full_dims)
  {
    return full_dims;
  }
  static auto output_dims(std::vector<int> const& full_dims)
  {
    return full_dims;
  }
};

template <typename RealT>
struct DimsHelper<RealT, El::Complex<RealT>>
{
  static auto input_dims(std::vector<int> const& full_dims)
  {
    return full_dims;
  }
  static auto output_dims(std::vector<int> const& full_dims)
  {
    return get_r2c_output_dims(full_dims);
  }
};

template <typename RealT>
struct DimsHelper<El::Complex<RealT>, RealT>
{
  static auto input_dims(std::vector<int> const& full_dims)
  {
    return get_r2c_output_dims(full_dims);
  }
  static auto output_dims(std::vector<int> const& full_dims)
  {
    return full_dims;
  }
};

template <typename InT, typename OutT, typename TransformT>
void r2c_to_full_1d(
  El::Matrix<InT, El::Device::CPU> const& r2c_input,
  El::Matrix<OutT, El::Device::CPU>& full_output,
  std::vector<int> const& full_dims,
  TransformT transform)
{
  if (full_dims.size() != 2UL)
    LBANN_ERROR("Only valid for 1-D feature maps.");

  auto const r2c_dims = lbann::fft::get_r2c_output_dims(full_dims);

  auto const feat_map_ndims = full_dims.size() - 1;
  auto const r2c_feat_map_size =
    lbann::get_linear_size(feat_map_ndims, r2c_dims.data()+1);
  auto const num_samples = r2c_input.Width();
  auto const num_feat_maps = full_dims[0];
  auto const num_entries_full = full_dims[1];
  auto const num_entries_r2c = r2c_dims[1];
  auto const num_diff_entries = num_entries_full - num_entries_r2c;

  // A function to conjugate an element and then apply the transform.
  auto conj_transform = [&t=transform](InT const& x){ return t(El::Conj(x)); };

  // Make sure output is setup.
  full_output.Resize(lbann::get_linear_size(full_dims), num_samples);
  for (int sample_id = 0; sample_id < num_samples; ++sample_id)
  {
    auto output_start = full_output.Buffer()
      + sample_id*full_output.LDim();      // Get to this sample
    for (int feat_map_id = 0; feat_map_id < num_feat_maps; ++feat_map_id)
    {
      // This is the part that gets copied directly.
      auto const r2c_fm_start = r2c_input.LockedBuffer()
        + sample_id*r2c_input.LDim()        // Get to this sample
        + feat_map_id*r2c_feat_map_size;    // Get to this feature map
      // This is the part that gets reverse-copied.
      auto const r2c_conj_fm_start = r2c_input.LockedBuffer()
        + sample_id*r2c_input.LDim()        // Get to this sample
        + feat_map_id*r2c_feat_map_size     // Get to this feature map
        + 1;
      auto const r2c_conj_fm_end = r2c_conj_fm_start + num_diff_entries;

      // Direct copy bit.
      output_start = std::transform(
        r2c_fm_start, r2c_fm_start+num_entries_r2c,
        output_start, transform);

      // Reverse conjugate-and-copy bit.
      auto const r2c_conj_rbegin =
        std::reverse_iterator<InT const*>(r2c_conj_fm_end);
      auto const r2c_conj_rend =
        std::reverse_iterator<InT const*>(r2c_conj_fm_start);
      output_start = std::transform(
        r2c_conj_rbegin, r2c_conj_rend,
        output_start, conj_transform);
    }
  }
}

template <typename InT, typename OutT, typename TransformT>
void r2c_to_full_2d(
  El::Matrix<InT, El::Device::CPU> const& r2c_input,
  El::Matrix<OutT, El::Device::CPU>& full_output,
  std::vector<int> const& full_dims,
  TransformT transform)
{
  if (full_dims.size() != 3UL)
    LBANN_ERROR("Only valid for 2-D feature maps.");

  auto const r2c_dims = lbann::fft::get_r2c_output_dims(full_dims);

  auto const feat_map_ndims = 2;
  auto const r2c_feat_map_size =
    lbann::get_linear_size(feat_map_ndims, r2c_dims.data()+1);
  auto const num_samples = r2c_input.Width();
  auto const num_feat_maps = full_dims[0];
  auto const num_rows = full_dims[1];
  auto const num_cols_full = full_dims[2];
  auto const num_cols_r2c = r2c_dims[2];
  auto const num_diff_cols = num_cols_full - num_cols_r2c;

  // Convenience function:
  auto conj_transform = [&t=transform](InT const& x){ return t(Conj(x)); };

  // Make sure output is setup.
  full_output.Resize(lbann::get_linear_size(full_dims), num_samples);
  for (int sample_id = 0; sample_id < num_samples; ++sample_id)
  {
    // This is the start of the feature map.
    auto output_start = full_output.Buffer()
      + sample_id*full_output.LDim();            // Get to this sample
    for (int feat_map_id = 0; feat_map_id < num_feat_maps; ++feat_map_id)
    {
      for (int row = 0; row < num_rows; ++row)
      {
        auto const conj_row_index = (row == 0 ? 0 : num_rows - row);

        // This is the part that gets copied directly.
        auto const r2c_row_start = r2c_input.LockedBuffer()
          + sample_id*r2c_input.LDim()      // Get to this sample
          + feat_map_id*r2c_feat_map_size   // Get to this feature map
          + row * num_cols_r2c;             // Get to this row
        // This is the part that gets reverse-copied
        auto const r2c_conj_row_start = r2c_input.LockedBuffer()
          + sample_id*r2c_input.LDim()      // Get to this sample
          + feat_map_id*r2c_feat_map_size   // Get to this feature map
          + conj_row_index * num_cols_r2c   // Get to this row
          + 1;                              // Get to the right col
        auto const r2c_conj_row_end = r2c_conj_row_start + num_diff_cols;

        // Directly copy the row
        output_start = std::transform(
          r2c_row_start, r2c_row_start+num_cols_r2c,
          output_start, transform);

        // Reverse copy the conjugated bits
        auto const r2c_conj_rbegin =
          std::reverse_iterator<InT const*>(r2c_conj_row_end);
        auto const r2c_conj_rend =
          std::reverse_iterator<InT const*>(r2c_conj_row_start);
        output_start = std::transform(
          r2c_conj_rbegin, r2c_conj_rend,
          output_start, conj_transform);
      }
    }
  }
}

template <typename InT, typename OutT>
void r2c_to_full(
  El::Matrix<InT, El::Device::CPU> const& r2c_input,
  El::Matrix<OutT, El::Device::CPU>& full_output,
  std::vector<int> const& full_dims)
{
  auto abs_val_func = [](InT const& in) { return std::abs(in); };
  switch (full_dims.size())
  {
  case 0:
  case 1:
    LBANN_ERROR("Invalid dimension size. Remember: "
                "The first entry in the dimension array MUST "
                "be the number of feature maps.");
    break;
  case 2:
    r2c_to_full_1d(r2c_input, full_output, full_dims, abs_val_func);
    break;
  case 3:
    r2c_to_full_2d(r2c_input, full_output, full_dims, abs_val_func);
    break;
  default:
    LBANN_ERROR("LBANN currently only supports 1D and 2D DFT algorithms. "
                "Please open an issue on GitHub describing the use-case "
                "for higher-dimensional DFTs.");
    break;
  }
}

}// namespace fft
}// namespace lbann
#endif // LBANN_UTILS_FFT_COMMON_HPP_
