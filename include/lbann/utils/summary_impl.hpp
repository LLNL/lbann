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

#ifndef LBANN_SUMMARY_IMPL_HPP_INCLUDED
#define LBANN_SUMMARY_IMPL_HPP_INCLUDED

#include "lbann/utils/summary.hpp"
#include "lbann/utils/profiling.hpp"

namespace lbann {

#ifdef LBANN_HAS_TBINF

template <typename TensorDataType>
inline void
lbann_summary::reduce_mean(const std::string tag,
                           const El::AbstractDistMatrix<TensorDataType>& mat,
                           int step)
{
  using AccumT = BiggerOf<TensorDataType, float>;
  // Local sum
  AccumT sum = 0.0;

  // Check distributed matrix format
  El::DistData mat_format(mat);
  if (mat_format.colDist == El::STAR && mat_format.rowDist == El::STAR) {
    // Compute local sum on master process if matrix is Star,Star
    if (mat.RedundantRank() == 0) {
      sum = local_sum(mat.LockedMatrix());
    }
  }
  else {
    // Compute local sum on all processes if matrix is in MC,MR;
    // Star,VC; or similar format
    // TODO: implement for matrices in Circ,Circ; MC,Star; or similar
    // formats
    sum = local_sum(mat.LockedMatrix());
  }

  // Add local sum to list of pending means
  m_pending_means.emplace_back(tag,
                               step,
                               sum,
                               0.0f,
                               mat.Height() * mat.Width());
}

template <typename TensorDataType>
inline void
lbann_summary::reduce_min(const std::string tag,
                          const El::AbstractDistMatrix<TensorDataType>& mat,
                          int step)
{
  using AccumT = BiggerOf<TensorDataType, float>;
  AccumT mat_local_min = local_min(mat.LockedMatrix());
  m_pending_mins.emplace_back(tag, step, mat_local_min);
}

template <typename TensorDataType>
inline void
lbann_summary::reduce_max(const std::string tag,
                          const El::AbstractDistMatrix<TensorDataType>& mat,
                          int step)
{
  using AccumT = BiggerOf<TensorDataType, float>;
  AccumT mat_local_max = local_max(mat.LockedMatrix());
  m_pending_maxes.emplace_back(tag, step, mat_local_max);
}

template <typename TensorDataType>
inline void
lbann_summary::reduce_stdev(const std::string tag,
                            const El::AbstractDistMatrix<TensorDataType>& mat,
                            int step)
{
  using AccumT = BiggerOf<TensorDataType, float>;
  // Local sum and squared sum
  AccumT sum = 0.0;
  AccumT sqsum = 0.0;

  // Check distributed matrix format
  El::DistData mat_format(mat);
  if (mat_format.colDist == El::STAR && mat_format.rowDist == El::STAR) {
    // Compute local sums on master process if matrix is Star,Star
    if (mat.RedundantRank() == 0) {
      local_sum_sqsum(mat.LockedMatrix(), sum, sqsum);
    }
  }
  else {
    // Compute local sums on all processes if matrix is in MC,MR;
    // Star,VC; or similar format
    // TODO: implement for matrices in Circ,Circ; MC,Star; or similar
    // formats
    local_sum_sqsum(mat.LockedMatrix(), sum, sqsum);
  }

  // Add local sums to list of pending stdevs.
  m_pending_stdevs.emplace_back(tag,
                                step,
                                sum,
                                sqsum,
                                mat.Height() * mat.Width());
}

template <typename TensorDataType>
inline void
lbann_summary::reduce_scalar(const std::string tag, TensorDataType s, int step)
{
  if (mat.RedundantRank() == 0) {
    m_pending_scalars.emplace_back(tag, step, s);
  }
}

template <typename TensorDataType>
inline void lbann_summary::sum_reduce_scalar(const std::string tag,
                                             TensorDataType s,
                                             int step)
{
  m_pending_sum_scalars.emplace_back(tag, step, s);
}

template <typename TensorDataType>
inline void lbann_summary::reduce_scalar_all(const std::string tag,
                                             TensorDataType s,
                                             int step)
{
  m_pending_scalar_alls.emplace_back(tag, step, s);
}

template <typename TensorDataType>
inline void lbann_summary::reduce_histogram(
  const std::string tag,
  const El::AbstractDistMatrix<TensorDataType>& mat,
  int step)
{
  using AccumT = BiggerOf<TensorDataType, float>;
  AccumT mat_local_min = local_min(mat.LockedMatrix());
  AccumT mat_local_max = local_max(mat.LockedMatrix());
  // Local sum and squared sum
  AccumT sum = 0.0;
  AccumT sqsum = 0.0;
  // Check distributed matrix format
  El::DistData mat_format(mat);
  if (mat_format.colDist == El::STAR && mat_format.rowDist == El::STAR) {
    // Compute local sums on master process if matrix is Star,Star
    if (mat.RedundantRank() == 0) {
      local_sum_sqsum(mat.LockedMatrix(), sum, sqsum);
    }
  }
  else {
    // Compute local sums on all processes if matrix is in MC,MR;
    // Star,VC; or similar format
    // TODO: implement for matrices in Circ,Circ; MC,Star; or similar
    // formats
    local_sum_sqsum(mat.LockedMatrix(), sum, sqsum);
  }
  // Compute local buckets.
  std::vector<double> buckets(m_histogram_buckets.size() + 1, 0.0);
  const auto height = mat.LocalHeight();
  const auto width = mat.LocalWidth();
  const auto ldim = mat.LDim();
  const auto* __restrict__ mat_buf = mat.LockedMatrix().LockedBuffer();
  for (auto row = 0; row < height; ++row) {
    for (auto col = 0; col < width; ++col) {
      // Note: This could be optimized; upper_bound takes O(logn) time.
      auto bucket = std::distance(m_histogram_buckets.begin(),
                                  std::upper_bound(m_histogram_buckets.begin(),
                                                   m_histogram_buckets.end(),
                                                   mat_buf[row + col * ldim]));
#ifdef LBANN_DEBUG
      buckets.at(bucket) += 1.0;
#else
      buckets[bucket] += 1.0;
#endif // LBANN_DEBUG
    }
  }
  // Add to list of pending histograms.
  m_pending_histograms.emplace_back(tag,
                                    step,
                                    std::move(buckets),
                                    mat_local_min,
                                    mat_local_max,
                                    mat.Height() * mat.Width(),
                                    sum,
                                    sqsum);
  // TODO: Support histograms on multiple models.
}

template <typename TensorDataType>
inline void
lbann_summary::reduce_2norm(const std::string tag,
                            const El::AbstractDistMatrix<TensorDataType>& mat,
                            int step)
{
  // Using a squared 2-norm so that we can just sum this.
  using AccumT = BiggerOf<TensorDataType, float>;
  AccumT local_norm = local_2norm(mat.LockedMatrix());
  sum_reduce_scalar(tag, local_norm * local_norm, step);
}

template <typename TensorDataType>
inline auto
lbann_summary::local_sum(const El::AbstractMatrix<TensorDataType>& mat) const
  -> BiggerOf<TensorDataType, float>
{
  LBANN_CALIPER_MARK_FUNCTION;
  // Note there are more numerically stable ways to compute a sum.
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  const El::Int ldim = mat.LDim();
  const auto* __restrict__ mat_buf = mat.LockedBuffer();
  using AccumT = BiggerOf<TensorDataType, float>;
  AccumT sum = AccumT(0);
  if (ldim == height) {
    const El::Int size = height * width;
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+ : sum))
    for (El::Int i = 0; i < size; ++i) {
      sum += mat_buf[i];
    }
  }
  else {
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+ : sum) collapse(2))
    for (El::Int row = 0; row < height; ++row) {
      for (El::Int col = 0; col < width; ++col) {
        sum += mat_buf[row + col * ldim];
      }
    }
  }
  return sum;
}

template <typename TensorDataType, typename AccumT>
inline void
lbann_summary::local_sum_sqsum(const El::AbstractMatrix<TensorDataType>& mat,
                               AccumT& sum,
                               AccumT& sqsum) const
{
  LBANN_CALIPER_MARK_FUNCTION;
  // Note there are more numerically stable ways to compute a sum.
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  const El::Int ldim = mat.LDim();
  const auto* __restrict__ mat_buf = mat.LockedBuffer();
  sum = AccumT(0);
  sqsum = AccumT(0);
  if (ldim == height) {
    const El::Int size = height * width;
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+ : sum, sqsum))
    for (El::Int i = 0; i < size; ++i) {
      const DataType val = mat_buf[i];
      sum += val;
      sqsum += val * val;
    }
  }
  else {
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+ : sum, sqsum) collapse(2))
    for (El::Int row = 0; row < height; ++row) {
      for (El::Int col = 0; col < width; ++col) {
        const DataType val = mat_buf[row + col * ldim];
        sum += val;
        sqsum += val * val;
      }
    }
  }
}

template <typename TensorDataType>
inline auto
lbann_summary::local_min(const El::AbstractMatrix<TensorDataType>& mat) const
  -> BiggerOf<TensorDataType, float>
{
  LBANN_CALIPER_MARK_FUNCTION;
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  const El::Int ldim = mat.LDim();
  const auto* __restrict__ mat_buf = mat.LockedBuffer();
  using AccumT = BiggerOf<TensorDataType, float>;
  AccumT min = std::numeric_limits<AccumT>::max();
  if (ldim == height) {
    const El::Int size = height * width;
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(min : min))
    for (El::Int i = 0; i < size; ++i) {
      min = El::Min(min, mat_buf[i]);
    }
  }
  else {
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(min : min) collapse(2))
    for (El::Int row = 0; row < height; ++row) {
      for (El::Int col = 0; col < width; ++col) {
        min = El::Min(min, mat_buf[row + col * ldim]);
      }
    }
  }
  return min;
}

template <typename TensorDataType>
inline auto
lbann_summary::local_max(const El::AbstractMatrix<TensorDataType>& mat) const
  -> BiggerOf<TensorDataType, float>
{
  LBANN_CALIPER_MARK_FUNCTION;
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  const El::Int ldim = mat.LDim();
  const auto* __restrict__ mat_buf = mat.LockedBuffer();
  using AccumT = BiggerOf<TensorDataType, float>;
  AccumT max = std::numeric_limits<AccumT>::min();
  if (ldim == height) {
    const El::Int size = height * width;
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(max : max))
    for (El::Int i = 0; i < size; ++i) {
      max = El::Max(max, mat_buf[i]);
    }
  }
  else {
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(max : max) collapse(2))
    for (El::Int row = 0; row < height; ++row) {
      for (El::Int col = 0; col < width; ++col) {
        max = El::Max(max, mat_buf[row + col * ldim]);
      }
    }
  }
  return max;
}

template <typename TensorDataType>
inline auto
lbann_summary::local_2norm(const El::AbstractMatrix<TensorDataType>& mat) const
  -> BiggerOf<TensorDataType, float>
{
  LBANN_CALIPER_MARK_FUNCTION;
  // Note there are more numerically stable ways to compute this.
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  const El::Int ldim = mat.LDim();
  const auto* __restrict__ mat_buf = mat.LockedBuffer();
  using AccumT = BiggerOf<TensorDataType, float>;
  AccumT norm = AccumT(0);
  if (ldim == height) {
    const El::Int size = height * width;
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+ : norm))
    for (El::Int i = 0; i < size; ++i) {
      norm += mat_buf[i] * mat_buf[i];
    }
  }
  else {
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+ : norm) collapse(2))
    for (El::Int row = 0; row < height; ++row) {
      for (El::Int col = 0; col < width; ++col) {
        norm += mat_buf[row + col * ldim] * mat_buf[row + col * ldim];
      }
    }
  }
  return El::Sqrt(norm);
}

#endif // LBANN_HAS_TBINF

} // namespace lbann

#endif // LBANN_SUMMARY_IMPL_HPP_INCLUDED
