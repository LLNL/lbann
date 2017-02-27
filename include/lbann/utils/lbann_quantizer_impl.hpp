////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC. 
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
//
// lbann_quantizer_impl .hpp - Quantization of matrices
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_QUANTIZER_IMPL_HPP_INCLUDED
#define LBANN_QUANTIZER_IMPL_HPP_INCLUDED

#include <omp.h>

namespace lbann
{

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_threshold_quantize(
  const Mat& mat, std::vector<rowT>& q, Mat& qerror, int proportion) {
  // Ensure types are reasonable.
  static_assert(std::is_integral<colT>::value && std::is_integral<rowT>::value,
                "Types must be integral");
  static_assert(std::is_unsigned<colT>::value && std::is_unsigned<rowT>::value,
                "Types must be unsigned");
  static_assert(sizeof(colT) == 2 || sizeof(colT) == 4 || sizeof(colT) == 8,
                "colT must be 2, 4, or 8 bytes.");
  static_assert(sizeof(rowT) == 2 || sizeof(rowT) == 4 || sizeof(rowT) == 8,
                "rowT must be 2, 4, or 8 bytes.");
  static_assert(sizeof(colT) >= sizeof(DataType),
                "colT must be at least as large as DataType");
  // This uses a header to store all information needed to do unquantization in
  // one spot, which makes unquantization easier to multi-thread. The header has
  // one entry for each column, consisting of the starting offset of the
  // quantized data in the array (including the header) and the two/three
  // reconstruction values. The number of quantized entries is included as a
  // final entry to simplify unquantization.
  const colT width = mat.Width();
  const colT height = mat.Height();
  const colT ldim = mat.LDim();
  const DataType* const __restrict__ mat_buf = mat.LockedBuffer();
  DataType* __restrict__ qerror_buf = qerror.Buffer();
  const colT row_header_factor = sizeof(rowT) == 2 ? 2 : 1;
  const colT header_len = row_header_factor * HEADER_FACTOR * width +
    row_header_factor;
  q.resize(header_len);  // Space for the header.
  // Select the appropriate number of threads.
  const int num_threads = get_adaptive_quantization_threads(width);
  std::vector<std::vector<rowT>> thread_qs(num_threads);
  std::vector<colT> quantized_sums(num_threads, 0);
  std::vector<colT> quantized_counts(num_threads, 0);
  // Compute the thresholds.
  const adaptive_thresholds threshes =
    proportion_threshold(mat, qerror, proportion);
  // This is for accessing q in different ways.
  colT* q_col = (colT*) q.data();
  #pragma omp parallel firstprivate(threshes, height, width, ldim, mat_buf, qerror_buf) num_threads(num_threads)
  {
    const int tid = omp_get_thread_num();
    colT num_quantized = 0;
    std::vector<rowT>& thread_q = thread_qs[tid];
    thread_q.resize(std::max(
      2 * height * width / proportion / num_threads,
      (colT) 4));
    colT size = thread_q.size();
    #pragma omp for schedule(static)
    for (colT col = 0; col < width; ++col) {
      const colT header_loc = HEADER_FACTOR * col;
      q_col[header_loc] = num_quantized;
      const adaptive_reconstructions recons =
        col_reconstruction(mat, qerror, col, threshes);
      // Store the averages for reconstruction.
      q_col[header_loc + 1] = 0;
      memcpy(&q_col[header_loc + 1], &recons.pos_recon, sizeof(recons.pos_recon));
      q_col[header_loc + 2] = 0;
      memcpy(&q_col[header_loc + 2], &recons.neg_recon, sizeof(recons.neg_recon));
#if LBANN_QUANTIZER_TERNARY
      q_col[header_loc + 3] = 0;
      memcpy(&q_col[header_loc + 3], &recons.zero_recon, sizeof(recons.zero_recon));
#endif
      const colT col_offset = col * ldim;
      const DataType* const __restrict__ mat_col = &mat_buf[col_offset];
      DataType* __restrict__ qerror_col = &qerror_buf[col_offset];
      for (rowT row = 0; row < height; ++row) {
        const DataType val = mat_col[row] + qerror_col[row];
        const bool x = val >= threshes.pos_thresh;
        const bool y = val <= threshes.neg_thresh;
        if (__builtin_expect(!!(x || y), 0)) {  // Unlikely.
          if (x) {
            qerror_col[row] = val - recons.pos_recon;
            thread_q[num_quantized++] = (row << 1) | 1;
          } else {
            qerror_col[row] = val - recons.neg_recon;
            thread_q[num_quantized++] = row << 1;
          }
          if (__builtin_expect(!!(num_quantized >= size), 0)) {
            thread_q.resize(2 * size);
            size *= 2;
          }
        } else {
#if LBANN_QUANTIZER_TERNARY
          qerror_col[row] = val - recons.zero_recon;
#else
          qerror_col[row] = val;
#endif
        }
      }
    }
    quantized_counts[tid] = num_quantized;
    #pragma omp barrier
    #pragma omp single
    {
      // Compute the amount to adjust header counts by. This is essentially
      // a shifted prefix-sum.
      for (int t = 1; t < num_threads; ++t) {
        quantized_sums[t] = quantized_sums[t - 1] + quantized_counts[t - 1];
      }
    }
    // Have threads patch up the header counts.
    // Static schedule guarantees threads are assigned the same way.
    #pragma omp for schedule(static)
    for (colT col = 0; col < width; ++col) {
      q_col[HEADER_FACTOR * col] += quantized_sums[tid] + header_len;
    }
  }
  colT total_quantized = std::accumulate(quantized_counts.begin(),
    quantized_counts.end(), 0);
  q.resize(header_len + total_quantized);
  // Only use half the threads here for two reasons:
  // - Diminishing returns on memory bandwidth.
  // - Helps avoid load imbalance.
  const int num_copy_threads =
    get_adaptive_quantization_copy_threads(width);
  #pragma omp parallel for schedule(dynamic, 1) num_threads(num_copy_threads)
  for (unsigned tid = 0; tid < thread_qs.size(); ++tid) {
    std::copy(thread_qs[tid].begin(),
              thread_qs[tid].begin() + quantized_counts[tid],
              q.begin() + quantized_sums[tid] + header_len);
  }
  // Store the final number of entries. Get a new q_col pointer because of the
  // resize.
  q_col = (colT*) q.data();
  q_col[HEADER_FACTOR * width] = (colT) q.size();
  quantized_count = q.size() - header_len;
  adaptive_threshold_bound<colT, rowT>(mat, qerror, q, proportion);
}

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_threshold_quantize(
  const DistMat& mat, std::vector<rowT>& q, Mat& qerror, int proportion) {
  adaptive_threshold_quantize<colT, rowT>(mat.LockedMatrix(), q, qerror, proportion);
}

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_threshold_unquantize(
  const std::vector<rowT>& q, Mat& mat) {
  // Ensure types are reasonable.
  static_assert(std::is_integral<colT>::value && std::is_integral<rowT>::value,
                "Types must be integral");
  static_assert(std::is_unsigned<colT>::value && std::is_unsigned<rowT>::value,
                "Types must be unsigned");
  static_assert(sizeof(colT) == 2 || sizeof(colT) == 4 || sizeof(colT) == 8,
                "colT must be 2, 4, or 8 bytes.");
  static_assert(sizeof(rowT) == 2 || sizeof(rowT) == 4 || sizeof(rowT) == 8,
                "rowT must be 2, 4, or 8 bytes.");
  static_assert(sizeof(colT) >= sizeof(DataType),
                "colT must be at least as large as DataType");
  DataType* __restrict__ buf = mat.Buffer();
  const colT header_len = mat.Width() * HEADER_FACTOR;
  const colT height = mat.Height();
  const colT ldim = mat.LDim();
  const colT* q_col = (const colT*) q.data();
  #pragma omp parallel for schedule(dynamic, 1), firstprivate(header_len, buf)
  for (colT header_loc = 0; header_loc < header_len; header_loc += HEADER_FACTOR) {
    const colT col_offset = (header_loc / HEADER_FACTOR) * ldim;
    // Extract averages.
    DataType pos_recon, neg_recon;
    memcpy(&pos_recon, &q_col[header_loc + 1], sizeof(pos_recon));
    memcpy(&neg_recon, &q_col[header_loc + 2], sizeof(neg_recon));
#if LBANN_QUANTIZER_TERNARY
    DataType zero_recon;
    memcpy(&zero_recon, &q_col[header_loc + 3], sizeof(zero_recon));
    // Fill the column, then update with the other values.
    std::fill_n(&buf[(header_loc / HEADER_FACTOR) * ldim], height, zero_recon);
#endif
    DataType* __restrict__ buf_col = &buf[col_offset];
    const colT chunk_start = q_col[header_loc];
    const colT chunk_end = q_col[header_loc + HEADER_FACTOR] - chunk_start;
    const rowT* const __restrict__ q_ = &(q.data()[chunk_start]);
    for (rowT i = 0; i < chunk_end; ++i) {
      const rowT val = q_[i];
      const rowT row = val >> 1;
      buf_col[row] = val & 0x1 ? pos_recon : neg_recon;
    }
  }
}

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_threshold_unquantize(
  const std::vector<rowT>& q, DistMat& mat) {
  adaptive_threshold_unquantize<colT, rowT>(q, mat.Matrix());
}

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_threshold_unquantize_add(
  const std::vector<rowT>& q, Mat& mat) {
  // Ensure types are reasonable.
  static_assert(std::is_integral<colT>::value && std::is_integral<rowT>::value,
                "Types must be integral");
  static_assert(std::is_unsigned<colT>::value && std::is_unsigned<rowT>::value,
                "Types must be unsigned");
  static_assert(sizeof(colT) == 2 || sizeof(colT) == 4 || sizeof(colT) == 8,
                "colT must be 2, 4, or 8 bytes.");
  static_assert(sizeof(rowT) == 2 || sizeof(rowT) == 4 || sizeof(rowT) == 8,
                "rowT must be 2, 4, or 8 bytes.");
  static_assert(sizeof(colT) >= sizeof(DataType),
                "colT must be at least as large as DataType");
  DataType* __restrict__ buf = mat.Buffer();
  const colT header_len = mat.Width() * HEADER_FACTOR;
  const colT height = mat.Height();
  const colT ldim = mat.LDim();
  const colT* q_col = (const colT*) q.data();
  #pragma omp parallel for schedule(dynamic, 1), firstprivate(header_len, buf)
  for (colT header_loc = 0; header_loc < header_len; header_loc += HEADER_FACTOR) {
    const colT col_offset = (header_loc / HEADER_FACTOR) * ldim;
    // Extract averages.
    DataType pos_recon, neg_recon;
    memcpy(&pos_recon, &q_col[header_loc + 1], sizeof(pos_recon));
    memcpy(&neg_recon, &q_col[header_loc + 2], sizeof(neg_recon));
#if LBANN_QUANTIZER_TERNARY
    DataType zero_recon;
    memcpy(&zero_recon, &q_col[header_loc + 3], sizeof(zero_recon));
    // Add zero_recon to everything and adjust the other means.
    for (rowT row = 0; row < height; ++row) {
      buf[row + col_offset] += zero_recon;
    }
    pos_recon -= zero_recon;
    neg_recon += zero_recon;
#endif
    DataType* __restrict__ buf_col = &buf[col_offset];
    const colT chunk_start = q_col[header_loc];
    const colT chunk_end = q_col[header_loc + HEADER_FACTOR] - chunk_start;
    const rowT* const __restrict__ q_ = &(q.data()[chunk_start]);
    for (rowT i = 0; i < chunk_end; ++i) {
      const rowT val = q_[i];
      const rowT row = val >> 1;
      buf_col[row] += val & 0x1 ? pos_recon : neg_recon;
    }
  }
}

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_threshold_quantize_replace(
  Mat& mat, std::vector<rowT>& q, Mat& qerror, int proportion) {
  // Ensure types are reasonable.
  static_assert(std::is_integral<colT>::value && std::is_integral<rowT>::value,
                "Types must be integral");
  static_assert(std::is_unsigned<colT>::value && std::is_unsigned<rowT>::value,
                "Types must be unsigned");
  static_assert(sizeof(colT) == 2 || sizeof(colT) == 4 || sizeof(colT) == 8,
                "colT must be 2, 4, or 8 bytes.");
  static_assert(sizeof(rowT) == 2 || sizeof(rowT) == 4 || sizeof(rowT) == 8,
                "rowT must be 2, 4, or 8 bytes.");
  static_assert(sizeof(colT) >= sizeof(DataType),
                "colT must be at least as large as DataType");
  const colT width = mat.Width();
  const colT height = mat.Height();
  const colT ldim = mat.LDim();
  DataType* __restrict__ mat_buf = mat.Buffer();
  DataType* __restrict__ qerror_buf = qerror.Buffer();
  const colT row_header_factor = sizeof(rowT) == 2 ? 2 : 1;
  const colT header_len = row_header_factor * HEADER_FACTOR * width +
    row_header_factor;
  q.resize(header_len);  // Space for the header.
  std::vector<std::vector<rowT>> thread_qs(omp_get_max_threads());
  std::vector<colT> quantized_sums(omp_get_max_threads(), 0);
  std::vector<colT> quantized_counts(omp_get_max_threads(), 0);
  // Compute the thresholds.
  const adaptive_thresholds threshes =
    proportion_threshold(mat, qerror, proportion);
  colT* q_col = (colT*) q.data();
  #pragma omp parallel firstprivate(threshes, height, width, ldim, mat_buf, qerror_buf)
  {
    const int tid = omp_get_thread_num();
    colT num_quantized = 0;
    std::vector<rowT>& thread_q = thread_qs[tid];
    thread_q.resize(std::max(
      2 * height * width / proportion / omp_get_max_threads(),
      (colT) 4));
    colT size = thread_q.size();
    #pragma omp for schedule(static)
    for (colT col = 0; col < width; ++col) {
      const colT header_loc = HEADER_FACTOR * col;
      q_col[header_loc] = num_quantized;
      const adaptive_reconstructions recons =
        col_reconstruction(mat, qerror, col, threshes);
      // Store the averages for reconstruction.
      q_col[header_loc + 1] = 0;
      memcpy(&q_col[header_loc + 1], &recons.pos_recon, sizeof(recons.pos_recon));
      q_col[header_loc + 2] = 0;
      memcpy(&q_col[header_loc + 2], &recons.neg_recon, sizeof(recons.neg_recon));
#if LBANN_QUANTIZER_TERNARY
      q_col[header_loc + 3] = 0;
      memcpy(&q_col[header_loc + 3], &recons.zero_recon, sizeof(recons.zero_recon));
#endif
      const colT col_offset = col * ldim;
      DataType* __restrict__ mat_col = &mat_buf[col_offset];
      DataType* __restrict__ qerror_col = &qerror_buf[col_offset];
      for (rowT row = 0; row < height; ++row) {
        const DataType val = mat_col[row] + qerror_col[row];
        const bool x = val >= threshes.pos_thresh;
        const bool y = val <= threshes.neg_thresh;
        if (__builtin_expect(!!(x || y), 0)) {  // Unlikely.
          if (x) {
            qerror_col[row] = val - recons.pos_recon;
            thread_q[num_quantized++] = (row << 1) | 1;
            mat_col[row] = recons.pos_recon;
          } else {
            qerror_col[row] = val - recons.neg_recon;
            thread_q[num_quantized++] = row << 1;
            mat_col[row] = recons.neg_recon;
          }
          if (__builtin_expect(!!(num_quantized >= size), 0)) {
            thread_q.resize(2 * size);
            size *= 2;
          }
        } else {
#if LBANN_QUANTIZER_TERNARY
          qerror_col[row] = val - recons.zero_recon;
          mat_col[row] = recons.zero_recon;
#else
          qerror_col[row] = val;
#endif
        }
      }
    }
    quantized_counts[tid] = num_quantized;
    #pragma omp barrier
    #pragma omp single
    {
      // Compute the amount to adjust header counts by. This is essentially
      // a shifted prefix-sum.
      for (int t = 1; t < omp_get_max_threads(); ++t) {
        quantized_sums[t] = quantized_sums[t - 1] + quantized_counts[t - 1];
      }
    }
    // Have threads patch up the header counts.
    // Static schedule guarantees threads are assigned the same way.
    #pragma omp for schedule(static)
    for (colT col = 0; col < width; ++col) {
      q_col[HEADER_FACTOR * col] += quantized_sums[tid] + header_len;
    }
  }
  colT total_quantized = std::accumulate(quantized_counts.begin(),
    quantized_counts.end(), 0);
  q.resize(header_len + total_quantized);
  // Only use half the threads here for two reasons:
  // - Diminishing returns on memory bandwidth.
  // - Helps avoid load imbalance.
#pragma omp parallel for schedule(dynamic, 1) num_threads(omp_get_max_threads() / 2)
  for (unsigned tid = 0; tid < thread_qs.size(); ++tid) {
    std::copy(thread_qs[tid].begin(),
              thread_qs[tid].begin() + quantized_counts[tid],
              q.begin() + quantized_sums[tid] + header_len);
  }
  // Store the final number of entries. Get a new q_col pointer because of the
  // resize.
  q_col = (colT*) q.data();
  q_col[HEADER_FACTOR * width] = q.size();
  quantized_count = q.size() - header_len;
  adaptive_threshold_bound<colT, rowT>(mat, qerror, q, proportion);
}

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_threshold_bound(
  const Mat& mat, Mat& qerror, std::vector<rowT>& q, int proportion) {
  // Ensure types are reasonable.
  static_assert(std::is_integral<colT>::value && std::is_integral<rowT>::value,
                "Types must be integral");
  static_assert(std::is_unsigned<colT>::value && std::is_unsigned<rowT>::value,
                "Types must be unsigned");
  static_assert(sizeof(colT) == 2 || sizeof(colT) == 4 || sizeof(colT) == 8,
                "colT must be 2, 4, or 8 bytes.");
  static_assert(sizeof(rowT) == 2 || sizeof(rowT) == 4 || sizeof(rowT) == 8,
                "rowT must be 2, 4, or 8 bytes.");
  static_assert(sizeof(colT) >= sizeof(DataType),
                "colT must be at least as large as DataType");
  const DataType* const __restrict__ mat_buf = mat.LockedBuffer();
  DataType* __restrict__ qerror_buf = qerror.Buffer();
  const colT width = mat.Width();
  const colT height = mat.Height();
  const colT ldim = mat.LDim();
  const colT row_header_factor = sizeof(rowT) == 2 ? 2 : 1;
  const colT header_len = row_header_factor * HEADER_FACTOR * width +
    row_header_factor;
  const colT num_quantized = q.size() - header_len;
  colT* q_col = (colT*) q.data();
  if (num_quantized > MAX_QUANTIZED_EXCESS * width * height / proportion) {
    // Ensure there is a maximum bound on the number of entries sent.
    // This should only occur if the threshold sampling is really bad.
    // As a simple recovery process, this just removes enough entries to fit
    // within the appropriate size. Removals begin from the end to avoid copies
    // when deleting entries.
    colT excess = num_quantized -
      (MAX_QUANTIZED_EXCESS * width * height / proportion);
    std::vector<colT> remove_counts(width, 0);
    for (colT header_loc = (width - 1) * HEADER_FACTOR;
         header_loc >= 0 && excess > 0;
         header_loc -= HEADER_FACTOR) {
      colT num_in_col = q_col[header_loc + HEADER_FACTOR] - q_col[header_loc];
      if (num_in_col == 0) continue;
      const colT col_offset = (header_loc / HEADER_FACTOR) * ldim;
      colT num_remove = std::min(excess, num_in_col);
      colT num_left = num_in_col - num_remove;
      DataType pos_recon, neg_recon;
      memcpy(&pos_recon, &q_col[header_loc + 1], sizeof(pos_recon));
      memcpy(&neg_recon, &q_col[header_loc + 2], sizeof(neg_recon));
      DataType* __restrict__ qerror_col = &qerror_buf[col_offset];
      // Add the deleted portions to qerror.
      for (rowT i = q_col[header_loc] + num_left;
           i < q_col[header_loc + HEADER_FACTOR]; ++i) {
        const rowT val = q[i];
        const rowT row = val >> 1;
        if (val & 1) {
          qerror_col[row] += pos_recon;
        } else {
          qerror_col[row] += neg_recon;
        }
      }
      // TODO: When this is called from quantize_replace, this does not update
      // the local matrix.
      q.erase(q.begin() + q_col[header_loc] + num_left, q.end());
      excess -= num_remove;
      remove_counts[header_loc / HEADER_FACTOR] = num_remove;
    }
    // Update all the header locations.
    std::partial_sum(remove_counts.begin(), remove_counts.end(),
                     remove_counts.begin());
    for (colT header_loc = 0; header_loc < width * HEADER_FACTOR;
         header_loc += HEADER_FACTOR) {
      q_col[header_loc + HEADER_FACTOR] -= remove_counts[header_loc / HEADER_FACTOR];
    }
  }
}

template <typename colT, typename rowT>
void lbann_quantizer::intermodel_sum_adaptive_threshold_quantized_impl(
  lbann_comm* comm, Mat& mat, Mat& qerror, int proportion, Mat& im_qerror,
  std::unordered_map<Int, std::vector<rowT>>& adaptive_recv_bufs1,
  std::unordered_map<Int, std::vector<rowT>>& adaptive_recv_bufs2) {
  // Ensure types are reasonable.
  static_assert(std::is_integral<colT>::value && std::is_integral<rowT>::value,
                "Types must be integral");
  static_assert(std::is_unsigned<colT>::value && std::is_unsigned<rowT>::value,
                "Types must be unsigned");
  static_assert(sizeof(colT) == 2 || sizeof(colT) == 4 || sizeof(colT) == 8,
                "colT must be 2, 4, or 8 bytes.");
  static_assert(sizeof(rowT) == 2 || sizeof(rowT) == 4 || sizeof(rowT) == 8,
                "rowT must be 2, 4, or 8 bytes.");
  static_assert(sizeof(colT) >= sizeof(DataType),
                "colT must be at least as large as DataType");
  // Elemental does not seem to support 16-bit MPI types. Using bytes keeps all
  // versions simpler.
  typedef uint8_t mpi_rowT;
  if (qerror.Height() == 0) {
    qerror.Resize(mat.Height(), mat.Width(), mat.LDim());
    Zero(qerror);
  }
  const colT row_header_factor = sizeof(rowT) == 2 ? 2 : 1;
  const colT header_len = row_header_factor * HEADER_FACTOR * mat.Width() +
    row_header_factor;
  const Int max_size = header_len +
    MAX_QUANTIZED_EXCESS * mat.Width() * mat.Height() / proportion;
  if (adaptive_recv_bufs1.find(max_size) == adaptive_recv_bufs1.end()) {
    // Initialize receive buffers.
    adaptive_recv_bufs1.emplace(std::make_pair(max_size, std::vector<rowT>(max_size)));
    adaptive_recv_bufs2.emplace(std::make_pair(max_size, std::vector<rowT>(max_size)));
  }
  std::vector<rowT> rs_quant;
  std::vector<rowT>& rs_recv = adaptive_recv_bufs1[max_size];
  /* NOTE: std::vector::resize() initializes elements. This is unnecessary, but
   * there is no way around it. You cannot use reserve() because that does not
   * update the size or guarantee data() returns anything useful. As far as I
   * can tell, the only way around this would be to either ensure the
   * _implementation_ makes guarantees for reserve(), or implement a custom
   * version of vector.
   */
  auto rs_send_trans = 
    [&qerror, &rs_quant, proportion, this]
    (Mat& mat, IR h, IR w, int& count) {
      auto to_send = mat(h, w);
      auto to_send_qerr = qerror(h, w);
      rs_quant.clear();
      adaptive_threshold_quantize<colT, rowT>(to_send, rs_quant, to_send_qerr,
                                              proportion);
      count = sizeof(rowT) * rs_quant.size();
      return (mpi_rowT*) rs_quant.data();
    };
  auto rs_get_recv_buf = 
    [&rs_recv, max_size] (Mat& mat, int& count) {
      count = sizeof(rowT) * max_size;
      return (mpi_rowT*) rs_recv.data();
    };
  auto rs_recv_trans = 
    [&rs_recv, this]
    (mpi_rowT* buf, Mat& accum) {
      adaptive_threshold_unquantize_add<colT, rowT>(rs_recv, accum);
      // Fix the received bytes count.
      colT recv_size = ((colT*) rs_recv.data())[accum.Width() * HEADER_FACTOR];
      rs_bytes_received -= rs_recv.size() * sizeof(rowT);
      rs_bytes_received += recv_size * sizeof(rowT);
    };
  intermodel_ring_reduce_scatter<mpi_rowT>(comm, mat, false, rs_send_trans,
                                           rs_get_recv_buf, rs_recv_trans);
  std::vector<rowT> local_send;
  std::vector<rowT> ag_send = adaptive_recv_bufs1[max_size];
  std::vector<rowT> ag_recv = adaptive_recv_bufs2[max_size];
  int send_size = 0;
  bool local_sent = false;
  auto ag_reduced_trans =
    [&im_qerror, &local_send, &send_size, proportion, this]
    (Mat& reduced) {
      if (im_qerror.Height() == 0) {
        im_qerror.Resize(reduced.Height(), reduced.Width(), reduced.LDim());
        Zero(im_qerror);
      }
      adaptive_threshold_quantize_replace<colT, rowT>(reduced, local_send,
                                                      im_qerror, proportion);
      send_size = sizeof(rowT) * local_send.size();
    };
  auto ag_get_send_buf = [&ag_send, &local_send, &send_size, &local_sent]
    (int& count) {
      count = send_size;
      if (!local_sent) {
        local_sent = true;
        return (mpi_rowT*) local_send.data();
      } else {
        return (mpi_rowT*) ag_send.data();
      }
    };
  auto ag_get_recv_buf =
    [&ag_recv, max_size] (Mat& recv_view, int& count) {
      count = sizeof(rowT) * max_size;
      return (mpi_rowT*) ag_recv.data();
    };
  auto ag_recv_trans = 
    [&ag_recv, &send_size, proportion, this]
    (mpi_rowT*, Mat& accum) {
      adaptive_threshold_unquantize<colT, rowT>(ag_recv, accum);
      const colT* q_col = (const colT*) ag_recv.data();
      send_size = sizeof(rowT) * q_col[accum.Width() * HEADER_FACTOR];
      // Fix the received bytes count.
      ag_bytes_received -= ag_recv.size() * sizeof(rowT);
      ag_bytes_received += send_size;
    };
  auto ag_swap_bufs =
    [&ag_send, &ag_recv, max_size] (mpi_rowT*, mpi_rowT*) {
      std::swap(ag_send, ag_recv);
    };
  intermodel_ring_allgather<mpi_rowT>(comm, mat, false, ag_reduced_trans,
                                      ag_get_send_buf, ag_get_recv_buf,
                                      ag_recv_trans, ag_swap_bufs);
}

template <typename T>
void lbann_quantizer::intermodel_ring_reduce_scatter(
  lbann_comm* comm, Mat& mat, bool var_recv,
  std::function<T*(Mat&, IR, IR, int&)> send_trans,
  std::function<T*(Mat&, int&)> get_recv_buf,
  std::function<void(T*, Mat&)> recv_trans) {
  double rs_start = get_time();
  int rank = comm->get_model_rank();
  int nprocs = comm->get_num_models();
  // Compute the number of columns each processor sends.
  // The last processor handles the excess.
  Int cols_per_proc = mat.Width() / nprocs;
  Int cols_remainder = mat.Width() % nprocs;
  Int local_col_width = cols_per_proc;
  if (rank == nprocs - 1) local_col_width += cols_remainder;
  // Local view into which to accumulate our received data.
  auto accum_view = mat(IR(0, mat.Height()),
                        IR(rank * cols_per_proc,
                           rank * cols_per_proc + local_col_width));
  // Do the reduce-scatter.
  for (int step = 1; step < nprocs; ++step) {
    // Compute the source/destination.
    int dst = (rank + step) % nprocs;
    int src = (rank - step) % nprocs;
    if (src < 0) src += nprocs;
    // Determine the number of columns to send.
    int send_col_width = cols_per_proc;
    if (dst == nprocs - 1) send_col_width += cols_remainder;
    // Transform the portion to send.
    int send_size;
    double send_trans_start = get_time();
    T* send_buf = send_trans(
      mat, IR(0, mat.Height()),
      IR(dst * cols_per_proc, dst * cols_per_proc + send_col_width), send_size);
    rs_send_trans_time += get_time() - send_trans_start;
    // Send.
    mpi::Request<T> req;
    comm->nb_send(send_buf, send_size, dst, req);
    rs_bytes_sent += send_size * sizeof(T);
    // Get receive buffer.
    double recv_buf_start = get_time();
    int recv_size = 0;
    if (var_recv) {
      recv_size = comm->get_count<T>(src);
    }
    T* recv_buf = get_recv_buf(accum_view, recv_size);
    rs_recv_buf_time += get_time() - recv_buf_start;
    // Receive.
    comm->recv(recv_buf, recv_size, src);
    rs_bytes_received += recv_size * sizeof(T);
    // Transform the received portion.
    double recv_trans_start = get_time();
    recv_trans(recv_buf, accum_view);
    rs_recv_trans_time += get_time() - recv_trans_start;
    comm->wait<T>(req);
  }
  rs_time += get_time() - rs_start;
}

template <typename T>
void lbann_quantizer::intermodel_ring_allgather(
    lbann_comm* comm, Mat& mat, bool var_recv,
    std::function<void(Mat&)> reduced_trans,
    std::function<T*(int&)> get_send_buf,
    std::function<T*(Mat&, int&)> get_recv_buf,
    std::function<void(T*, Mat&)> recv_trans,
    std::function<void(T*, T*)> swap_bufs) {
  double ag_start = get_time();
  int rank = comm->get_model_rank();
  int nprocs = comm->get_num_models();
  // Compute the number of columns each processor sends.
  // The last processor handles the excess.
  int cols_per_proc = mat.Width() / nprocs;
  int cols_remainder = mat.Width() % nprocs;
  int local_col_width = cols_per_proc;
  if (rank == nprocs - 1) local_col_width += cols_remainder;
  // Get the portion of mat that was reduced.
  double reduced_start = get_time();
  auto reduced = mat(IR(0, mat.Height()),
                     IR(rank * cols_per_proc,
                        rank * cols_per_proc + local_col_width));
  // Transform the reduced data.
  reduced_trans(reduced);
  ag_reduced_trans_time += get_time() - reduced_start;
  // Compute the previous/next ranks in the ring.
  int src = rank - 1;
  if (src < 0) src = nprocs - 1;
  int dst = (rank + 1) % nprocs;
  // Do the allgather.
  for (int step = 0; step < nprocs - 1; ++step) {
    // Send our data or forward received data.
    mpi::Request<T> req;
    int send_size;
    T* send_buf = get_send_buf(send_size);
    comm->nb_send(send_buf, send_size, dst, req);
    ag_bytes_sent += send_size * sizeof(T);
    // Compute the original rank that sent the data we're going to receive.
    int data_src = (rank - step - 1) % nprocs;
    if (data_src < 0) data_src += nprocs;
    // Compute the amount of data we're receiving.
    int recv_col_width = cols_per_proc;
    if (data_src == nprocs - 1) recv_col_width += cols_remainder;
    // Get the portion of mat to receive into.
    auto recv_view = mat(IR(0, mat.Height()),
                         IR(data_src * cols_per_proc,
                            data_src * cols_per_proc + recv_col_width));
    // Get receive buffer.
    double recv_buf_start = get_time();
    int recv_size = 0;
    if (var_recv) {
      recv_size = comm->get_count<T>(src);
    }
    T* recv_buf = get_recv_buf(recv_view, recv_size);
    ag_recv_buf_time += get_time() - recv_buf_start;
    // Receive data.
    comm->recv(recv_buf, recv_size, src);
    ag_bytes_received += recv_size * sizeof(T);
    // Transform the received portion.
    double recv_trans_start = get_time();
    recv_trans(recv_buf, recv_view);
    ag_recv_trans_time += get_time() - recv_trans_start;
    comm->wait<T>(req);
    // Swap so we forward the data we just received.
    swap_bufs(send_buf, recv_buf);
    send_size = recv_size;
  }
  ag_time += get_time() - ag_start;
}

}  // namespace lbann

#endif  // LBANN_QUANTIZER_IMPL_HPP_INCLUDED
