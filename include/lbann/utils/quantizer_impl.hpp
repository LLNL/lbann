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

namespace lbann {

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_quantize(
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
  const DataType *const __restrict__ mat_buf = mat.LockedBuffer();
  DataType *__restrict__ qerror_buf = qerror.Buffer();
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
  colT *q_col = (colT *) q.data();
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
      const DataType *const __restrict__ mat_col = &mat_buf[col_offset];
      DataType *__restrict__ qerror_col = &qerror_buf[col_offset];
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
  q_col = (colT *) q.data();
  q_col[HEADER_FACTOR * width] = (colT) q.size();
  quantized_count = q.size() - header_len;
  adaptive_bound<colT, rowT>(mat, qerror, q, proportion);
}

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_quantize(
  const DistMat& mat, std::vector<rowT>& q, Mat& qerror, int proportion) {
  adaptive_quantize<colT, rowT>(mat.LockedMatrix(), q, qerror, proportion);
}

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_unquantize(
  const rowT *q, Mat& mat) {
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
  DataType *__restrict__ buf = mat.Buffer();
  const colT header_len = mat.Width() * HEADER_FACTOR;
#if LBANN_QUANTIZER_TERNARY
  const colT height = mat.Height();
#endif
  const colT ldim = mat.LDim();
  const colT *q_col = (const colT *) q;
  const int num_threads = get_adaptive_quantization_threads(mat.Width());
  #pragma omp parallel for schedule(dynamic, 1), firstprivate(header_len, buf) num_threads(num_threads)
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
    DataType *__restrict__ buf_col = &buf[col_offset];
    const colT chunk_start = q_col[header_loc];
    const colT chunk_end = q_col[header_loc + HEADER_FACTOR] - chunk_start;
    const rowT *const __restrict__ q_ = &(q[chunk_start]);
    for (rowT i = 0; i < chunk_end; ++i) {
      const rowT val = q_[i];
      const rowT row = val >> 1;
      buf_col[row] = val & 0x1 ? pos_recon : neg_recon;
    }
  }
}

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_unquantize(
  const rowT *q, DistMat& mat) {
  adaptive_unquantize<colT, rowT>(q, mat.Matrix());
}

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_unquantize_add(
  const rowT *q, Mat& mat) {
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
  DataType *__restrict__ buf = mat.Buffer();
  const colT header_len = mat.Width() * HEADER_FACTOR;
#if LBANN_QUANTIZER_TERNARY
  const colT height = mat.Height();
#endif
  const colT ldim = mat.LDim();
  const colT *q_col = (const colT *) q;
  const int num_threads = get_adaptive_quantization_threads(mat.Width());
  #pragma omp parallel for schedule(dynamic, 1), firstprivate(header_len, buf) num_threads(num_threads)
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
    DataType *__restrict__ buf_col = &buf[col_offset];
    const colT chunk_start = q_col[header_loc];
    const colT chunk_end = q_col[header_loc + HEADER_FACTOR] - chunk_start;
    const rowT *const __restrict__ q_ = &(q[chunk_start]);
    for (rowT i = 0; i < chunk_end; ++i) {
      const rowT val = q_[i];
      const rowT row = val >> 1;
      buf_col[row] += val & 0x1 ? pos_recon : neg_recon;
    }
  }
}

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_quantize_replace(
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
  DataType *__restrict__ mat_buf = mat.Buffer();
  DataType *__restrict__ qerror_buf = qerror.Buffer();
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
  colT *q_col = (colT *) q.data();
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
      DataType *__restrict__ mat_col = &mat_buf[col_offset];
      DataType *__restrict__ qerror_col = &qerror_buf[col_offset];
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
  q_col = (colT *) q.data();
  q_col[HEADER_FACTOR * width] = q.size();
  quantized_count = q.size() - header_len;
  adaptive_bound<colT, rowT>(mat, qerror, q, proportion);
}

template <typename colT, typename rowT>
void lbann_quantizer::adaptive_bound(
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
  DataType *__restrict__ qerror_buf = qerror.Buffer();
  const colT width = mat.Width();
  const colT height = mat.Height();
  const colT ldim = mat.LDim();
  const colT row_header_factor = sizeof(rowT) == 2 ? 2 : 1;
  const colT header_len = row_header_factor * HEADER_FACTOR * width +
                          row_header_factor;
  const colT num_quantized = q.size() - header_len;
  colT *q_col = (colT *) q.data();
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
         excess > 0;
         header_loc -= HEADER_FACTOR) {
      colT num_in_col = q_col[header_loc + HEADER_FACTOR] - q_col[header_loc];
      if (num_in_col == 0) {
        continue;
      }
      const colT col_offset = (header_loc / HEADER_FACTOR) * ldim;
      colT num_remove = std::min(excess, num_in_col);
      colT num_left = num_in_col - num_remove;
      DataType pos_recon, neg_recon;
      memcpy(&pos_recon, &q_col[header_loc + 1], sizeof(pos_recon));
      memcpy(&neg_recon, &q_col[header_loc + 2], sizeof(neg_recon));
      DataType *__restrict__ qerror_col = &qerror_buf[col_offset];
      // Add the deleted portions to qerror.
      for (colT i = q_col[header_loc] + num_left;
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
void lbann_quantizer::adaptive_quantize_slice(
  const std::vector<rowT>& q, const Mat& mat, Mat& qerror,
  std::vector<rowT>& slice, colT start, colT end, int proportion) {
  const colT width = end - start;
  const colT row_header_factor = sizeof(rowT) == 2 ? 2 : 1;
  const colT header_len = row_header_factor * width * HEADER_FACTOR +
                          row_header_factor;
  // Copy the header over. Locations will need to be adjusted later.
  const colT *q_col = (const colT *) q.data();
  const colT total_len = header_len + q_col[HEADER_FACTOR * end] - q_col[HEADER_FACTOR * start];
  slice.resize(total_len);
  colT *slice_col = (colT *) slice.data();
  std::copy(&q_col[HEADER_FACTOR*start], &q_col[HEADER_FACTOR*end + 1],
            slice_col);
  // Copy data over.
  std::copy(q.begin() + slice_col[0], q.begin() + slice_col[HEADER_FACTOR * width],
            slice.begin() + header_len);
  // Adjust locations.
  const colT adjust = slice_col[0] - header_len;
  for (colT header_loc = 0; header_loc <= HEADER_FACTOR * width; header_loc += HEADER_FACTOR) {
    slice_col[header_loc] -= adjust;
  }
  adaptive_bound<colT, rowT>(mat, qerror, slice, proportion);
}

template <typename colT, typename rowT>
void lbann_quantizer::intermodel_sum_adaptive_quantized_impl(
  lbann_comm *comm, Mat& mat, Mat& qerror, int proportion) {
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
  if (qerror.Height() == 0) {
    qerror.Resize(mat.Height(), mat.Width(), mat.LDim());
    Zero(qerror);
  }
  const colT row_header_factor = sizeof(rowT) == 2 ? 2 : 1;
  const colT header_len = row_header_factor * HEADER_FACTOR * mat.Width() +
                          row_header_factor;
  const Int max_size = (header_len +
                        MAX_QUANTIZED_EXCESS * mat.Width() * mat.Height() / proportion) *
                       sizeof(rowT);
  std::vector<rowT> quant;
  std::vector<std::vector<rowT>> quant_slices(4);
  auto send_transform =
    [&qerror, &quant, &quant_slices, proportion, this]
  (Mat& to_trans, IR h, IR w, int& count, bool const_data, int call_idx) {
    auto to_send = to_trans(h, w);
    auto to_send_qerr = qerror(h, w);
    if (const_data) {
      // In this case we can quantize the entire matrix then slice it.
      if (quant.empty()) {
        adaptive_quantize<colT, rowT>(to_trans, quant, qerror, proportion);
      }
      std::vector<rowT>& quant_slice = quant_slices[call_idx];
      quant_slice.clear();
      adaptive_quantize_slice<colT, rowT>(quant, to_send, to_send_qerr,
                                          quant_slice, w.beg, w.end,
                                          proportion);
      count = sizeof(rowT) * quant_slice.size();
      return (uint8_t *) quant_slice.data();
    } else {
      quant.clear();
      adaptive_quantize_replace<colT, rowT>(to_send, quant, to_send_qerr,
                                            proportion);
      count = sizeof(rowT) * quant.size();
      return (uint8_t *) quant.data();
    }
  };
  auto recv_transform =
  [this] (uint8_t *recv_buf, Mat& accum) {
    adaptive_unquantize<colT, rowT>((rowT *) recv_buf, accum);
    const colT *q_col = (colT *) recv_buf;
    return sizeof(rowT) * q_col[accum.Width() * HEADER_FACTOR];
  };
  auto recv_apply_transform =
  [this] (uint8_t *recv_buf, Mat& accum, bool is_local) -> int {
    if (is_local) {
      Mat recv_mat;
      recv_mat.LockedAttach(accum.Height(), accum.Width(),
                            (DataType *) recv_buf, accum.LDim());
      accum += recv_mat;
      return sizeof(DataType) * recv_mat.Height() * recv_mat.Width();
    } else {
      adaptive_unquantize_add<colT, rowT>((rowT *) recv_buf, accum);
      const colT *q_col = (colT *) recv_buf;
      return sizeof(rowT) * q_col[accum.Width() * HEADER_FACTOR];
    }
  };
  lbann_comm::allreduce_options opts;
  opts.max_reduces = 4;
  comm->intermodel_allreduce(
    mat, max_size,
    std::function<uint8_t *(Mat&, IR, IR, int&, bool, int)>(send_transform),
    std::function<int(uint8_t *, Mat&)>(recv_transform),
    std::function<int(uint8_t *, Mat&, bool)>(recv_apply_transform),
    opts);
}

}  // namespace lbann

#endif  // LBANN_QUANTIZER_IMPL_HPP_INCLUDED
