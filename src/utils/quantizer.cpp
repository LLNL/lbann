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
// lbann_quantizer .hpp .cpp - Quantization of matrices
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include "lbann/utils/quantizer.hpp"
#include "lbann/utils/random.hpp"
#include <cmath>
#include <omp.h>

namespace lbann {

lbann_quantizer::lbann_quantizer() {
  reset_counters();
}

lbann_quantizer::~lbann_quantizer() {

}

void lbann_quantizer::onebit_quantize(
  const Mat& mat, QuantizedMatrix& qmat, Mat& qerror, bool sample) {
  // Set up the quantized matrix. (+2 for the averages.)
  const El::Int qheight = get_onebit_quantized_matrix_height(mat);
  const El::Int qwidth = mat.Width();
  qmat.Resize(qheight, qwidth);

  const El::Int width = mat.Width();
  const El::Int height = mat.Height();
  const El::Int ldim = mat.LDim();
  const El::Int qmat_ldim = qmat.LDim();
  const DataType *__restrict__ mat_buf = mat.LockedBuffer();
  DataType *__restrict__ qerror_buf = qerror.Buffer();
  qtype *__restrict__ qmat_buf = qmat.Buffer();
  #pragma omp parallel for schedule(static)
  for (El::Int col = 0; col < width; ++col) {
    // First compute the positive and negative column averages.
    DataType pos_sum = 0.0f;
    DataType neg_sum = 0.0f;
    El::Unsigned num_pos = 0;
    El::Unsigned num_neg = 0;
    if (height <= NUM_ONEBIT_SAMPLES || !sample) {
      for (El::Int row = 0; row < height; ++row) {
        const El::Int pos = row + col * ldim;
        const DataType val = mat_buf[pos] + qerror_buf[pos];
        if (val >= 0.0f) {
          pos_sum += val;
          ++num_pos;
        } else {
          neg_sum += val;
          ++num_neg;
        }
      }
    } else {
      // Randomly sample NUM_ONEBIT_SAMPLES to approximate.
      fast_rng_gen& gen = get_fast_generator();
      for (El::Int i = 0; i < NUM_ONEBIT_SAMPLES; ++i) {
        const El::Int pos = fast_rand_int(gen, height);
        const DataType val = mat_buf[pos] + qerror_buf[pos];
        if (val >= 0.0f) {
          pos_sum += val;
          ++num_pos;
        } else {
          neg_sum += val;
          ++num_neg;
        }
      }
    }
    DataType avg_pos = 0.0f;
    if (num_pos > 0) {
      avg_pos = pos_sum / num_pos;
    }
    DataType avg_neg = 0.0f;
    if (num_neg > 0) {
      avg_neg = neg_sum / num_neg;
    }

    // Store the averages.
    // Use memcpy so that we don't violate aliasing rules.
    qtype tmp = 0;
    memcpy(&tmp, &avg_pos, sizeof(avg_pos));
    qmat.Set(0, col, tmp);
    tmp = 0;
    memcpy(&tmp, &avg_neg, sizeof(avg_neg));
    qmat.Set(1, col, tmp);

    // Now quantize the column, NUM_BITS entries at a time.
    El::Int qrow = 2;
    for (El::Int row_chunk = 0; row_chunk < height; row_chunk += NUM_BITS) {
      uqtype q = 0;
      for (uqtype bit = 0; bit < NUM_BITS; ++bit) {
        El::Int row = row_chunk + bit;
        if (row >= height) {
          break;
        }
        const El::Int pos = row + col * ldim;
        const DataType val = mat_buf[pos] + qerror_buf[pos];
        if (val >= 0.0f) {
          q |= uqtype(1) << bit;
          qerror_buf[pos] = val - avg_pos;
        } else {
          qerror_buf[pos] = val - avg_neg;
        }
      }
      qmat_buf[qrow + col * qmat_ldim] = (qtype) q;
      ++qrow;
    }
  }
}

void lbann_quantizer::onebit_quantize(const DistMat& mat, QuantizedMatrix& qmat,
                                      Mat& qerror, bool sample) {
  onebit_quantize(mat.LockedMatrix(), qmat, qerror, sample);
}

void lbann_quantizer::onebit_unquantize(const QuantizedMatrix& qmat, Mat& mat) {
  const El::Int width = mat.Width();
  const El::Int height = mat.Height();
  const El::Int ldim = mat.LDim();
  const El::Int qmat_ldim = qmat.LDim();
  const qtype *__restrict__ qmat_buf = qmat.LockedBuffer();
  DataType *__restrict__ mat_buf = mat.Buffer();
  #pragma omp parallel for schedule(static)
  for (El::Int col = 0; col < width; ++col) {
    El::Int qrow = 2;
    // Extract the averages.
    qtype tmp = qmat.Get(0, col);
    DataType avg_pos;
    memcpy(&avg_pos, &tmp, sizeof(avg_pos));
    tmp = qmat.Get(1, col);
    DataType avg_neg;
    memcpy(&avg_neg, &tmp, sizeof(avg_neg));
    // Unquantize this column.
    for (El::Int row_chunk = 0; row_chunk < height; row_chunk += NUM_BITS) {
      uqtype q = (uqtype) qmat_buf[qrow + col * qmat_ldim];
      for (size_t bit = 0; bit < NUM_BITS; ++bit) {
        El::Int row = row_chunk + bit;
        if (row >= height) {
          break;
        }
        mat_buf[row + col * ldim] = (q >> bit) & 0x1 ? avg_pos : avg_neg;
      }
      ++qrow;
    }
  }
}

void lbann_quantizer::onebit_unquantize(const QuantizedMatrix& qmat,
                                        DistMat& mat) {
  onebit_unquantize(qmat, mat.Matrix());
}

void lbann_quantizer::onebit_unquantize_add(const QuantizedMatrix& qmat,
    Mat& mat) {
  const El::Int width = mat.Width();
  const El::Int height = mat.Height();
  const El::Int ldim = mat.LDim();
  const El::Int qmat_ldim = qmat.LDim();
  const qtype *__restrict__ qmat_buf = qmat.LockedBuffer();
  DataType *__restrict__ mat_buf = mat.Buffer();
  #pragma omp parallel for schedule(static)
  for (El::Int col = 0; col < width; ++col) {
    El::Int qrow = 2;
    // Extract the averages.
    qtype tmp = qmat.Get(0, col);
    DataType avg_pos;
    memcpy(&avg_pos, &tmp, sizeof(avg_pos));
    tmp = qmat.Get(1, col);
    DataType avg_neg;
    memcpy(&avg_neg, &tmp, sizeof(avg_neg));
    // Unquantize this column.
    for (El::Int row_chunk = 0; row_chunk < height; row_chunk += NUM_BITS) {
      uqtype q = (uqtype) qmat_buf[qrow + col * qmat_ldim];
      for (size_t bit = 0; bit < NUM_BITS; ++bit) {
        El::Int row = row_chunk + bit;
        if (row >= height) {
          break;
        }
        mat_buf[row + col * ldim] += (q >> bit) & 0x1 ? avg_pos : avg_neg;
      }
      ++qrow;
    }
  }
}

void lbann_quantizer::intermodel_sum_onebit_quantized(
  lbann_comm *comm, Mat& mat, Mat& qerror) {
  // Initialize qerror.
  if (qerror.Height() == 0) {
    qerror.Resize(mat.Height(), mat.Width(), mat.LDim());
    Zero(qerror);
  }
  std::vector<QuantizedMatrix> qmats(4);
  auto send_transform =
    [&qerror, &qmats, this] (Mat& to_trans, El::IR h, El::IR w, int& count,
                             bool const_data, int call_idx) {
    auto to_send = to_trans(h, w);
    auto to_send_qerr = qerror(h, w);
    QuantizedMatrix& qmat = qmats[call_idx];
    onebit_quantize(to_send, qmat, to_send_qerr);
    count = sizeof(qtype) * qmat.Height() * qmat.Width();
    if (!const_data) {
      // Need to accumulate local errors.
      onebit_unquantize(qmat, to_send);
    }
    return (uint8_t *) qmat.Buffer();
  };
  auto recv_transform =
  [this] (uint8_t *recv_buf, Mat& accum) {
    QuantizedMatrix recv_mat;
    recv_mat.LockedAttach(
      get_onebit_quantized_matrix_height(accum), accum.Width(),
      (qtype *) recv_buf, get_onebit_quantized_matrix_height(accum));
    onebit_unquantize(recv_mat, accum);
    return sizeof(qtype) * recv_mat.Height() * recv_mat.Width();
  };
  auto recv_apply_transform =
  [this] (uint8_t *recv_buf, Mat& accum, bool is_local) {
    if (is_local) {
      Mat recv_mat;
      recv_mat.LockedAttach(accum.Height(), accum.Width(),
                            (DataType *) recv_buf, accum.LDim());
      accum += recv_mat;
      return sizeof(DataType) * recv_mat.Height() * recv_mat.Width();
    } else {
      QuantizedMatrix recv_mat;
      recv_mat.LockedAttach(get_onebit_quantized_matrix_height(accum),
                            accum.Width(), (qtype *) recv_buf,
                            get_onebit_quantized_matrix_height(accum));
      onebit_unquantize_add(recv_mat, accum);
      return sizeof(qtype) * recv_mat.Height() * recv_mat.Width();
    }
  };
  lbann_comm::allreduce_options opts;
  opts.max_reduces = 4;
  comm->intermodel_allreduce(
    mat, sizeof(qtype) * get_onebit_quantized_matrix_height(mat) * mat.Width(),
    std::function<uint8_t *(Mat&, El::IR, El::IR, int&, bool, int)>(send_transform),
    std::function<int(uint8_t *, Mat&)>(recv_transform),
    std::function<int(uint8_t *, Mat&, bool)>(recv_apply_transform),
    opts);
}

void lbann_quantizer::intermodel_sum_onebit_quantized(
  lbann_comm *comm, DistMat& mat, Mat& qerror) {
  intermodel_sum_onebit_quantized(comm, mat.Matrix(), qerror);
}

void lbann_quantizer::threshold_quantize(const Mat& mat, ThreshQuantized& quant,
    Mat& qerror, DataType pos_thresh,
    DataType neg_thresh, bool delta) {
  const El::Int ldim = mat.LDim();
  const El::Int width = mat.Width();
  const El::Int height = mat.Height();
  if (ldim != qerror.LDim()) {
    std::cout << "ldims don't match!" << std::endl;
  }
  const DataType *__restrict__ mat_buf = mat.LockedBuffer();
  DataType *__restrict__ qerror_buf = qerror.Buffer();
  std::vector<ThreshQuantized> thread_qs(omp_get_max_threads());
  if (delta) {
    El::Unsigned prev_pos = 0;
    for (El::Int col = 0; col < width; ++col) {
      for (El::Int row = 0; row < height; ++row) {
        const El::Unsigned pos = row + col * ldim;
        const DataType val = mat_buf[pos] + qerror_buf[pos];
        if (val >= pos_thresh) {
          qerror_buf[pos] = val - pos_thresh;
          // Delta encode pos.
          quant.emplace_back(((pos - prev_pos) << 1) | 1);
          prev_pos = pos;
        } else if (val <= neg_thresh) {
          qerror_buf[pos] = val - neg_thresh;
          quant.emplace_back((pos - prev_pos) << 1);
          prev_pos = pos;
        } else {
          qerror_buf[pos] = val;
        }
      }
    }
  } else {
    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      #pragma omp for schedule(static)
      for (El::Int col = 0; col < width; ++col) {
        for (El::Int row = 0; row < height; ++row) {
          const El::Unsigned pos = row + col * ldim;
          const DataType val = mat_buf[pos] + qerror_buf[pos];
          if (val >= pos_thresh) {
            qerror_buf[pos] = val - pos_thresh;
            thread_qs[tid].emplace_back((pos << 1) | 1);
          } else if (val <= neg_thresh) {
            qerror_buf[pos] = val - neg_thresh;
            thread_qs[tid].emplace_back(pos << 1);
          } else {
            qerror_buf[pos] = val;
          }
        }
      }
    }
    // Copy the temporary vectors.
    for (auto&& thread_q : thread_qs) {
      quant.insert(quant.end(), thread_q.begin(), thread_q.end());
    }
  }
}

void lbann_quantizer::threshold_quantize(
  const DistMat& mat, ThreshQuantized& q, Mat& qerror, DataType pos_thresh,
  DataType neg_thresh, bool delta) {
  threshold_quantize(mat.LockedMatrix(), q, qerror, pos_thresh, neg_thresh,
                     delta);
}

void lbann_quantizer::threshold_unquantize(
  const ThreshQuantized& quant, Mat& mat, DataType pos_thresh,
  DataType neg_thresh, bool delta) {
  DataType *__restrict__ buf = mat.Buffer();
  if (delta) {
    El::Unsigned prev_pos = 0;
    for (El::Unsigned i = 0; i < quant.size(); ++i) {
      const uqtype q = quant[i];
      const El::Unsigned pos = (q >> 1) + prev_pos;
      prev_pos = pos;
      if (q & 1) {
        buf[pos] = pos_thresh;
      } else {
        buf[pos] = neg_thresh;
      }
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (El::Unsigned i = 0; i < quant.size(); ++i) {
      const uqtype q = quant[i];
      const El::Unsigned pos = q >> 1;
      if (q & 1) {
        buf[pos] = pos_thresh;
      } else {
        buf[pos] = neg_thresh;
      }
    }
  }
}

void lbann_quantizer::threshold_unquantize(
  const ThreshQuantized& quant, DistMat& mat, DataType pos_thresh,
  DataType neg_thresh, bool delta) {
  threshold_unquantize(quant, mat.Matrix(), pos_thresh, neg_thresh, delta);
}

void lbann_quantizer::threshold_unquantize_apply(
  const ThreshQuantized& quant, Mat& mat, DataType pos_thresh,
  DataType neg_thresh, std::vector<El::Unsigned>& positions, bool delta) {
  // A general note on positions that I'm putting here because I'm not sure
  // where else to: Using a vector admits the possibility that we have
  // duplicate entries. This could be fixed by using an unordered_set, but when
  // I benchmarked this, it increased our runtime by ~5 times. Having duplicate
  // entries should not change the final result: it means that
  // threshold_quantize_apply may quantize the same entry multiple times, but
  // the final unquantize is not an _apply, and so will just set that entry to
  // the same value multiple times. We send some extra data, but the overhead
  // is small.
  DataType *__restrict__ buf = mat.Buffer();
  if (delta) {
    El::Unsigned prev_pos = 0;
    for (El::Unsigned i = 0; i < quant.size(); ++i) {
      const uqtype q = quant[i];
      const El::Unsigned pos = (q >> 1) + prev_pos;
      prev_pos = pos;
      positions.emplace_back(pos);
      if (q & 1) {
        buf[pos] += pos_thresh;
      } else {
        buf[pos] += neg_thresh;
      }
    }
  } else {
    for (El::Unsigned i = 0; i < quant.size(); ++i) {
      const uqtype q = quant[i];
      const El::Unsigned pos = q >> 1;
      positions.emplace_back(pos);
      if (q & 1) {
        buf[pos] += pos_thresh;
      } else {
        buf[pos] += neg_thresh;
      }
    }
  }
}

void lbann_quantizer::threshold_quantize_apply(
  const Mat& mat, ThreshQuantized& quant, Mat& qerror, DataType pos_thresh,
  DataType neg_thresh, std::vector<El::Unsigned>& positions, bool delta) {
  const DataType *__restrict__ mat_buf = mat.LockedBuffer();
  DataType *__restrict__ qerror_buf = qerror.Buffer();
  if (delta) {
    // Need to sort so positions are in order, otherwise our delta encoding
    // doesn't work. (Could be solved by adding stops, but maybe not worth it.)
    std::sort(positions.begin(), positions.end());
    El::Unsigned prev_pos = 0;
    for (const auto& pos : positions) {
      const DataType val = mat_buf[pos] + qerror_buf[pos];
      if (val >= pos_thresh) {
        qerror_buf[pos] = val - pos_thresh;
        quant.emplace_back(((pos - prev_pos) << 1) | 1);
        prev_pos = pos;
      } else if (val <= neg_thresh) {
        qerror_buf[pos] = val - neg_thresh;
        quant.emplace_back((pos - prev_pos) << 1);
        prev_pos = pos;
      } else {
        qerror_buf[pos] = val;
      }
    }
  } else {
    for (const auto& pos : positions) {
      const DataType val = mat_buf[pos] + qerror_buf[pos];
      if (val >= pos_thresh) {
        quant.emplace_back((pos << 1) | 1);
        qerror_buf[pos] = val - pos_thresh;
      } else if (val <= neg_thresh) {
        quant.emplace_back(pos << 1);
        qerror_buf[pos] = val - neg_thresh;
      } else {
        qerror_buf[pos] = val;
      }
    }
  }
}

void lbann_quantizer::intermodel_sum_threshold_quantized(
  lbann_comm *comm, Mat& mat, Mat& qerror, DataType pos_thresh,
  DataType neg_thresh) {
  // Temporarily not supported until threshold quantization is updated to
  // have upper bounds on its send size.
  throw lbann_exception("Threshold quantized allreduce not supported");
}

void lbann_quantizer::intermodel_sum_threshold_quantized(
  lbann_comm *comm, DistMat& mat, Mat& qerror, DataType pos_thresh,
  DataType neg_thresh) {
  intermodel_sum_threshold_quantized(comm, mat.Matrix(), qerror, pos_thresh,
                                     neg_thresh);
}

void lbann_quantizer::intermodel_sum_adaptive_quantized(
  lbann_comm *comm, Mat& mat, Mat& qerror, int proportion) {
  // Select which algorithm to use based on the size of mat.
  // Multiply at 64 bits to avoid overflows.
  size_t mat_size = ((size_t) mat.Height()) * ((size_t) mat.Width());
  // Check signed version because we need one bit for the quantized value.
  if (mat_size > std::numeric_limits<int32_t>::max()) {
    intermodel_sum_adaptive_quantized_impl<uint64_t, uint64_t>(
      comm, mat, qerror, proportion);
  } else {
    // Check whether we can use 16-bit row indices.
    // Determine the column type (at compile time) based upon DataType.
    typedef std::conditional<sizeof(DataType) <= 4, uint32_t, uint64_t>::type colT;
    if (mat.Height() > std::numeric_limits<int16_t>::max()) {
      intermodel_sum_adaptive_quantized_impl<colT, uint32_t>(
        comm, mat, qerror, proportion);
    } else {
      intermodel_sum_adaptive_quantized_impl<colT, uint16_t>(
        comm, mat, qerror, proportion);
    }
  }
}

void lbann_quantizer::intermodel_sum_adaptive_quantized(
  lbann_comm *comm, DistMat& mat, Mat& qerror, int proportion) {
  intermodel_sum_adaptive_quantized(comm, mat.Matrix(), qerror,
                                    proportion);
}

lbann_quantizer::adaptive_thresholds lbann_quantizer::proportion_threshold(
  const Mat& mat, const Mat& qerror, int proportion, bool sample) {
  double proportion_start = get_time();
  std::vector<DataType> entries;
  const El::Int height = mat.Height();
  const El::Int width = mat.Width();
  const El::Int ldim = mat.LDim();
  const DataType *__restrict__ mat_buf = mat.LockedBuffer();
  const DataType *__restrict__ qerror_buf = qerror.LockedBuffer();
  // Bail out if needed.
  if (width == 0) {
    return { 0.0f, 0.0f };
  }
  if (width * height <= NUM_THRESHOLD_SAMPLES || !sample) {
    // Copy entire matrix into vector.
    entries.reserve(width * height);
    for (El::Int col = 0; col < width; ++col) {
      const El::Int col_offset = col * ldim;
      for (El::Int row = 0; row < height; ++row) {
        const El::Unsigned pos = row + col_offset;
        entries.emplace_back(mat_buf[pos] + qerror_buf[pos]);
      }
    }
  } else {
    // Randomly sample entries to approximate everything.
    entries.reserve(NUM_THRESHOLD_SAMPLES);
    fast_rng_gen& gen = get_fast_generator();
    std::vector<El::Unsigned> poses(NUM_THRESHOLD_SAMPLES);
    for (El::Unsigned i = 0; i < NUM_THRESHOLD_SAMPLES; ++i) {
      const El::Unsigned pos = fast_rand_int(gen, height) + fast_rand_int(gen, width) * ldim;
      __builtin_prefetch(&mat_buf[pos]);
      __builtin_prefetch(&qerror_buf[pos]);
      poses[i] = pos;
    }
    for (El::Unsigned i = 0; i < NUM_THRESHOLD_SAMPLES; ++i) {
      const El::Unsigned pos = poses[i];
      entries.emplace_back(mat_buf[pos] + qerror_buf[pos]);
    }
  }
  // Determine the number of entries to keep.
  El::Int num_to_keep = std::max(1, (int) entries.size() / proportion);
  // Determine the threshold values.
  // This finds the num_to_keep'th value if sample were sorted by magnitude
  // and assigns it to the appropriate threshold, then checks the upper portion
  // of the partially-sorted vector to find the other threshold.
  // In the case that the threshold would be 0, it is instead a small non-zero
  // value.
  DataType pos_thresh = std::numeric_limits<DataType>::max();
  DataType neg_thresh = -std::numeric_limits<DataType>::max();
  auto i = entries.begin() + (entries.size() - num_to_keep);
  std::nth_element(entries.begin(), i, entries.end(),
  [] (const DataType a, const DataType b) {
    return std::abs(a) < std::abs(b);
  });
  if (*i > 0) {
    pos_thresh = *i;
    for (++i; i < entries.end(); ++i) {
      // Find the largest (closest to 0) negative value.
      if (*i < 0) {
        neg_thresh = std::max(neg_thresh, *i);
      }
    }
  } else if (*i < 0) {
    neg_thresh = *i;
    for (++i; i < entries.end(); ++i) {
      // Find the smallest (closest to 0) positive value.
      if (*i > 0) {
        pos_thresh = std::min(pos_thresh, *i);
      }
    }
  }
  // If there are no values of a sign, select threshold such that none are sent.
  if (pos_thresh == std::numeric_limits<DataType>::max()) {
    pos_thresh = -neg_thresh;
  }
  if (neg_thresh == -std::numeric_limits<DataType>::max()) {
    neg_thresh = -pos_thresh;
  }
  proportion_time += get_time() - proportion_start;
  return { pos_thresh, neg_thresh };
}

lbann_quantizer::adaptive_reconstructions lbann_quantizer::col_reconstruction(
  const Mat& mat, const Mat& qerror, El::Int col,
  const adaptive_thresholds threshes, bool sample) {
  DataType pos_sum = 0.0f;
  El::Unsigned pos_count = 0;
  DataType neg_sum = 0.0f;
  El::Unsigned neg_count = 0;
#if LBANN_QUANTIZER_TERNARY
  DataType zero_sum = 0.0f;
  El::Unsigned zero_count = 0;
#endif
  const El::Int height = mat.Height();
  const El::Int col_offset = col * mat.LDim();
  const DataType *__restrict__ mat_buf = mat.LockedBuffer();
  const DataType *__restrict__ qerror_buf = qerror.LockedBuffer();
  if (height <= NUM_RECON_SAMPLES || !sample) {
    for (El::Int row = 0; row < height; ++row) {
      const El::Unsigned pos = row + col_offset;
      const DataType val = mat_buf[pos] + qerror_buf[pos];
      if (val >= threshes.pos_thresh) {
        pos_sum += val;
        ++pos_count;
      } else {
        if (val <= threshes.neg_thresh) {
          neg_sum += val;
          ++neg_count;
        }
#if LBANN_QUANTIZER_TERNARY
        else {
          zero_sum += val;
          ++zero_count;
        }
#endif
      }
    }
  } else {
    // Randomly sample entries to approximate the means.
    fast_rng_gen& gen = get_fast_generator();
    bool is_pow2 = !(height & (height - 1));  // Assumes height != 0.
    std::vector<El::Unsigned> poses(NUM_RECON_SAMPLES);
    if (is_pow2) {
      for (El::Unsigned i = 0; i < NUM_RECON_SAMPLES; ++i) {
        const El::Unsigned pos = fast_rand_int_pow2(gen, height) + col_offset;
        __builtin_prefetch(&mat_buf[pos]);
        __builtin_prefetch(&qerror_buf[pos]);
        poses[i] = pos;
      }
    } else {
      for (El::Unsigned i = 0; i < NUM_RECON_SAMPLES; ++i) {
        const El::Unsigned pos = fast_rand_int(gen, height) + col_offset;
        __builtin_prefetch(&mat_buf[pos]);
        __builtin_prefetch(&qerror_buf[pos]);
        poses[i] = pos;
      }
    }
    for (El::Unsigned i = 0; i < NUM_RECON_SAMPLES; ++i) {
      //const unsigned pos = row_dist(gen) + col_offset;
      const El::Unsigned pos = poses[i];
      const DataType val = mat_buf[pos] + qerror_buf[pos];
      if (val >= threshes.pos_thresh) {
        pos_sum += val;
        ++pos_count;
      } else {
        if (val <= threshes.neg_thresh) {
          neg_sum += val;
          ++neg_count;
        }
#if LBANN_QUANTIZER_TERNARY
        else {
          zero_sum += val;
          ++zero_count;
        }
#endif
      }
    }
  }
  // Compute the means. Use the thresholds as initial values in case the
  // sampling does not include any positive or negative values.
  DataType pos_recon = threshes.pos_thresh;
  DataType neg_recon = threshes.neg_thresh;
#if LBANN_QUANTIZER_TERNARY
  DataType zero_recon = 0.0f;
#endif
  if (pos_count > 0) {
    pos_recon = pos_sum / pos_count;
  }
  if (neg_count > 0) {
    neg_recon = neg_sum / neg_count;
  }
#if LBANN_QUANTIZER_TERNARY
  if (zero_count > 0) {
    zero_recon = zero_sum / zero_count;
  }
  return { pos_recon, neg_recon, zero_recon };
#else
  return { pos_recon, neg_recon };
#endif
}

}  // namespace lbann
