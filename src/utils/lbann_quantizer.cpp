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
#include "lbann/utils/lbann_quantizer.hpp"
#include "lbann/utils/lbann_random.hpp"
#include <cmath>
#include <omp.h>

namespace lbann {

lbann_quantizer::lbann_quantizer() {
  reset_bytes_counters();
  reset_time_counters();
  quantized_count = 0;
}

lbann_quantizer::~lbann_quantizer() {

}

void lbann_quantizer::intermodel_sum(lbann_comm* comm, Mat& mat) {
  Mat rs_recv;
  auto rs_send_trans = 
    [] (Mat& mat, IR h, IR w, int& count) {
      // Assumes h is the full height of the matrix and column-major order
      // so the buffer is contiguous.
      auto to_send = mat(h, w);
      count = to_send.Height() * to_send.Width();
      return to_send.Buffer();
    };
  auto rs_get_recv_buf =
    [&rs_recv] (Mat& mat, int& count) {
      if (rs_recv.Width() != mat.Width()) {
        rs_recv.Resize(mat.Height(), mat.Width());
      }
      count = rs_recv.Height() * rs_recv.Width();
      return rs_recv.Buffer();
    };
  auto rs_recv_trans =
    [&rs_recv] (DataType*, Mat& accum) {
      accum += rs_recv;
    };
  intermodel_ring_reduce_scatter<DataType>(comm, mat, false, rs_send_trans,
                                           rs_get_recv_buf, rs_recv_trans);
  Mat ag_send;
  Mat ag_recv;
  auto ag_reduced_trans =
    [&ag_send] (Mat& reduced) {
      View(ag_send, reduced);
    };
  auto ag_get_send_buf = [&ag_send] (int& count) {
      count = ag_send.Width() * ag_send.Height();
      return ag_send.Buffer();
    };
  auto ag_get_recv_buf =
    [&ag_recv] (Mat& recv_view, int& count) {
      count = recv_view.Height() * recv_view.Width();
      View(ag_recv, recv_view);
      return recv_view.Buffer();
    };
  auto ag_recv_trans =
    [] (DataType*, Mat& accum) {
      // NOP.
    };
  auto ag_swap_bufs =
    [&ag_send, &ag_recv] (DataType*, DataType*) {
      Mat tmp_view = View(ag_send);
      View(ag_send, ag_recv);
      View(ag_recv, tmp_view);
    };
  intermodel_ring_allgather<DataType>(comm, mat, false, ag_reduced_trans,
                                      ag_get_send_buf, ag_get_recv_buf,
                                      ag_recv_trans, ag_swap_bufs);
}

void lbann_quantizer::intermodel_sum(lbann_comm* comm, DistMat& mat) {
  intermodel_sum(comm, mat.Matrix());
}

void lbann_quantizer::quantize(
  const Mat& mat, QuantizedMatrix& qmat, Mat& qerror, bool sample) {
  // Set up the quantized matrix. (+2 for the averages.)
  const Int qheight = get_quantized_matrix_height(mat);
  const Int qwidth = mat.Width();
  qmat.Resize(qheight, qwidth);

  const Int width = mat.Width();
  const Int height = mat.Height();
  const Int ldim = mat.LDim();
  const Int qmat_ldim = qmat.LDim();
  const DataType* __restrict__ mat_buf = mat.LockedBuffer();
  DataType* __restrict__ qerror_buf = qerror.Buffer();
  qtype* __restrict__ qmat_buf = qmat.Buffer();
  #pragma omp parallel for schedule(static)
  for (Int col = 0; col < width; ++col) {
    // First compute the positive and negative column averages.
    DataType pos_sum = 0.0f;
    DataType neg_sum = 0.0f;
    Unsigned num_pos = 0;
    Unsigned num_neg = 0;
    if (height <= NUM_ONEBIT_SAMPLES || !sample) {
      for (Int row = 0; row < height; ++row) {
        const Int pos = row + col * ldim;
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
      for (Int i = 0; i < NUM_ONEBIT_SAMPLES; ++i) {
        const Int pos = fast_rand_int(gen, height);
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
    Int qrow = 2;
    for (Int row_chunk = 0; row_chunk < height; row_chunk += NUM_BITS) {
      uqtype q = 0;
      for (unsigned bit = 0; bit < NUM_BITS; ++bit) {
        Int row = row_chunk + bit;
        if (row >= height) {
          break;
        }
        const Int pos = row + col * ldim;
        const DataType val = mat_buf[pos] + qerror_buf[pos];
        if (val >= 0.0f) {
          q |= 1 << bit;
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

void lbann_quantizer::quantize(const DistMat& mat, QuantizedMatrix& qmat,
                               Mat& qerror, bool sample) {
  quantize(mat.LockedMatrix(), qmat, qerror, sample);
}

void lbann_quantizer::unquantize(const QuantizedMatrix& qmat, Mat& mat) {
  const Int width = mat.Width();
  const Int height = mat.Height();
  const Int ldim = mat.LDim();
  const Int qmat_ldim = qmat.LDim();
  const qtype* __restrict__ qmat_buf = qmat.LockedBuffer();
  DataType* __restrict__ mat_buf = mat.Buffer();
  #pragma omp parallel for schedule(static)
  for (Int col = 0; col < width; ++col) {
    Int qrow = 2;
    // Extract the averages.
    qtype tmp = qmat.Get(0, col);
    DataType avg_pos;
    memcpy(&avg_pos, &tmp, sizeof(avg_pos));
    tmp = qmat.Get(1, col);
    DataType avg_neg;
    memcpy(&avg_neg, &tmp, sizeof(avg_neg));
    // Unquantize this column.
    for (Int row_chunk = 0; row_chunk < height; row_chunk += NUM_BITS) {
      uqtype q = (uqtype) qmat_buf[qrow + col * qmat_ldim];
      for (size_t bit = 0; bit < NUM_BITS; ++bit) {
        Int row = row_chunk + bit;
        if (row >= height) {
          break;
        }
        mat_buf[row + col * ldim] = (q >> bit) & 0x1 ? avg_pos : avg_neg;
      }
      ++qrow;
    }
  }
}

void lbann_quantizer::unquantize(const QuantizedMatrix& qmat, DistMat& mat) {
  unquantize(qmat, mat.Matrix());
}

void lbann_quantizer::unquantize_add(const QuantizedMatrix& qmat, Mat& mat) {
  const Int width = mat.Width();
  const Int height = mat.Height();
  const Int ldim = mat.LDim();
  const Int qmat_ldim = qmat.LDim();
  const qtype* __restrict__ qmat_buf = qmat.LockedBuffer();
  DataType* __restrict__ mat_buf = mat.Buffer();
  #pragma omp parallel for schedule(static)
  for (Int col = 0; col < width; ++col) {
    Int qrow = 2;
    // Extract the averages.
    qtype tmp = qmat.Get(0, col);
    DataType avg_pos;
    memcpy(&avg_pos, &tmp, sizeof(avg_pos));
    tmp = qmat.Get(1, col);
    DataType avg_neg;
    memcpy(&avg_neg, &tmp, sizeof(avg_neg));
    // Unquantize this column.
    for (Int row_chunk = 0; row_chunk < height; row_chunk += NUM_BITS) {
      uqtype q = (uqtype) qmat_buf[qrow + col * qmat_ldim];
      for (size_t bit = 0; bit < NUM_BITS; ++bit) {
        Int row = row_chunk + bit;
        if (row >= height) {
          break;
        }
        mat_buf[row + col * ldim] += (q >> bit) & 0x1 ? avg_pos : avg_neg;
      }
      ++qrow;
    }
  }
}

void lbann_quantizer::intermodel_sum_quantized(
  lbann_comm* comm, Mat& mat, Mat& qerror, Mat& im_qerror,
  bool do_adagrad, Mat* gradhist) {
  // Initialize qerror.
  if (qerror.Height() == 0) {
    qerror.Resize(mat.Height(), mat.Width(), mat.LDim());
    Zero(qerror);
  }
  QuantizedMatrix to_send_quant;
  QuantizedMatrix rs_recv;
  auto rs_send_trans =
    [&qerror, &to_send_quant, this] (Mat& mat, IR h, IR w, int& count) {
      auto to_send = mat(h, w);
      auto to_send_qerr = qerror(h, w);
      quantize(to_send, to_send_quant, to_send_qerr);
      count = to_send_quant.Height() * to_send_quant.Width();
      return to_send_quant.Buffer();
    };
  auto rs_get_recv_buf = 
    [&rs_recv, this] (Mat& mat, int& count) {
      if (rs_recv.Width() != mat.Width()) {
        rs_recv.Resize(get_quantized_matrix_height(mat), mat.Width());
      }
      count = rs_recv.Height() * rs_recv.Width();
      return rs_recv.Buffer();
    };
  auto rs_recv_trans = 
    [&rs_recv, this] (qtype*, Mat& accum) {
      unquantize_add(rs_recv, accum);
    };
  intermodel_ring_reduce_scatter<qtype>(comm, mat, false, rs_send_trans,
                                        rs_get_recv_buf, rs_recv_trans);
  QuantizedMatrix ag_send;
  QuantizedMatrix ag_recv;
  std::function<DataType(const DataType&)> _sq = [](const DataType& x) { return x*x; };
  std::function<DataType(const DataType&)> _sqrt =
    [](const DataType& x) { return 1.0f / (std::sqrt(x) + 1e-8f); };
  auto ag_reduced_trans =
    [&im_qerror, &ag_send, gradhist, do_adagrad, _sq, _sqrt, this] (Mat& reduced) {
      if (do_adagrad) {
        if (gradhist->Height() == 0) {
          Zeros(*gradhist, reduced.Height(), reduced.Width());
        }
        Mat tmp(reduced);  // Temporary for AdaGrad computations.
        // Compute squared gradient and store in history.
        EntrywiseMap(tmp, _sq);
        *gradhist += tmp;
        // Compute 1/sqrt(gradhist) with small perturbation.
        Copy(*gradhist, tmp);
        EntrywiseMap(tmp, _sqrt);                     
        // Adjust update.
        Mat reduced_copy(reduced);
        Hadamard(tmp, reduced_copy, reduced);
      }
      if (im_qerror.Height() == 0) {
        im_qerror.Resize(reduced.Height(), reduced.Width(), reduced.LDim());
        Zero(im_qerror);
      }
      quantize(reduced, ag_send, im_qerror);
      unquantize(ag_send, reduced);
    };
  auto ag_get_send_buf = [&ag_send] (int& count) {
      count = ag_send.Height() * ag_send.Width();
      return ag_send.Buffer();
    };
  auto ag_get_recv_buf = 
    [&ag_recv, this] (Mat& recv_view, int& count) {
      ag_recv.Resize(get_quantized_matrix_height(recv_view), recv_view.Width());
      count = ag_recv.Height() * ag_recv.Width();
      return ag_recv.Buffer();
    };
  auto ag_recv_trans = 
    [&ag_recv, this] (qtype*, Mat& accum) {
      unquantize(ag_recv, accum);
    };
  auto ag_swap_bufs = 
    [&ag_send, &ag_recv] (qtype*, qtype*) {
      std::swap(ag_send, ag_recv);
    };
  intermodel_ring_allgather<qtype>(comm, mat, false, ag_reduced_trans,
                                   ag_get_send_buf, ag_get_recv_buf,
                                   ag_recv_trans, ag_swap_bufs);
}

void lbann_quantizer::intermodel_sum_quantized(
  lbann_comm* comm, DistMat& mat, Mat& qerror, Mat& im_qerror,
  bool do_adagrad, Mat* gradhist) {
  intermodel_sum_quantized(comm, mat.Matrix(), qerror, im_qerror, do_adagrad,
                           gradhist);
}

void lbann_quantizer::threshold_quantize(const Mat& mat, ThreshQuantized& quant,
                                         Mat& qerror, DataType pos_thresh,
                                         DataType neg_thresh, bool delta) {
  const Int ldim = mat.LDim();
  const Int width = mat.Width();
  const Int height = mat.Height();
  if (ldim != qerror.LDim()) std::cout << "ldims don't match!" << std::endl;
  const DataType* __restrict__ mat_buf = mat.LockedBuffer();
  DataType* __restrict__ qerror_buf = qerror.Buffer();
  std::vector<ThreshQuantized> thread_qs(omp_get_max_threads());
  if (delta) {
    Unsigned prev_pos = 0;
    for (Int col = 0; col < width; ++col) {
      for (Int row = 0; row < height; ++row) {
        const Unsigned pos = row + col * ldim;
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
      for (Int col = 0; col < width; ++col) {
        for (Int row = 0; row < height; ++row) {
          const Unsigned pos = row + col * ldim;
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
  DataType* __restrict__ buf = mat.Buffer();
  if (delta) {
    Unsigned prev_pos = 0;
    for (Unsigned i = 0; i < quant.size(); ++i) {
      const uqtype q = quant[i];
      const Unsigned pos = (q >> 1) + prev_pos;
      prev_pos = pos;
      if (q & 1) buf[pos] = pos_thresh;
      else buf[pos] = neg_thresh;
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (Unsigned i = 0; i < quant.size(); ++i) {
      const uqtype q = quant[i];
      const Unsigned pos = q >> 1;
      if (q & 1) buf[pos] = pos_thresh;
      else buf[pos] = neg_thresh;
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
  DataType neg_thresh, std::vector<Unsigned>& positions, bool delta) {
  // A general note on positions that I'm putting here because I'm not sure
  // where else to: Using a vector admits the possibility that we have
  // duplicate entries. This could be fixed by using an unordered_set, but when
  // I benchmarked this, it increased our runtime by ~5 times. Having duplicate
  // entries should not change the final result: it means that
  // threshold_quantize_apply may quantize the same entry multiple times, but
  // the final unquantize is not an _apply, and so will just set that entry to
  // the same value multiple times. We send some extra data, but the overhead
  // is small.
  DataType* __restrict__ buf = mat.Buffer();
  if (delta) {
    Unsigned prev_pos = 0;
    for (Unsigned i = 0; i < quant.size(); ++i) {
      const uqtype q = quant[i];
      const Unsigned pos = (q >> 1) + prev_pos;
      prev_pos = pos;
      positions.emplace_back(pos);
      if (q & 1) buf[pos] += pos_thresh;
      else buf[pos] += neg_thresh;
    }
  } else {
    for (Unsigned i = 0; i < quant.size(); ++i) {
      const uqtype q = quant[i];
      const Unsigned pos = q >> 1;
      positions.emplace_back(pos);
      if (q & 1) buf[pos] += pos_thresh;
      else buf[pos] += neg_thresh;
    }
  }
}

void lbann_quantizer::threshold_quantize_apply(
  const Mat& mat, ThreshQuantized& quant, Mat& qerror, DataType pos_thresh,
  DataType neg_thresh, std::vector<Unsigned>& positions, bool delta) {
  const DataType* __restrict__ mat_buf = mat.LockedBuffer();
  DataType* __restrict__ qerror_buf = qerror.Buffer();
  if (delta) {
    // Need to sort so positions are in order, otherwise our delta encoding
    // doesn't work. (Could be solved by adding stops, but maybe not worth it.)
    std::sort(positions.begin(), positions.end());
    Unsigned prev_pos = 0;
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
  lbann_comm* comm, Mat& mat, Mat& qerror, DataType pos_thresh,
  DataType neg_thresh, Mat& im_qerror, bool compress) {
  if (qerror.Height() == 0) {
    qerror.Resize(mat.Height(), mat.Width(), mat.LDim());
    Zero(qerror);
  }
  ThreshQuantized rs_quant;
  ThreshQuantized rs_recv;
  std::vector<Unsigned> positions;
  auto rs_send_trans = 
    [&qerror, &rs_quant, compress, pos_thresh, neg_thresh, this]
    (Mat& mat, IR h, IR w, int& count) {
      auto to_send = mat(h, w);
      auto to_send_qerr = qerror(h, w);
      rs_quant.clear();
      threshold_quantize(to_send, rs_quant, to_send_qerr, pos_thresh,
                         neg_thresh, compress);
      if (compress) {
        ThreshQuantized comp;
        compress_thresholds(rs_quant, comp);
        std::swap(rs_quant, comp);
      }
      count = rs_quant.size();
      return rs_quant.data();
    };
  auto rs_get_recv_buf = 
    [&rs_recv] (Mat& mat, int& count) {
      rs_recv.resize(count);
      return rs_recv.data();
    };
  auto rs_recv_trans = 
    [&rs_recv, &positions, compress, pos_thresh, neg_thresh, this]
    (uqtype* buf, Mat& accum) {
      if (compress) {
        ThreshQuantized uncomp;
        uncompress_thresholds(rs_recv, uncomp);
        std::swap(rs_recv, uncomp);
      }
      threshold_unquantize_apply(rs_recv, accum, pos_thresh,
                                 neg_thresh, positions, compress);
    };
  intermodel_ring_reduce_scatter<uqtype>(comm, mat, true, rs_send_trans,
                                         rs_get_recv_buf, rs_recv_trans);
  ThreshQuantized ag_send;
  ThreshQuantized ag_recv;
  auto ag_reduced_trans =
    [&im_qerror, &ag_send, &positions, compress, pos_thresh, neg_thresh, this]
    (Mat& reduced) {
      if (im_qerror.Height() == 0) {
        im_qerror.Resize(reduced.Height(), reduced.Width(), reduced.LDim());
        Zero(im_qerror);
      }
      threshold_quantize_apply(reduced, ag_send, im_qerror, pos_thresh,
                               neg_thresh, positions, compress);
      threshold_unquantize(ag_send, reduced, pos_thresh, neg_thresh, compress);
      if (compress) {
        ThreshQuantized comp;
        compress_thresholds(ag_send, comp);
        std::swap(ag_send, comp);
      }
    };
  auto ag_get_send_buf = [&ag_send] (int& count) {
      count = ag_send.size();
      return ag_send.data();
    };
  auto ag_get_recv_buf =
    [&ag_recv] (Mat& recv_view, int& count) {
      ag_recv.resize(count);
      return ag_recv.data();
    };
  auto ag_recv_trans = 
    [&ag_recv, compress, pos_thresh, neg_thresh, this]
    (uqtype*, Mat& accum) {
      if (compress) {
        ThreshQuantized uncomp;
        uncompress_thresholds(ag_recv, uncomp);
        threshold_unquantize(uncomp, accum, pos_thresh, neg_thresh, compress);
      } else {
        threshold_unquantize(ag_recv, accum, pos_thresh, neg_thresh);
      }
    };
  auto ag_swap_bufs =
    [&ag_send, &ag_recv] (uqtype*, uqtype*) {
      std::swap(ag_send, ag_recv);
    };
  intermodel_ring_allgather<uqtype>(comm, mat, true, ag_reduced_trans,
                                    ag_get_send_buf, ag_get_recv_buf,
                                    ag_recv_trans, ag_swap_bufs);
}

void lbann_quantizer::intermodel_sum_threshold_quantized(
  lbann_comm* comm, DistMat& mat, Mat& qerror, DataType pos_thresh,
  DataType neg_thresh, Mat& im_qerror, bool compress) {
  intermodel_sum_threshold_quantized(comm, mat.Matrix(), qerror, pos_thresh,
                                     neg_thresh, im_qerror, compress);
}

void lbann_quantizer::intermodel_sum_adaptive_threshold_quantized(
  lbann_comm* comm, Mat& mat, Mat& qerror, int proportion, Mat& im_qerror) {
  // Select which algorithm to use based on the size of mat.
  // Multiply at 64 bits to avoid overflows.
  size_t mat_size = ((size_t) mat.Height()) * ((size_t) mat.Width());
  // Check signed version because we need one bit for the quantized value.
  if (mat_size > std::numeric_limits<int32_t>::max()) {
    intermodel_sum_adaptive_threshold_quantized_impl<uint64_t, uint64_t>(
      comm, mat, qerror, proportion, im_qerror, adaptive_recv64_bufs1,
      adaptive_recv64_bufs2);
  } else {
    // Check whether we can use 16-bit row indices.
    if (mat.Height() > std::numeric_limits<int16_t>::max()) {
      intermodel_sum_adaptive_threshold_quantized_impl<uint32_t, uint32_t>(
        comm, mat, qerror, proportion, im_qerror, adaptive_recv32_bufs1,
        adaptive_recv32_bufs2);
    } else {
      intermodel_sum_adaptive_threshold_quantized_impl<uint32_t, uint16_t>(
        comm, mat, qerror, proportion, im_qerror, adaptive_recv16_bufs1,
        adaptive_recv16_bufs2);
    }
  }
}

void lbann_quantizer::intermodel_sum_adaptive_threshold_quantized(
  lbann_comm* comm, DistMat& mat, Mat& qerror, int proportion, Mat& im_qerror) {
  intermodel_sum_adaptive_threshold_quantized(comm, mat.Matrix(), qerror,
                                              proportion, im_qerror);
}

void lbann_quantizer::compress_thresholds(const ThreshQuantized& q,
                                          ThreshQuantized& cq) {
  compress_thresholds(q, q.begin(), q.end(), cq);
}

void lbann_quantizer::compress_thresholds(
  const ThreshQuantized& q, ThreshQuantized::const_iterator qstart,
  ThreshQuantized::const_iterator qend, ThreshQuantized& comp_q) {
  // Current bit to write to. This is in the range [0, NUM_BITS-1] (e.g. 0-31).
  uqtype cur_bit = 0;
  comp_q.emplace_back(0);
  for (auto iter = qstart; iter != qend; ++iter) {
    uqtype entry = *iter;
    uqtype quotient = entry >> GR_K;
    uqtype remainder = entry & (GR_M - 1);
    size_t cur_pos = comp_q.size() - 1;
    // quotient should usually be 0; if not, choose a better GR_K.
    if (quotient == 0) {
      // Just increment the current bit, since cur_bit <= 31 here.
      ++cur_bit;
    } else {
      // The quotient needs quotient 1s then a 0.
      // Determine how many bits we need to set in the current word.
      uqtype bits_set_cur = std::min((uqtype) NUM_BITS - cur_bit, quotient);
      // This shift is done with 64 bits to deal with the case that:
      // cur_bit == 0 && quotient == NUM_BITS => bits_set_cur = 32
      // which breaks when sizeof(uqtype) == 32 (which we use).
      // If we switch to 64 bits, we'll need to come up with something else,
      // since the same problem will occur.
      comp_q[cur_pos] |= ((((uint64_t) 1) << bits_set_cur) - 1) << cur_bit;
      // Add the appropriate number of words filled with 1s (may be 0).
      comp_q.resize(comp_q.size() + (quotient - bits_set_cur) / NUM_BITS,
                    ~((uqtype) 0));
      // Add the final bits if any and update cur_bit.
      uqtype final_bits = (quotient - bits_set_cur) & (NUM_BITS - 1);
      comp_q.resize(comp_q.size() + (final_bits > 0), (1 << final_bits) - 1);
      cur_bit = (cur_bit + quotient) & (NUM_BITS - 1);
      // Add the 0 terminator.
      comp_q.resize(comp_q.size() + !cur_bit, 0);
      ++cur_bit;
    }
    // Write remainder using GR_K bits. cur_bit == NUM_BITS is possible here.
    uqtype bits_set_cur = std::min((uqtype) NUM_BITS - cur_bit,
                                   static_cast<uqtype>(GR_K));
    cur_pos = comp_q.size() - 1;
    // Write what we can to the current word.
    comp_q[cur_pos] |= (remainder & ((1 << bits_set_cur) - 1)) << cur_bit;
    // Write the rest to a new word, if needed.
    comp_q.resize(comp_q.size() + (bits_set_cur != GR_K),
                  remainder >> bits_set_cur);
    cur_bit = (cur_bit + GR_K) & (NUM_BITS - 1);
    // Add a new word if needed.
    comp_q.resize(comp_q.size() + !cur_bit, 0);
  }
  // Pad the final word with 1s to terminate.
  size_t cur_pos = comp_q.size() - 1;
  comp_q[cur_pos] |= ((1 << (NUM_BITS - cur_bit)) - 1) << cur_bit;
}

void lbann_quantizer::uncompress_thresholds(const ThreshQuantized& cq,
                                            ThreshQuantized& q) {
  uncompress_thresholds(cq, cq.begin(), cq.end(), q);
}

void lbann_quantizer::uncompress_thresholds(
  const ThreshQuantized& cq, ThreshQuantized::const_iterator cqstart,
  ThreshQuantized::const_iterator cqend, ThreshQuantized& q) {
  uqtype quotient = 0;
  uqtype remainder = 0;
  uqtype cur_bit = 0;
  uqtype cur = *cqstart;
  for (auto iter = cqstart; iter != cqend;) {
    // Decode the quotient by continuing until a 0 is found.
    // TODO: Need to switch this depending on the size of El::Int.
    // __builtin_ffs expects 32-bit ints; need __builtin_ffsll for 64-bit.
    int ffz;
    while ((ffz = __builtin_ffs(~cur)) == 0) {
      ++iter;
      if (iter == cqend) {
        return;
      }
      cur = *iter;
      quotient += NUM_BITS - cur_bit;
      cur_bit = 0;
    }
    quotient += ffz - 1 - cur_bit;
    cur_bit = ffz;
    // Decode the remainder (GR_K bits).
    uqtype bits_left = std::min((uqtype) NUM_BITS - cur_bit,
                                static_cast<uqtype>(GR_K));
    remainder = (cur >> cur_bit) & ((1 << bits_left) - 1);
    if (bits_left != GR_K) {
      ++iter;
      cur = *iter;
    }
    remainder |= (cur & ((1 << (GR_K - bits_left)) - 1)) << bits_left;
    cur_bit = (cur_bit + GR_K) & (NUM_BITS - 1);
    // Decode the final value.
    q.emplace_back(quotient * GR_M + remainder);
    quotient = 0;
    remainder = 0;
    // Advance to the next entry if needed.
    iter += !cur_bit;
    if (!cur_bit && iter != cqend) cur = *iter;
    // Fill everything before the current bit with 1s to avoid confusing the
    // quotient calculation.
    cur |= (1 << cur_bit) - 1;
  }
}

lbann_quantizer::adaptive_thresholds lbann_quantizer::proportion_threshold(
  const Mat& mat, const Mat& qerror, int proportion, bool sample) {
  double proportion_start = get_time();
  std::vector<DataType> entries;
  const Int height = mat.Height();
  const Int width = mat.Width();
  const Int ldim = mat.LDim();
  const DataType* __restrict__ mat_buf = mat.LockedBuffer();
  const DataType* __restrict__ qerror_buf = qerror.LockedBuffer();
  // Bail out if needed.
  if (width == 0) {
    return { 0.0f, 0.0f };
  }
  if (width * height <= NUM_THRESHOLD_SAMPLES || !sample) {
    // Copy entire matrix into vector.
    entries.reserve(width * height);
    for (Int col = 0; col < width; ++col) {
      const Int col_offset = col * ldim;
      for (Int row = 0; row < height; ++row) {
        const Unsigned pos = row + col_offset;
        entries.emplace_back(mat_buf[pos] + qerror_buf[pos]);
      }
    }
  } else {
    // Randomly sample entries to approximate everything.
    entries.reserve(NUM_THRESHOLD_SAMPLES);
    fast_rng_gen& gen = get_fast_generator();
    std::vector<Unsigned> poses(NUM_THRESHOLD_SAMPLES);
    for (Unsigned i = 0; i < NUM_THRESHOLD_SAMPLES; ++i) {
      const Unsigned pos = fast_rand_int(gen, height) + fast_rand_int(gen, width) * ldim;
      __builtin_prefetch(&mat_buf[pos]);
      __builtin_prefetch(&qerror_buf[pos]);
      poses[i] = pos;
    }
    for (Unsigned i = 0; i < NUM_THRESHOLD_SAMPLES; ++i) {
      const Unsigned pos = poses[i];
      entries.emplace_back(mat_buf[pos] + qerror_buf[pos]);
    }
  }
  // Determine the number of entries to keep.
  Int num_to_keep = std::max(1, (int) entries.size() / proportion);
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
      if (*i < 0) neg_thresh = std::max(neg_thresh, *i);
    }
  } else if (*i < 0) {
    neg_thresh = *i;
    for (++i; i < entries.end(); ++i) {
      // Find the smallest (closest to 0) positive value.
      if (*i > 0) pos_thresh = std::min(pos_thresh, *i);
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
  const Mat& mat, const Mat& qerror, Int col,
  const adaptive_thresholds threshes, bool sample) {
  DataType pos_sum = 0.0f;
  Unsigned pos_count = 0;
  DataType neg_sum = 0.0f;
  Unsigned neg_count = 0;
#if LBANN_QUANTIZER_TERNARY
  DataType zero_sum = 0.0f;
  Unsigned zero_count = 0;
#endif
  const Int height = mat.Height();
  const Int col_offset = col * mat.LDim();
  const DataType* __restrict__ mat_buf = mat.LockedBuffer();
  const DataType* __restrict__ qerror_buf = qerror.LockedBuffer();
  if (height <= NUM_RECON_SAMPLES || !sample) {
    for (Int row = 0; row < height; ++row) {
      const Unsigned pos = row + col_offset;
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
    std::vector<Unsigned> poses(NUM_RECON_SAMPLES);
    for (Unsigned i = 0; i < NUM_RECON_SAMPLES; ++i) {
      const Unsigned pos = fast_rand_int(gen, height) + col_offset;
      __builtin_prefetch(&mat_buf[pos]);
      __builtin_prefetch(&qerror_buf[pos]);
      poses[i] = pos;
    }
    for (Unsigned i = 0; i < NUM_RECON_SAMPLES; ++i) {
      //const unsigned pos = row_dist(gen) + col_offset;
      const Unsigned pos = poses[i];
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
