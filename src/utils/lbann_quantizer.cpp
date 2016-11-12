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
// lbann_quantizer .hpp .cpp - One-bit quantization of matrices
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
}

lbann_quantizer::~lbann_quantizer() {

}

void lbann_quantizer::quantize(
  const Mat& mat, QuantizedMatrix& qmat, Mat& qerror, bool sample) {
  // Set up the quantized matrix. (+2 for the averages.)
  const int qheight = get_quantized_matrix_height(mat);
  const int qwidth = mat.Width();
  qmat.Resize(qheight, qwidth);

  const Int width = mat.Width();
  const Int height = mat.Height();
  const Int ldim = mat.LDim();
  const Int qmat_ldim = qmat.LDim();
  const DataType* __restrict__ mat_buf = mat.LockedBuffer();
  DataType* __restrict__ qerror_buf = qerror.Buffer();
  qtype* __restrict__ qmat_buf = qmat.Buffer();
  #pragma omp parallel for schedule(static)
  for (int col = 0; col < width; ++col) {
    // First compute the positive and negative column averages.
    DataType pos_sum = 0.0f;
    DataType neg_sum = 0.0f;
    size_t num_pos = 0;
    size_t num_neg = 0;
    if (height <= NUM_ONEBIT_SAMPLES || !sample) {
      for (int row = 0; row < height; ++row) {
        const int pos = row + col * ldim;
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
      std::uniform_int_distribution<int> row_dist(0, height - 1);
      rng_gen& gen = get_generator();
      for (unsigned i = 0; i < NUM_ONEBIT_SAMPLES; ++i) {
        const unsigned pos = row_dist(gen) + col * ldim;
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
    qtype tmp;
    memcpy(&tmp, &avg_pos, sizeof(avg_pos));
    qmat.Set(0, col, tmp);
    memcpy(&tmp, &avg_neg, sizeof(avg_neg));
    qmat.Set(1, col, tmp);

    // Now quantize the column, NUM_BITS entries at a time.
    int qrow = 2;
    for (int row_chunk = 0; row_chunk < height; row_chunk += NUM_BITS) {
      uqtype q = 0;
      for (unsigned bit = 0; bit < NUM_BITS; ++bit) {
        int row = row_chunk + bit;
        if (row >= height) {
          break;
        }
        const int pos = row + col * ldim;
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

void lbann_quantizer::unquantize(const QuantizedMatrix& qmat, Mat& mat, bool apply) {
  const Int width = mat.Width();
  const Int height = mat.Height();
  const Int ldim = mat.LDim();
  const Int qmat_ldim = qmat.LDim();
  const qtype* __restrict__ qmat_buf = qmat.LockedBuffer();
  DataType* __restrict__ mat_buf = mat.Buffer();
  #pragma omp parallel for schedule(static)
  for (int col = 0; col < width; ++col) {
    int qrow = 2;
    // Extract the averages.
    qtype tmp = qmat.Get(0, col);
    DataType avg_pos;
    memcpy(&avg_pos, &tmp, sizeof(avg_pos));
    tmp = qmat.Get(1, col);
    DataType avg_neg;
    memcpy(&avg_neg, &tmp, sizeof(avg_neg));
    // Unquantize this column.
    for (int row_chunk = 0; row_chunk < height; row_chunk += NUM_BITS) {
      uqtype q = (uqtype) qmat_buf[qrow + col * qmat_ldim];
      for (size_t bit = 0; bit < NUM_BITS; ++bit) {
        int row = row_chunk + bit;
        if (row >= height) {
          break;
        }
        if (apply) {
          mat_buf[row + col * ldim] += (q >> bit) & 0x1 ? avg_pos : avg_neg;
        } else {
          mat_buf[row + col * ldim] = (q >> bit) & 0x1 ? avg_pos : avg_neg;
        }
      }
      ++qrow;
    }
  }
}

void lbann_quantizer::unquantize(const QuantizedMatrix& qmat, DistMat& mat,
                                 bool apply) {
  unquantize(qmat, mat.Matrix(), apply);
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
      unquantize(rs_recv, accum, true);
    };
  intermodel_ring_reduce_scatter<qtype>(comm, mat, false, rs_send_trans,
                                        rs_get_recv_buf, rs_recv_trans);
  QuantizedMatrix ag_send;
  QuantizedMatrix ag_recv;
  std::function<DataType(DataType)> _sq = [](DataType x) { return x*x; };
  std::function<DataType(DataType)> _sqrt =
    [](DataType x) { return 1.0f / (std::sqrt(x) + 1e-8f); };
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

void lbann_quantizer::intermodel_sum_quantized2(lbann_comm* comm, Mat& mat_,
                                                Mat& qerror_, Mat& im_qerror) {
  if (qerror_.Height() == 0) {
    Zeros(qerror_, mat_.Height(), mat_.Width());
  }
  Mat mat;
  Transpose(mat_, mat);
  Mat qerror;
  Transpose(qerror_, qerror);
  QuantizedMatrix qmat;
  quantize(mat, qmat, qerror);
  QuantizedMatrix recv_qmat;
  recv_qmat.Resize(qmat.Height(), qmat.Width());
  for (int i = 0; i < comm->get_num_models(); ++i) {
    if (i == comm->get_model_rank()) {
      for (int dst = 0; dst < comm->get_num_models(); ++dst) {
        if (dst != comm->get_model_rank()) {
          comm->send(qmat.Buffer(), qmat.Width() * qmat.Height(), dst);
        }
      }
    } else {
      comm->recv(recv_qmat.Buffer(), recv_qmat.Width() * recv_qmat.Height(), i);
      Mat uqmat;
      uqmat.Resize(mat.Height(), mat.Width());
      unquantize(recv_qmat, uqmat);
      mat += uqmat;
    }
  }
  Transpose(mat, mat_);
  Transpose(qerror, qerror_);
}

void lbann_quantizer::intermodel_sum_quantized2(lbann_comm* comm, DistMat& mat,
                                                Mat& qerror, Mat& im_qerror) {
  intermodel_sum_quantized2(comm, mat.Matrix(), qerror, im_qerror);
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
    unsigned prev_pos = 0;
    for (int col = 0; col < width; ++col) {
      for (int row = 0; row < height; ++row) {
        const unsigned pos = row + col * ldim;
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
      for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
          const unsigned pos = row + col * ldim;
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
    unsigned prev_pos = 0;
    for (unsigned i = 0; i < quant.size(); ++i) {
      const uqtype q = quant[i];
      const unsigned pos = (q >> 1) + prev_pos;
      prev_pos = pos;
      if (q & 1) buf[pos] = pos_thresh;
      else buf[pos] = neg_thresh;
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < quant.size(); ++i) {
      const uqtype q = quant[i];
      const unsigned pos = q >> 1;
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
  DataType neg_thresh, std::vector<unsigned>& positions, bool delta) {
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
    unsigned prev_pos = 0;
    for (unsigned i = 0; i < quant.size(); ++i) {
      const uqtype q = quant[i];
      const unsigned pos = (q >> 1) + prev_pos;
      prev_pos = pos;
      positions.emplace_back(pos);
      if (q & 1) buf[pos] += pos_thresh;
      else buf[pos] += neg_thresh;
    }
  } else {
    for (unsigned i = 0; i < quant.size(); ++i) {
      const uqtype q = quant[i];
      const unsigned pos = q >> 1;
      positions.emplace_back(pos);
      if (q & 1) buf[pos] += pos_thresh;
      else buf[pos] += neg_thresh;
    }
  }
}

void lbann_quantizer::threshold_quantize_apply(
  const Mat& mat, ThreshQuantized& quant, Mat& qerror, DataType pos_thresh,
  DataType neg_thresh, std::vector<unsigned>& positions, bool delta) {
  const DataType* __restrict__ mat_buf = mat.LockedBuffer();
  DataType* __restrict__ qerror_buf = qerror.Buffer();
  if (delta) {
    // Need to sort so positions are in order, otherwise our delta encoding
    // doesn't work. (Could be solved by adding stops, but maybe not worth it.)
    std::sort(positions.begin(), positions.end());
    unsigned prev_pos = 0;
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

void lbann_quantizer::adaptive_threshold_quantize(
  const Mat& mat, ThreshQuantized& q, Mat& qerror, int proportion,
  bool compress) {
  // This uses a header to store all information needed to do unquantization in
  // one spot, which makes unquantization easier to multi-thread. The header has
  // one entry for each column, consisting of the starting offset of the
  // quantized data in the array (including the header) and the three
  // reconstruction values. The number of quantized entries is included as a
  // final entry to simplify unquantization.
  const Int width = mat.Width();
  const Int height = mat.Height();
  const Int ldim = mat.LDim();
  const DataType* __restrict__ mat_buf = mat.LockedBuffer();
  DataType* __restrict__ qerror_buf = qerror.Buffer();
  const int header_len = 4 * width + 1;
  q.resize(header_len);  // Space for the header.
  std::vector<ThreshQuantized> thread_qs(omp_get_max_threads());
  std::vector<unsigned> quantized_sums(omp_get_max_threads(), 0);
  if (!compress) {
    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      int num_quantized = header_len;
      #pragma omp for schedule(static)
      for (int col = 0; col < width; ++col) {
        const int header_loc = 4 * col;
        q[header_loc] = num_quantized;
        adaptive_info ainfo = 
          proportion_threshold_average(mat, qerror, proportion, col);
        // Store the averages for reconstruction.
        memcpy(&q[header_loc + 1], &ainfo.pos_avg, sizeof(ainfo.pos_avg));
        memcpy(&q[header_loc + 2], &ainfo.neg_avg, sizeof(ainfo.neg_avg));
        memcpy(&q[header_loc + 3], &ainfo.zero_avg, sizeof(ainfo.zero_avg));
        for (int row = 0; row < height; ++row) {
          const unsigned pos = row + col * ldim;
          const DataType val = mat_buf[pos] + qerror_buf[pos];
          if (val >= ainfo.pos_thresh) {
            qerror_buf[pos] = val - ainfo.pos_avg;
            thread_qs[tid].emplace_back((pos << 1) | 1);
            ++num_quantized;
          } else if (val <= ainfo.neg_thresh) {
            qerror_buf[pos] = val - ainfo.neg_avg;
            thread_qs[tid].emplace_back(pos << 1);
            ++num_quantized;
          } else {
            qerror_buf[pos] = val - ainfo.zero_avg;
          }
        }
      }
      #pragma omp single
      {
        // Compute the amount to adjust header counts by. This is essentially
        // a shifted prefix-sum.
        quantized_sums[0] = 0;
        for (int t = 1; t < omp_get_max_threads(); ++t) {
          quantized_sums[t] = quantized_sums[t - 1] + thread_qs[t - 1].size();
        }
      }
      // Have threads patch up the header counts.
      // Static schedule guarantees threads are assigned the same way.
      #pragma omp for schedule(static)
      for (int col = 0; col < width; ++col) {
        q[4 * col] += quantized_sums[tid];
      }
    }
    for (auto&& thread_q : thread_qs) {
      q.insert(q.end(), thread_q.begin(), thread_q.end());
    }
    // Store the final number of entries.
    q[4 * width] = q.size();
  } else {
    // Delta encoding is done within each column:
    // only the difference between rows is recorded.
    #pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      int start_offset = header_len;
      #pragma omp for schedule(static)
      for (int col = 0; col < width; ++col) {
        const int header_loc = 4 * col;
        ThreshQuantized uncomp;
        q[header_loc] = start_offset;
        DataType pos_thresh, neg_thresh, pos_avg, neg_avg;
        adaptive_info ainfo =
          proportion_threshold_average(mat, qerror, proportion, col);
        // Store the averages for reconstruction.
        memcpy(&q[header_loc + 1], &ainfo.pos_avg, sizeof(ainfo.pos_avg));
        memcpy(&q[header_loc + 2], &ainfo.neg_avg, sizeof(ainfo.neg_avg));
        memcpy(&q[header_loc + 3], &ainfo.zero_avg, sizeof(ainfo.zero_avg));
        unsigned prev_row = 0;
        for (int row = 0; row < height; ++row) {
          const unsigned pos = row + col * ldim;
          const DataType val = mat_buf[pos] + qerror_buf[pos];
          if (val >= ainfo.pos_thresh) {
            qerror_buf[pos] = val - ainfo.pos_avg;
            uncomp.emplace_back(((row - prev_row) << 1) | 1);
            prev_row = row;
          } else if (val <= ainfo.neg_thresh) {
            qerror_buf[pos] = val - ainfo.neg_avg;
            uncomp.emplace_back((row - prev_row) << 1);
            prev_row = row;
          } else {
            qerror_buf[pos] = val - ainfo.zero_avg;
          }
        }
        // Compress and store into the global thread-local array.
        compress_thresholds(uncomp, uncomp.begin(), uncomp.end(),
                            thread_qs[tid]);
        start_offset = header_len + thread_qs[tid].size();
      }
      #pragma omp single
      {
        // Compute the amount to adjust header counts by. This is essentially
        // a shifted prefix-sum.
        quantized_sums[0] = 0;
        for (int t = 1; t < omp_get_max_threads(); ++t) {
          quantized_sums[t] = quantized_sums[t - 1] + thread_qs[t - 1].size();
        }
      }
      // Have threads patch up the header counts.
      // Static schedule guarantees threads are assigned the same way.
      #pragma omp for schedule(static)
      for (int col = 0; col < width; ++col) {
        q[4 * col] += quantized_sums[tid];
      }
    }
    for (auto&& thread_q : thread_qs) {
      q.insert(q.end(), thread_q.begin(), thread_q.end());
    }
    // Store the final number of entries.
    q[4 * width] = q.size();
  }
}

void lbann_quantizer::adaptive_threshold_quantize(
  const DistMat& mat, ThreshQuantized& q, Mat& qerror, int proportion,
  bool compress) {
  adaptive_threshold_quantize(mat.LockedMatrix(), q, qerror, proportion,
                              compress);
}

void lbann_quantizer::adaptive_threshold_unquantize(
  const ThreshQuantized& q, Mat& mat, bool compress) {
  DataType* __restrict__ buf = mat.Buffer();
  const unsigned header_len = mat.Width() * 4;
  const Int height = mat.Height();
  const Int ldim = mat.LDim();
  if (!compress) {
    #pragma omp parallel for schedule(static)
    for (unsigned header_loc = 0; header_loc < header_len; header_loc += 4) {
      // Extract averages.
      DataType pos_avg, neg_avg, zero_avg;
      memcpy(&pos_avg, &q[header_loc + 1], sizeof(pos_avg));
      memcpy(&neg_avg, &q[header_loc + 2], sizeof(neg_avg));
      memcpy(&zero_avg, &q[header_loc + 3], sizeof(zero_avg));
      // Fill the column, then update with the other values.
      std::fill_n(&buf[(header_loc / 4) * ldim], height, zero_avg);
      for (unsigned i = q[header_loc]; i < q[header_loc + 4]; ++i) {
        const uqtype val = q[i];
        const unsigned pos = val >> 1;
        if (val & 1) buf[pos] = pos_avg;
        else buf[pos] = neg_avg;
      }
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (unsigned header_loc = 0; header_loc < header_len; header_loc += 4) {
      // Extract averages.
      DataType pos_avg, neg_avg, zero_avg;
      memcpy(&pos_avg, &q[header_loc + 1], sizeof(pos_avg));
      memcpy(&neg_avg, &q[header_loc + 2], sizeof(neg_avg));
      memcpy(&zero_avg, &q[header_loc + 3], sizeof(zero_avg));
      // Uncompress the range.
      ThreshQuantized uncomp;
      uncompress_thresholds(q, q.begin() + q[header_loc],
                            q.begin() + q[header_loc + 4], uncomp);
      unsigned prev_row = 0;
      const unsigned col_offset = (header_loc / 4) * ldim;
      // Fill the column, then update with the other values.
      std::fill_n(&buf[(header_loc / 4) * ldim], height, zero_avg);
      for (unsigned i = 0; i < uncomp.size(); ++i) {
        const uqtype val = uncomp[i];
        const unsigned row = (val >> 1) + prev_row;
        // header_loc/4 is the same as the column.
        const unsigned pos = row + col_offset;
        prev_row = row;
        if (val & 1) buf[pos] = pos_avg;
        else buf[pos] = neg_avg;
      }
    }
  }
}

void lbann_quantizer::adaptive_threshold_unquantize(
  const ThreshQuantized& q, DistMat& mat, bool compress) {
  adaptive_threshold_unquantize(q, mat.Matrix(), compress);
}

void lbann_quantizer::adaptive_threshold_unquantize_add(
  const ThreshQuantized& q, Mat& mat, bool compress) {
  DataType* __restrict__ buf = mat.Buffer();
  const unsigned header_len = mat.Width() * 4;
  const Int height = mat.Height();
  const Int ldim = mat.LDim();
  if (!compress) {
    #pragma omp parallel for schedule(static)
    for (unsigned header_loc = 0; header_loc < header_len; header_loc += 4) {
      // Extract averages.
      DataType pos_avg, neg_avg, zero_avg;;
      memcpy(&pos_avg, &q[header_loc + 1], sizeof(pos_avg));
      memcpy(&neg_avg, &q[header_loc + 2], sizeof(neg_avg));
      memcpy(&zero_avg, &q[header_loc + 3], sizeof(zero_avg));
      // Add zero_avg to everything and adjust the other means.
      const unsigned col_offset = (header_loc / 4) * ldim;
      for (unsigned row = 0; row < height; ++row) {
        buf[row + col_offset] += zero_avg;
      }
      pos_avg -= zero_avg;
      neg_avg += zero_avg;
      for (unsigned i = q[header_loc]; i < q[header_loc + 4]; ++i) {
        const uqtype val = q[i];
        const unsigned pos = val >> 1;
        if (val & 1) buf[pos] += pos_avg;
        else buf[pos] += neg_avg;
      }
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (unsigned header_loc = 0; header_loc < header_len; header_loc += 4) {
      // Extract averages.
      DataType pos_avg, neg_avg, zero_avg;
      memcpy(&pos_avg, &q[header_loc + 1], sizeof(pos_avg));
      memcpy(&neg_avg, &q[header_loc + 2], sizeof(neg_avg));
      memcpy(&zero_avg, &q[header_loc + 3], sizeof(zero_avg));
      // Uncompress the range.
      ThreshQuantized uncomp;
      uncompress_thresholds(q, q.begin() + q[header_loc],
                            q.begin() + q[header_loc + 4], uncomp);
      unsigned prev_row = 0;
      const unsigned col_offset = (header_loc / 4) * ldim;
      // Add zero_avg to everything and adjust the other means.
      for (unsigned row = 0; row < height; ++row) {
        buf[row + col_offset] += zero_avg;
      }
      pos_avg -= zero_avg;
      neg_avg += zero_avg;
      for (unsigned i = 0; i < uncomp.size(); ++i) {
        const uqtype val = uncomp[i];
        const unsigned row = (val >> 1) + prev_row;
        // header_loc/4 is the same as the column.
        const unsigned pos = row + col_offset;
        prev_row = row;
        if (val & 1) buf[pos] += pos_avg;
        else buf[pos] += neg_avg;
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
  std::vector<unsigned> positions;
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
  lbann_comm* comm, Mat& mat, Mat& qerror, int proportion, Mat& im_qerror,
  bool compress) {
  if (qerror.Height() == 0) {
    qerror.Resize(mat.Height(), mat.Width(), mat.LDim());
    Zero(qerror);
  }
  ThreshQuantized rs_quant;
  ThreshQuantized rs_recv;
  ThreshQuantized comp_buf;
  ThreshQuantized uncomp_buf;
  auto rs_send_trans = 
    [&qerror, &rs_quant, compress, proportion, this]
    (Mat& mat, IR h, IR w, int& count) {
      auto to_send = mat(h, w);
      auto to_send_qerr = qerror(h, w);
      rs_quant.clear();
      adaptive_threshold_quantize(to_send, rs_quant, to_send_qerr, proportion,
                                  compress);
      count = rs_quant.size();
      return rs_quant.data();
    };
  auto rs_get_recv_buf = 
    [&rs_recv] (Mat& mat, int& count) {
      rs_recv.resize(count);
      return rs_recv.data();
    };
  auto rs_recv_trans = 
    [&rs_recv, compress, this]
    (uqtype* buf, Mat& accum) {
      adaptive_threshold_unquantize_add(rs_recv, accum, compress);
    };
  intermodel_ring_reduce_scatter<uqtype>(comm, mat, true, rs_send_trans,
                                         rs_get_recv_buf, rs_recv_trans);
  ThreshQuantized ag_send;
  ThreshQuantized ag_recv;
  auto ag_reduced_trans =
    [&im_qerror, &ag_send, compress, proportion, this]
    (Mat& reduced) {
      if (im_qerror.Height() == 0) {
        im_qerror.Resize(reduced.Height(), reduced.Width(), reduced.LDim());
        Zero(im_qerror);
      }
      adaptive_threshold_quantize(reduced, ag_send, im_qerror, proportion,
                                  compress);
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
    [&ag_recv, compress, proportion, this]
    (uqtype*, Mat& accum) {
      adaptive_threshold_unquantize(ag_recv, accum, compress);
    };
  auto ag_swap_bufs =
    [&ag_send, &ag_recv] (uqtype*, uqtype*) {
      std::swap(ag_send, ag_recv);
    };
  intermodel_ring_allgather<uqtype>(comm, mat, true, ag_reduced_trans,
                                    ag_get_send_buf, ag_get_recv_buf,
                                    ag_recv_trans, ag_swap_bufs);
}

void lbann_quantizer::intermodel_sum_adaptive_threshold_quantized(
  lbann_comm* comm, DistMat& mat, Mat& qerror, int proportion, Mat& im_qerror,
  bool compress) {
  intermodel_sum_adaptive_threshold_quantized(comm, mat.Matrix(), qerror,
                                              proportion, im_qerror, compress);
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

lbann_quantizer::adaptive_info lbann_quantizer::proportion_threshold_average(
  const Mat& mat, const Mat& qerror, int proportion, int col, bool sample) {
  double pta_start = get_time();
  std::vector<DataType> pos_entries;
  std::vector<DataType> neg_entries;
  const Int height = mat.Height();
  const Int col_offset = col * mat.LDim();
  const DataType* __restrict__ mat_buf = mat.LockedBuffer();
  const DataType* __restrict__ qerror_buf = qerror.LockedBuffer();
  if (height <= NUM_PTA_SAMPLES || !sample) {
    for (int row = 0; row < height; ++row) {
      const unsigned pos = row + col_offset;
      const DataType val = mat_buf[pos] + qerror_buf[pos];
      if (val >= 0.0f) {
        pos_entries.emplace_back(val);
      } else {
        neg_entries.emplace_back(val);
      }
    }
  } else {
    // Randomly sample NUM_PTA_SAMPLES entries and use these to approximate
    // everything.
    std::uniform_int_distribution<int> row_dist(0, height - 1);
    rng_gen& gen = get_generator();
    for (unsigned i = 0; i < NUM_PTA_SAMPLES; ++i) {
      const unsigned pos = row_dist(gen) + col_offset;
      const DataType val = mat_buf[pos] + qerror_buf[pos];
      if (val >= 0.0f) {
        pos_entries.emplace_back(val);
      } else {
        neg_entries.emplace_back(val);
      }
    }
  }
  // Determine how many positive/negative entries we need to keep.
  int pos_to_keep = pos_entries.size() / proportion;
  if (pos_to_keep == 0) {
    pos_to_keep = 1;
  }
  int neg_to_keep = neg_entries.size() / proportion;
  if (neg_to_keep == 0) {
    neg_to_keep = 1;
  }
  // Determine the threshold value with a selection algorithm to keep only the
  // largest pos/neg_to_keep elements.
  // Set to 0 if there's none.
  // The partitioning also guarantees everything after the i'th element is
  // greater than or equal to it.
  DataType pos_thresh = 0.0f;
  DataType neg_thresh = 0.0f;
  DataType pos_avg = 0.0f;
  DataType neg_avg = 0.0f;
  DataType zero_avg = 0.0f;
  if (pos_to_keep > 0 && pos_entries.size() > 0) {
    auto i = pos_entries.begin() + (pos_entries.size() - pos_to_keep);
    std::nth_element(pos_entries.begin(), i, pos_entries.end());
    pos_thresh = *i;
    pos_avg = std::accumulate(i, pos_entries.end(), 0.0f) / pos_to_keep;
    zero_avg = std::accumulate(pos_entries.begin(), i, 0.0f);
  }
  if (neg_to_keep > 0 && neg_entries.size() > 0) {
    auto i = neg_entries.begin() + neg_to_keep - 1;
    std::nth_element(neg_entries.begin(), i, neg_entries.end());
    neg_thresh = *i;
    neg_avg = std::accumulate(neg_entries.begin(), i + 1, 0.0f) / neg_to_keep;
    zero_avg += std::accumulate(i + 1, neg_entries.end(), 0.0f);
  }
  zero_avg /= pos_entries.size() + neg_entries.size() -
    pos_to_keep - neg_to_keep;
  pta_time += get_time() - pta_start;
  return { pos_thresh, neg_thresh, pos_avg, neg_avg, zero_avg };
}

}  // namespace lbann
