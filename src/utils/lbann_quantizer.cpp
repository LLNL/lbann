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
#include <cmath>

namespace lbann {

lbann_quantizer::lbann_quantizer() {
  reset_bytes_counters();
  reset_time_counters();
}

lbann_quantizer::~lbann_quantizer() {

}

void lbann_quantizer::quantize(const Mat& mat, QuantizedMatrix& qmat, Mat& qerror) {
  // Set up the quantized matrix. (+2 for the averages.)
  size_t qheight = get_quantized_matrix_height(mat);
  size_t qwidth = mat.Width();
  qmat.Resize(qheight, qwidth);

  for (int col = 0; col < mat.Width(); ++col) {
    // First compute the positive and negative column averages.
    DataType pos_sum = 0.0f;
    DataType neg_sum = 0.0f;
    size_t num_pos = 0;
    size_t num_neg = 0;
    for (int row = 0; row < mat.Height(); ++row) {
      DataType val = mat.Get(row, col) + qerror.Get(row, col);
      if (val >= 0.0f) {
        pos_sum += val;
        ++num_pos;
      } else {
        neg_sum += val;
        ++num_neg;
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
    int row_chunk = 0;
    int qrow = 2;
    for (; row_chunk < mat.Height(); row_chunk += NUM_BITS) {
      uqtype q = 0;
      for (size_t bit = 0; bit < NUM_BITS; ++bit) {
        int row = row_chunk + bit;
        if (row >= mat.Height()) {
          break;
        }
        DataType val = mat.Get(row, col) + qerror.Get(row, col);
        if (val >= 0.0f) {
          q |= 1 << bit;
          qerror.Set(row, col, val - avg_pos);
        } else {
          qerror.Set(row, col, val - avg_neg);
        }
      }
      qmat.Set(qrow, col, (qtype) q);
      ++qrow;
    }
  }
}

void lbann_quantizer::quantize(const DistMat& mat, QuantizedMatrix& qmat,
                               Mat& qerror) {
  quantize(mat.LockedMatrix(), qmat, qerror);
}

void lbann_quantizer::unquantize(const QuantizedMatrix& qmat, Mat& mat, bool apply) {
  for (int col = 0; col < mat.Width(); ++col) {
    int row_chunk = 0;
    int qrow = 2;
    // Extract the averages.
    qtype tmp = qmat.Get(0, col);
    DataType avg_pos;
    memcpy(&avg_pos, &tmp, sizeof(avg_pos));
    tmp = qmat.Get(1, col);
    DataType avg_neg;
    memcpy(&avg_neg, &tmp, sizeof(avg_neg));
    // Unquantize this column.
    for (; row_chunk < mat.Height(); row_chunk += NUM_BITS) {
      uqtype q = (uqtype) qmat.Get(qrow, col);
      for (size_t bit = 0; bit < NUM_BITS; ++bit) {
        int row = row_chunk + bit;
        if (row >= mat.Height()) {
          break;
        }
        if (apply) {
          mat.Update(row, col, (q >> bit) & 0x1 ? avg_pos : avg_neg);
        } else {
          mat.Set(row, col, (q >> bit) & 0x1 ? avg_pos : avg_neg);
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
  lbann_comm* comm, Mat& mat_, Mat& qerror_, Mat& im_qerror,
  bool do_adagrad, Mat* gradhist) {
  // Local transpose so that we quantize each neuron instead of each feature.
  Mat mat;
  Transpose(mat_, mat);
  Mat qerror;
  Transpose(qerror_, qerror);
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
        Zeros(im_qerror, reduced.Height(), reduced.Width());
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
  Transpose(mat, mat_);
  Transpose(qerror, qerror_);
}

void lbann_quantizer::intermodel_sum_quantized(
  lbann_comm* comm, DistMat& mat, Mat& qerror, Mat& im_qerror,
  bool do_adagrad, Mat* gradhist) {
  intermodel_sum_quantized(comm, mat.Matrix(), qerror, im_qerror, do_adagrad,
                           gradhist);
}

void lbann_quantizer::intermodel_sum_quantized2(lbann_comm* comm, Mat& mat_,
                                                Mat& qerror_, Mat& im_qerror) {
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
                                         DataType neg_thresh, DataType pos_avg,
                                         DataType neg_avg) {
  if (pos_avg == 0.0f) {
    pos_avg = pos_thresh;
  }
  if (neg_avg == 0.0f) {
    neg_avg = neg_thresh;
  }
  const Int ldim = mat.LDim();
  size_t prev_pos = 0;
  for (int col = 0; col < mat.Width(); ++col) {
    for (int row = 0; row < mat.Height(); ++row) {
      DataType val = mat.Get(row, col) + qerror.Get(row, col);
      size_t pos = row + col * ldim;
      if (val >= pos_thresh) {
        // Delta encoding.
        pos -= prev_pos;
        prev_pos += pos;
        uqtype q = pos << 1;
        q |= 1;
        qerror.Set(row, col, val - pos_avg);
        quant.push_back(q);
      } else if (val <= neg_thresh) {
        pos -= prev_pos;
        prev_pos += pos;
        uqtype q = pos << 1;
        qerror.Set(row, col, val - neg_avg);
        quant.push_back(q);
      } else {
        qerror.Set(row, col, val);
      }
    }
  }
}

void lbann_quantizer::threshold_quantize(const DistMat& mat, ThreshQuantized& q,
                                         Mat& qerror, DataType pos_thresh,
                                         DataType neg_thresh, DataType pos_avg,
                                         DataType neg_avg) {
  threshold_quantize(mat.LockedMatrix(), q, qerror, pos_thresh, neg_thresh,
                     pos_avg, neg_avg);
}

void lbann_quantizer::threshold_unquantize(const ThreshQuantized& quant, Mat& mat,
                                           DataType pos_avg, DataType neg_avg,
                                           bool apply) {
  threshold_unquantize(quant, quant.begin(), mat, pos_avg, neg_avg, apply);
}

void lbann_quantizer::threshold_unquantize(
  const ThreshQuantized& quant, ThreshQuantized::const_iterator quant_start,
  Mat& mat, DataType pos_avg, DataType neg_avg, bool apply) {
  DataType* buf = mat.Buffer();
  if (std::distance(quant_start, quant.end()) == 0) return;
  size_t prev_pos = 0;
  for (auto iter = quant_start; iter != quant.end(); ++iter) {
    uqtype q = *iter;
    size_t pos = (q >> 1) + prev_pos;
    prev_pos = pos;
    if (apply) {
      if (q & 1) buf[pos] += pos_avg;
      else buf[pos] += neg_avg;
    } else {
      if (q & 1) buf[pos] = pos_avg;
      else buf[pos] = neg_avg;
    }
  }
}

void lbann_quantizer::threshold_unquantize(const ThreshQuantized& quant, DistMat& mat,
                                           DataType pos_avg, DataType neg_avg,
                                           bool apply) {
  threshold_unquantize(quant, mat.Matrix(), pos_avg, neg_avg, apply);
}

void lbann_quantizer::adaptive_threshold_quantize(const Mat& mat, ThreshQuantized& q,
                                                  Mat& qerror, int proportion) {
  DataType pos_thresh, neg_thresh, pos_avg, neg_avg;
  std::tie(pos_thresh, neg_thresh, pos_avg, neg_avg) =
    proportion_threshold_average(mat, qerror, proportion);
  // Store the averages for reconstruction.
  uqtype tmp;
  memcpy(&tmp, &pos_avg, sizeof(pos_avg));
  q.push_back(tmp);
  memcpy(&tmp, &neg_avg, sizeof(neg_avg));
  q.push_back(tmp);
  // Do regular thresholded quantization with the computed values.
  threshold_quantize(mat, q, qerror, pos_thresh, neg_thresh, pos_avg, neg_avg);
}

void lbann_quantizer::adaptive_threshold_quantize(const DistMat& mat,
                                                  ThreshQuantized& q,
                                                  Mat& qerror, int proportion) {
  adaptive_threshold_quantize(mat.LockedMatrix(), q, qerror, proportion);
}

void lbann_quantizer::adaptive_threshold_unquantize(
  const ThreshQuantized& q, Mat& mat, bool apply) {
  // Get the averages out.
  DataType pos_avg;
  memcpy(&pos_avg, &(q[0]), sizeof(pos_avg));
  DataType neg_avg;
  memcpy(&neg_avg, &(q[1]), sizeof(neg_avg));
  threshold_unquantize(q, std::next(q.begin(), 2), mat, pos_avg, neg_avg, apply);
}

void lbann_quantizer::adaptive_threshold_unquantize(
  const ThreshQuantized& q, DistMat& mat, bool apply) {
  adaptive_threshold_unquantize(q, mat.Matrix(), apply);
}

void lbann_quantizer::intermodel_sum_threshold_quantized(
  lbann_comm* comm, Mat& mat, Mat& qerror, DataType pos_thresh,
  DataType neg_thresh, Mat& im_qerror, bool compress) {
  ThreshQuantized rs_quant;
  ThreshQuantized rs_recv;
  auto rs_send_trans = 
    [&qerror, &rs_quant, compress, pos_thresh, neg_thresh, this]
    (Mat& mat, IR h, IR w, int& count) {
      auto to_send = mat(h, w);
      auto to_send_qerr = qerror(h, w);
      rs_quant.clear();
      threshold_quantize(to_send, rs_quant, to_send_qerr, pos_thresh,
                         neg_thresh);
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
      rs_recv.reserve(count);
      return rs_recv.data();
    };
  auto rs_recv_trans = 
    [&rs_recv, compress, pos_thresh, neg_thresh, this]
    (uqtype* buf, Mat& accum) {
      if (compress) {
        ThreshQuantized uncomp;
        uncompress_thresholds(rs_recv, uncomp);
        std::swap(rs_recv, uncomp);
      }
      threshold_unquantize(rs_recv, accum, pos_thresh, neg_thresh, true);
    };
  intermodel_ring_reduce_scatter<uqtype>(comm, mat, true, rs_send_trans,
                                         rs_get_recv_buf, rs_recv_trans);
  ThreshQuantized ag_send;
  ThreshQuantized ag_recv;
  auto ag_reduced_trans =
    [&im_qerror, &ag_send, compress, pos_thresh, neg_thresh, this]
    (Mat& reduced) {
      if (im_qerror.Height() == 0) {
        Zeros(im_qerror, reduced.Height(), reduced.Width());
      }
      threshold_quantize(reduced, ag_send, im_qerror, pos_thresh, neg_thresh);
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
      ag_recv.reserve(count);
      return ag_recv.data();
    };
  auto ag_recv_trans = 
    [&ag_recv, compress, pos_thresh, neg_thresh, this]
    (uqtype*, Mat& accum) {
      if (compress) {
        ThreshQuantized uncomp;
        uncompress_thresholds(ag_recv, uncomp);
        threshold_unquantize(uncomp, accum, pos_thresh, neg_thresh);
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
      adaptive_threshold_quantize(to_send, rs_quant, to_send_qerr, proportion);
      if (compress) {
        ThreshQuantized comp;
        compress_adaptive_thresholds(rs_quant, comp);
        std::swap(rs_quant, comp);
      }
      count = rs_quant.size();
      return rs_quant.data();
    };
  auto rs_get_recv_buf = 
    [&rs_recv] (Mat& mat, int& count) {
      rs_recv.reserve(count);
      return rs_recv.data();
    };
  auto rs_recv_trans = 
    [&rs_recv, compress, this]
    (uqtype* buf, Mat& accum) {
      if (compress) {
        ThreshQuantized uncomp;
        uncompress_adaptive_thresholds(rs_recv, uncomp);
        std::swap(rs_recv, uncomp);
      }
      adaptive_threshold_unquantize(rs_recv, accum, true);
    };
  intermodel_ring_reduce_scatter<uqtype>(comm, mat, true, rs_send_trans,
                                         rs_get_recv_buf, rs_recv_trans);
  ThreshQuantized ag_send;
  ThreshQuantized ag_recv;
  auto ag_reduced_trans =
    [&im_qerror, &ag_send, compress, proportion, this]
    (Mat& reduced) {
      if (im_qerror.Height() == 0) {
        Zeros(im_qerror, reduced.Height(), reduced.Width());
      }
      adaptive_threshold_quantize(reduced, ag_send, im_qerror, proportion);
      if (compress) {
        ThreshQuantized comp;
        compress_adaptive_thresholds(ag_send, comp);
        std::swap(comp, ag_send);
      }
    };
  auto ag_get_send_buf = [&ag_send] (int& count) {
      count = ag_send.size();
      return ag_send.data();
    };
  auto ag_get_recv_buf =
    [&ag_recv] (Mat& recv_view, int& count) {
      ag_recv.reserve(count);
      return ag_recv.data();
    };
  auto ag_recv_trans = 
    [&ag_recv, compress, proportion, this]
    (uqtype*, Mat& accum) {
      if (compress) {
        ThreshQuantized uncomp;
        uncompress_adaptive_thresholds(ag_recv, uncomp);
        adaptive_threshold_unquantize(uncomp, accum);
      } else {
        adaptive_threshold_unquantize(ag_recv, accum);
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

void lbann_quantizer::intermodel_sum_adaptive_threshold_quantized(
  lbann_comm* comm, DistMat& mat, Mat& qerror, int proportion, Mat& im_qerror,
  bool compress) {
  intermodel_sum_adaptive_threshold_quantized(comm, mat.Matrix(), qerror,
                                              proportion, im_qerror, compress);
}

void lbann_quantizer::compress_thresholds(const ThreshQuantized& q,
                                          ThreshQuantized& cq) {
  compress_thresholds(q, q.begin(), cq);
}

void lbann_quantizer::compress_thresholds(
  const ThreshQuantized& q, ThreshQuantized::const_iterator qstart,
  ThreshQuantized& cq) {
  // Handle empty input.
  if (std::distance(qstart, q.end()) == 0) {
    cq.push_back(~((uqtype) 0));
    return;
  }
  // Write to cur starting from cur's LSB.
  uqtype cur = 0;
  // The current bit to write to. This is between 0 and NUM_BITS-1.
  // E.g., between 0, ... 31 inclusive, so the bit is 1 << cur_bit.
  // Thus there are NUM_BITS - cur_bit bits left that can be written.
  uqtype cur_bit = 0;
  for (auto iter = qstart; iter != q.end(); ++iter) {
    uqtype ent = *iter;
    uqtype quotient = ent >> GR_K;
    uqtype remainder = ent & (GR_M - 1);
    uqtype bits_left = NUM_BITS - cur_bit;
    // Write quotient 1s.
    if (bits_left >= quotient) {
      // Can fit in the current chunk.
      if (quotient == NUM_BITS) {
        cq.push_back(~((uqtype) 0));
        // Don't need to reset cur, cur_bit: already 0.
      } else {
        cur |= ((1 << quotient) - 1) << cur_bit;
        cur_bit += quotient;
        if (cur_bit == NUM_BITS) {
          cq.push_back(cur);
          cur = 0;
          cur_bit = 0;
        }
      }
    } else {
      // Need to split quotient into multiple chunks.
      // Write the first bits_left 1s to the current chunk.
      if (bits_left == NUM_BITS) {
        cur = ~((uqtype) 0);
      } else {
        cur |= ((1 << bits_left) - 1) << cur_bit;
      }
      cq.push_back(cur);
      quotient -= bits_left;
      // Write chunks of 1s until we have less than NUM_BITS left to write.
      for (uqtype i = 0; i < quotient / NUM_BITS; ++i) {
        cq.push_back(~((uqtype) 0));
      }
      // Lastly write the remaining 1s to a new chunk.
      quotient %= NUM_BITS;
      if (quotient > 0) {
        cur = (1 << quotient) - 1;
        cur_bit = quotient;
      } else {
        cur = 0;
        cur_bit = 0;
      }
    }
    // Write trailing 0.
    // There should always be at least one bit available here.
    cur_bit += 1;
    if (cur_bit == NUM_BITS) {
      cq.push_back(cur);
      cur = 0;
      cur_bit = 0;
    }
    // Write remainder as a GR_K-length binary string.
    // Always fits in at most two chunks, since GR_K <= 31.
    bits_left = NUM_BITS - cur_bit;
    if (bits_left >= GR_K) {
      // Can fit the remainder in the current chunk.
      cur |= remainder << cur_bit;
      cur_bit += GR_K;
      if (cur_bit == NUM_BITS) {
        cq.push_back(cur);
        cur = 0;
        cur_bit = 0;
      }
    } else {
      // Need to split the remainder into two chunks.
      // Write the first bits_left bits to the current chunk.
      cur |= (remainder & ((1 << bits_left) - 1)) << cur_bit;
      cq.push_back(cur);
      // Now write the remaining GR_K - bits_left bits to the new chunk.
      cur = remainder >> bits_left;
      cur_bit = GR_K - bits_left;
    }
  }
  // Pad the end of cur with 1s to terminate it (if needed).
  if (cur_bit > 0) {
    uqtype bits_left = NUM_BITS - cur_bit;
    cur |= ((1 << bits_left) - 1) << cur_bit;
  }
  if (cur) {
    cq.push_back(cur);
  }
}

void lbann_quantizer::compress_adaptive_thresholds(const ThreshQuantized& q,
                                                   ThreshQuantized& cq) {
  cq.push_back(q[0]);
  cq.push_back(q[1]);
  compress_thresholds(q, std::next(q.begin(), 2), cq);
}

void lbann_quantizer::uncompress_thresholds(const ThreshQuantized& cq,
                                            ThreshQuantized& q) {
  uncompress_thresholds(cq, cq.begin(), q);
}

void lbann_quantizer::uncompress_thresholds(
  const ThreshQuantized& cq, ThreshQuantized::const_iterator cqstart,
  ThreshQuantized& q) {
  uqtype quotient = 0;
  uqtype remainder = 0;
  // Like in compress, cur_bit is the current bit being read.
  uqtype cur_bit = 0;
  for (size_t i = std::distance(cq.begin(), cqstart); i < cq.size();) {
    uqtype cur = cq[i];
    // Decode the quotient by continuing until we find a 0.
    // If we hit the end without finding a 0, this was the end of the list.
    while ((cur >> cur_bit) & 0x1) {
      ++quotient;
      ++cur_bit;
      if (cur_bit == NUM_BITS) {
        // Hit end of current chunk.
        ++i;
        if (i == cq.size()) {
          return;  // Nothing left.
        }
        cur = cq[i];
        cur_bit = 0;
      }
    }
    // Skip past the 0.
    ++cur_bit;
    if (cur_bit == NUM_BITS) {
      ++i;
      cur = cq[i];
      cur_bit = 0;
    }
    // Decode the remainder (GR_K bits).
    if (cur_bit + GR_K <= NUM_BITS) {
      // Remainder is entirely in the current chunk.
      remainder = (cur >> cur_bit) & (GR_M - 1);
      cur_bit += GR_K;
      if (cur_bit == NUM_BITS) {
        ++i;
        cur_bit = 0;
      }
    } else {
      // Remainder is split over this and the next chunk.
      uqtype bits_left = NUM_BITS - cur_bit;
      // Start with the top bits_left bits.
      remainder = cur >> cur_bit;
      ++i;
      cur = cq[i];
      // Now get remaining GR_K - bits_left bits from the new cur.
      remainder |= (cur & ((1 << (GR_K - bits_left)) - 1)) << bits_left;
      cur_bit = GR_K - bits_left;
    }
    // Now decode the final value.
    q.push_back(quotient * GR_M + remainder);
    quotient = 0;
    remainder = 0;
  }
}

void lbann_quantizer::uncompress_adaptive_thresholds(const ThreshQuantized& cq,
                                                     ThreshQuantized& q) {
  q.push_back(cq[0]);
  q.push_back(cq[1]);
  uncompress_thresholds(cq, std::next(cq.begin(), 2), q);
}

std::tuple<DataType, DataType, DataType, DataType>
lbann_quantizer::proportion_threshold_average(
  const Mat& mat, const Mat& qerror, int proportion) {
  // It would be nice if there were a better way to do this...
  std::vector<DataType> pos_entries;
  std::vector<DataType> neg_entries;
  for (int row = 0; row < mat.Height(); ++row) {
    for (int col = 0; col < mat.Width(); ++col) {
      DataType val = mat.Get(row, col) + qerror.Get(row, col);
      if (val >= 0.0f) {
        pos_entries.push_back(val);
      } else {
        // Flip negative entries to make selection easier.
        neg_entries.push_back(-1 * val);
      }
    }
  }
  // Determine how many positive/negative entries we need to keep.
  int pos_to_keep = pos_entries.size() / proportion;
  int neg_to_keep = neg_entries.size() / proportion;
  // Determine the threshold value with a selection algorithm to keep only the
  // largest pos/neg_to_keep elements.
  // Set to 0 if there's none.
  // The partitioning also guarantees everything after the i'th element is
  // greater than or equal to it.
  DataType pos_thresh = 0.0f;
  DataType neg_thresh = 0.0f;
  DataType pos_avg = 0.0f;
  DataType neg_avg = 0.0f;
  if (pos_to_keep > 0 && pos_entries.size() > 0) {
    auto i = pos_entries.begin() + (pos_entries.size() - pos_to_keep);
    std::nth_element(pos_entries.begin(), i, pos_entries.end());
    pos_thresh = *i;
    for (; i != pos_entries.end(); ++i) {
      pos_avg += *i;
    }
    pos_avg /= pos_to_keep;
  }
  if (neg_to_keep > 0 && neg_entries.size() > 0) {
    auto i = neg_entries.begin() + (neg_entries.size() - neg_to_keep);
    std::nth_element(neg_entries.begin(), i, neg_entries.end());
    neg_thresh = -1 * (*i);
    for (; i != neg_entries.end(); ++i) {
      neg_avg -= *i;
    }
    neg_avg /= neg_to_keep;
  }
  return std::make_tuple(pos_thresh, neg_thresh, pos_avg, neg_avg);
}

}  // namespace lbann
