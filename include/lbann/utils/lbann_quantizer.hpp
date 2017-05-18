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

#ifndef LBANN_QUANTIZER_HPP_INCLUDED
#define LBANN_QUANTIZER_HPP_INCLUDED

#include <unordered_map>

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"
#include "lbann/utils/lbann_timer.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include <omp.h>
using namespace El;

#ifndef LBANN_QUANTIZER_TERNARY
#define LBANN_QUANTIZER_TERNARY 0
#endif

namespace lbann
{

/**
 * Support different kinds of quantization.
 * Relevant references:
 * "1-Bit Stochastic Gradient Descent and its Application to Data-Parallel
 * Distributed Training of Speech DNNs" by Frank Seide et al. (MSR)
 * "Scalable Distributed DNN Training Using Commodity GPU Cloud Computing"
 * by Nikko Strom. (Amazon)
 * "Communication Quantization for Data-parallel Training of Deep Neural
 * Networks" by Nikoli Dryden et al. (LLNL/UIUC)
 */
class lbann_quantizer
{
public:
  /** We require that sizeof(DataType) <= sizeof(qtype) == sizeof(uqtype). */
  typedef El::Unsigned uqtype;
  typedef El::Int qtype;
  /**
   * This represents a quantized version of a matrix.
   * Each column is quantized separately. The first two entries are floats
   * representing the positive and negative averages for the column (used in
   * dequantizion). The rest is one-bit quantized entries.
   * Quantization is by column to keep averages nice and because Elemental uses
   * column-major ordering.
   */
  typedef El::Matrix<qtype> QuantizedMatrix;
  typedef std::vector<uqtype> ThreshQuantized;
  typedef std::vector<uint32_t> ThreshQuantized32;
  typedef std::vector<uint64_t> ThreshQuantized64;

  /** Thresholds for use in adaptive quantization. */
  struct adaptive_thresholds {
    /** The positive/upper threshold. */
    DataType pos_thresh;
    /** The negative/lower threshold. */
    DataType neg_thresh;
  };
  /** Reconstruction values for adaptive quantization. */
  struct adaptive_reconstructions {
    /** The positive/upper reconstruction value. */
    DataType pos_recon;
    /** The negative/lower reconstruction value. */
    DataType neg_recon;
#if LBANN_QUANTIZER_TERNARY
    /** The zero/middle reconstruction value. */
    DataType zero_recon;
#endif
  };

  lbann_quantizer();
  ~lbann_quantizer();

  /**
   * Quantize a matrix with onebit quantization.
   * qerror needs to be initialized with:
   * Zeros(qerror, mat.Height(), mat.Width()).
   * @param mat The matrix to quantize.
   * @param qmat The output quantized matrix (will be resized).
   * @param qerror Running quantization error.
   * @param sample Whether to use samples to approximate averages.
   */
  void onebit_quantize(const Mat& mat, QuantizedMatrix& qmat, Mat& qerror,
                bool sample = true);
  void onebit_quantize(const DistMat& mat, QuantizedMatrix& qmat, Mat& qerror,
                bool sample = true);
  /**
   * Unquantize a onebit-quantized matrix..
   * @param qmat The matrix to unquantize.
   * @param mat The output unquantized matrix.
   */
  void onebit_unquantize(const QuantizedMatrix& qmat, Mat& mat);
  void onebit_unquantize(const QuantizedMatrix& qmat, DistMat& mat);
  /**
   * Do a sum reduction of mat over comm's inter-model communicator, with all
   * communication being quantized. im_querror is a separate quantization error
   * matrix that should be passed in each time this is called.
   * This implements the allreduce using a ring-based reduce-scatter followed by
   * a ring-based allgather. Matrices are sent quantized, are unquantized for
   * the reduction, then the reduced matrix is requantized for the allgather.
   * If do_adagrad is true, this scales the inter-mediate unquantized result as
   * in AdaGrad, and uses gradhist to store the gradient history. If used, you
   * should use SGD as the optimizer for those layers to avoid applying AdaGrad
   * twice.
   */
  void intermodel_sum_onebit_quantized(lbann_comm* comm, Mat& mat, Mat& qerror);
  void intermodel_sum_onebit_quantized(lbann_comm* comm, DistMat& mat,
                                       Mat& qerror);

  /**
   * Threshold and quantize a matrix. qerror needs to be initialized with:
   * Zeros(qerror, mat.Height(), mat.Width())).
   * @param mat The matrix to quantize.
   * @param q The output list of quantized entries.
   * @param qerror Running quantization error.
   * @param pos_thresh The positive threshold level.
   * @param neg_thresh The negative threshold level.
   * @param delta Whether to do delta encoding (default false).
   */
  void threshold_quantize(const Mat& mat, ThreshQuantized& q, Mat& qerror,
                          DataType pos_thresh, DataType neg_thresh,
                          bool delta = false);
  void threshold_quantize(const DistMat& mat, ThreshQuantized& q, Mat& qerror,
                          DataType pos_thresh, DataType neg_thresh,
                          bool delta = false);
  /**
   * Unquantize a thresholded-and-quantized matrix.
   * @param q The quantized matrix.
   * @param mat The output unquantized matrix.
   * @param pos_thresh The positive threshold value.
   * @param neg_thresh The negative negative value.
   * @param delta Whether q was quantized with delta encoding (default false).
   */
  void threshold_unquantize(const ThreshQuantized& q, Mat& mat,
                            DataType pos_thresh, DataType neg_thresh,
                            bool delta = false);
  void threshold_unquantize(const ThreshQuantized& q, DistMat& mat,
                            DataType pos_thresh, DataType neg_thresh,
                            bool delta = false);
  /**
   * As with intermodel_sum_onebit_quantized, but use threshold quantization.
   */
  void intermodel_sum_threshold_quantized(lbann_comm* comm, Mat& mat,
                                          Mat& qerror, DataType pos_thresh,
                                          DataType neg_thresh);
  void intermodel_sum_threshold_quantized(lbann_comm* comm, DistMat& mat,
                                          Mat& qerror, DataType pos_thresh,
                                          DataType neg_thresh);

  /**
   * Adaptively quantize a matrix.
   * qerror needs to be initialized with:
   * Zeros(qerror, mat.Height(), mat.Width()).
   * @param mat The matrix to quantize.
   * @param q The output list of quantized entries.
   * @param qerror Running quantization error.
   * @param proportion Quantize one in proportion of the values.
   */
  template <typename colT, typename rowT>
  void adaptive_quantize(const Mat& mat, std::vector<rowT>& q, Mat& qerror,
                                   int proportion);
  template <typename colT, typename rowT>
  void adaptive_quantize(const DistMat& mat, std::vector<rowT>& q,
                                   Mat& qerror, int proportion);
  /**
   * Unquantize an adaptively-quantized matrix.
   * @param q The quantizd matrix.
   * @param mat The output unquantized matrix.
   */
  template <typename colT, typename rowT>
  void adaptive_unquantize(const rowT* q, Mat& mat);
  template <typename colT, typename rowT>
  void adaptive_unquantize(const rowT* q, DistMat& mat);

  /**
   * As with intermodel_sum_onebit_quantized, but use adaptive quantization.
   */
  void intermodel_sum_adaptive_quantized(
    lbann_comm* comm, Mat& mat, Mat& qerror, int proportion);
  void intermodel_sum_adaptive_quantized(
    lbann_comm* comm, DistMat& mat, Mat& qerror, int proportion);

  /**
   * Compute positive and negative thresholds such that only one in proportion
   * of values in mat are >= to the positive threshold or <= to the negative
   * threshold.
   * @param mat The matrix to compute threshold values for.
   * @param qerror The accumulated quantization error in mat.
   * @param proportion Proportion of entries to keep.
   * @param sample Whether to approximate stats by randomly sampling mat.
   * @return The threshold values.
   */
  adaptive_thresholds proportion_threshold(
    const Mat& mat, const Mat& qerror, int proportion, bool sample = true);
  /**
   * Compute reconstruction values for col.
   * @param mat The matrix to compute reconstruction values for.
   * @param qerror The accumulated quantization error in mat.
   * @param col The column to compute reconstruction values for.
   * @param ainfo Adaptive quantization info with thresholds filled in.
   * @param sample Whether to approximate stats by randomly sampling mat.
   * @return Adaptive reconstruction values.
   */
  adaptive_reconstructions col_reconstruction(
    const Mat& mat, const Mat& qerror, Int col,
    const adaptive_thresholds threshes, bool sample = true);

  /** Get the total number of bytes sent during quantization. */
  size_t get_bytes_sent() const { return rs_bytes_sent + ag_bytes_sent; }
  /** Get the total number of bytes sent during the reduce-scatter phase. */
  size_t get_rs_bytes_sent() const { return rs_bytes_sent; }
  /** Get the total number of bytes sent during the all-gather phase. */
  size_t get_ag_bytes_sent() const { return ag_bytes_sent; }
  /** Get the total number of bytes received during quantization. */
  size_t get_bytes_received() const {
    return rs_bytes_received + ag_bytes_received;
  }
  /** Get the total number of bytes received during the reduce-scatter phase. */
  size_t get_rs_bytes_received() const { return rs_bytes_received; }
  /** Get the total number of bytes received during the all-gather phase. */
  size_t get_ag_bytes_received() const { return ag_bytes_received; }
  /** Reset recorded bytes counters. */
  void reset_bytes_counters() {
    rs_bytes_sent = 0;
    ag_bytes_sent = 0;
    rs_bytes_received = 0;
    ag_bytes_received = 0;
  }
  /** Get the time spent in the reduce-scatter. */
  double get_rs_time() const { return rs_time; }
  /** Get the time spent in the allgather. */
  double get_ag_time() const { return ag_time; }
  /** Get the time spent in the reduce-scatter send_trans. */
  double get_rs_send_trans_time() const { return rs_send_trans_time; }
  /** Get the time spent in the reduce-scatter get_recv_buf. */
  double get_rs_recv_buf_time() const { return rs_recv_buf_time; }
  /** Get the time spent in the reduce-scatter recv_trans. */
  double get_rs_recv_trans_time() const { return rs_recv_trans_time; }
  /** Get the time spent in sends in the reduce-scatter. */
  double get_rs_send_time() const { return rs_send_time; }
  /** Get the time spent in recvs/waiting for recvs in the reduce-scatter. */
  double get_rs_recv_wait_time() const { return rs_recv_wait_time; }
  /** Get the time spent in the allgather reduce_trans. */
  double get_ag_reduced_trans_time() const { return ag_reduced_trans_time; }
  /** Get the time spent in the allgather get_recv_buf. */
  double get_ag_recv_buf_time() const { return ag_recv_buf_time; }
  /** Get the time spent in the all-gather send_trans. */
  double get_ag_recv_trans_time() const { return ag_recv_trans_time; }
  /** Get the time spent in sends in the allgather. */
  double get_ag_send_time() const { return ag_send_time; }
  /** Get the time spent in recvs/waiting for recvs in the allgather. */
  double get_ag_recv_wait_time() const { return ag_recv_wait_time; }
  /** Get the time spent in proportion_threshold. */
  double get_proportion_time() const { return proportion_time; }
  /** Reset recorded time counters. */
  void reset_time_counters() {
    rs_time = 0.0;
    ag_time = 0.0;
    rs_send_trans_time = 0.0;
    rs_recv_buf_time = 0.0;
    rs_recv_trans_time = 0.0;
    rs_send_time = 0.0;
    rs_recv_wait_time = 0.0;
    ag_reduced_trans_time = 0.0;
    ag_recv_buf_time = 0.0;
    ag_recv_trans_time = 0.0;
    ag_send_time = 0.0;
    ag_recv_wait_time = 0.0;
    proportion_time = 0.0;
  }
  /** Return the most recent number of quantized entries. */
  size_t get_quantized_count() const { return quantized_count; }

private:
  /** Number of bits per quantized word. */
  static const size_t NUM_BITS = sizeof(qtype) * 8;
  /** Number of samples to use in proportion_threshold. */
  static const Int NUM_THRESHOLD_SAMPLES = 1024;
  /** Number of samples to use in col_reconstruction. */
  static const Int NUM_RECON_SAMPLES = 128;
  /** Samples to use to approximate column averages in onebit quantization. */
  static const Int NUM_ONEBIT_SAMPLES = 128;
  /** Factor used when computing header lengths in adaptive quantization. */
#if LBANN_QUANTIZER_TERNARY
  static const int HEADER_FACTOR = 4;
#else
  static const int HEADER_FACTOR = 3;
#endif
  /** Max factor by which adaptive quantization can exceed optimal amount. */
  static const Int MAX_QUANTIZED_EXCESS = 4;

  /** Bytes sent in doing the reduce-scatter. */
  size_t rs_bytes_sent;
  /** Bytes sent in doing the all-gather. */
  size_t ag_bytes_sent;
  /** Bytes received in doing the reduce-scatter. */
  size_t rs_bytes_received;
  /** Bytes received in doing the all-gather. */
  size_t ag_bytes_received;
  /** Time spent in the reduce-scatter. */
  double rs_time;
  /** Time spent in the all-gather. */
  double ag_time;
  /** Time spent in the reduce-scatter send_trans. */
  double rs_send_trans_time;
  /** Time spent in the reduce-scatter get_recv_buf. */
  double rs_recv_buf_time;
  /** Time spent in the reduce-scatter recv_trans. */
  double rs_recv_trans_time;
  /** Time spent in the reduce-scatter sending. */
  double rs_send_time;
  /** Time spent in the reduce-scatter receiving/waiting for the receive. */
  double rs_recv_wait_time;
  /** Time spent in the allgather reduced_trans. */
  double ag_reduced_trans_time;
  /** Time spent in the allgather get_recv_buf. */
  double ag_recv_buf_time;
  /** Time spent in the all-gather recv_trans. */
  double ag_recv_trans_time;
  /** Time spent in the allgather sending. */
  double ag_send_time;
  /** Time spent in the allgather receiving/waiting for the receive. */
  double ag_recv_wait_time;
  /** Time spent in proportion_threshold. */
  double proportion_time;
  /** Most recent number of quantized entries. */
  size_t quantized_count;

  /** Return the height of mat after quantization with onebit_quantize(). */
  inline Int get_onebit_quantized_matrix_height(const Mat& mat) const {
    return (mat.Height() + (NUM_BITS-1)) / NUM_BITS + 2;
  }

  /** Variant of unquantize that adds its entries. */
  void onebit_unquantize_add(const QuantizedMatrix& qmat, Mat& mat);

  /**
   * Do threshold unquantization from arbitrary locations, adding the
   * unquantized values to existing ones instead of replacing them, and storing
   * the locations applied.
   */
  void threshold_unquantize_apply(const ThreshQuantized& q, Mat& mat,
                                  DataType pos_thresh, DataType neg_thresh,
                                  std::vector<Unsigned>& positions,
                                  bool delta = false);
  /**
   * Quantize only the locations in mat in positions; the companion of
   * threshold_unquantize_apply.
   */
  void threshold_quantize_apply(const Mat& mat, ThreshQuantized& q, Mat& qerror,
                                DataType pos_thresh, DataType neg_thresh,
                                std::vector<Unsigned>& positions,
                                bool delta = false);

  /**
   * Variant of adaptive_unquantize that adds its entries.
   */
  template <typename colT, typename rowT>
  void adaptive_unquantize_add(const rowT* q, Mat& mat);
  /**
   * Variant of adaptive_quantize that also replaces entries in mat
   * with their quantized version. This is equivalent to:
   * adaptive_quantize(mat, q, qerror, proportion);
   * adaptive_unquantize(q, mat);
   */
  template <typename colT, typename rowT>
  void adaptive_quantize_replace(Mat& mat, std::vector<rowT>& q,
                                 Mat& qerror, int proportion);
  /**
   * Ensure that q is no more than a factor of MAX_QUANTIZED_EXCESS larger
   * than optimal.
   */
  template <typename colT, typename rowT>
  void adaptive_bound(const Mat& mat, Mat& qerror, std::vector<rowT>& q,
                      int proportion);
  template <typename colT, typename rowT>
  void adaptive_quantize_slice(const std::vector<rowT>& q,
                               const Mat& mat, Mat& qerror,
                               std::vector<rowT>& slice, colT start,
                               colT end, int proportion);
  template <typename colT, typename rowT>
  void intermodel_sum_adaptive_quantized_impl(
    lbann_comm* comm, Mat& mat, Mat& qerror, int proportion);

  /**
   * Return the number of threads adaptive quantization should use for a matrix
   * with the given width.
   * This number of threads is empirically determined.
   * @todo Make this configurable at compile time.
   */
  inline int get_adaptive_quantization_threads(El::Int width) {
    int num_threads = omp_get_max_threads();
    if (width <= 64) {
      num_threads = 2;
    } else if (width <= 128) {
      num_threads = 8;
    } else if (width <= 256) {
      num_threads = 12;
    } else if (width <= 1024) {
      num_threads = 24;
    }
    return std::min(omp_get_max_threads(), num_threads);
  }

  /**
   * Return the number of threads adaptive quantization uses in its copy loop.
   * This is empirically determined.
   * @param width The width of the matrix being quantized.
   * @todo Make this configurable at compile time.
   * @note If this and get_adaptive_quantization_threads return different values
   * for the same width, OpenMP may reap its threads and add additional overhead
   * when invoking a parallel region with more threads.
   */
  inline int get_adaptive_quantization_copy_threads(El::Int width) {
    int num_threads = get_adaptive_quantization_threads(width);
    if (width >= 16384) {
      num_threads /= 2;
    }
    return num_threads;
  }
};

}  // namespace lbann

#include "lbann_quantizer_impl.hpp"

#endif  // LBANN_QUANTIZER_HPP_INCLUDED
