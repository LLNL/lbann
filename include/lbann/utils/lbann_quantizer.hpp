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

#ifndef LBANN_QUANTIZER_HPP_INCLUDED
#define LBANN_QUANTIZER_HPP_INCLUDED

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"
#ifdef __LIB_ELEMENTAL
using namespace El;
#endif

namespace lbann
{

/**
 * Perform one-bit quantization of matrices (by column) with the quantization
 * threshold at 0. Optionally do thresholding and entropy coding.
 * Relevant references:
 * "1-Bit Stochastic Gradient Descent and its Application to Data-Parallel
 * Distributed Training of Speech DNNs" by Frank Seide et al. (MSR)
 * "Scalable Distributed DNN Training Using Commodity GPU Cloud Computing"
 * by Nikko Strom. (Amazon)
 */
class lbann_quantizer
{
public:
  /** We require that sizeof(DataType) == sizeof(qtype) == sizeof(uqtype). */
  typedef uint32_t uqtype;
  typedef int32_t qtype;
  /**
   * This represents a quantized version of a matrix.
   * Each column is quantized separately. The first two entries are floats
   * representing the positive and negative averages for the column (used in
   * dequantizion). The rest is one-bit quantized entries.
   * Quantization is by column to keep averages nice and because Elemental uses
   * column-major ordering.
   * This is int32_t because Elemental matrices don't support unsigned or
   * >32-bit types. It forces some type-casting annoyances.
   */
  typedef El::Matrix<qtype> QuantizedMatrix;
  typedef std::vector<uqtype> ThreshQuantized;

  lbann_quantizer();
  ~lbann_quantizer();

  /**
   * Quantize a matrix.
   * @param mat The matrix to quantize.
   * @param qmat The output quantized matrix (will be resized).
   * @param qerror Running quantization error.
   */
  void quantize(const Mat& mat, QuantizedMatrix& qmat, Mat& qerror);
  void quantize(const DistMat& mat, QuantizedMatrix& qmat, Mat& qerror);
  /**
   * Unquantize a matrix.
   * @param qmat The matrix to unquantize.
   * @param mat The output unquantized matrix.
   * @param apply Whether to add or replace existing entries.
   */
  void unquantize(const QuantizedMatrix& qmat, Mat& mat, bool apply = false);
  void unquantize(const QuantizedMatrix& qmat, DistMat& mat, bool apply = false);
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
  void intermodel_sum_quantized(lbann_comm* comm, Mat& mat, Mat& qerror,
                                Mat& im_qerror, bool do_adagrad = false,
                                Mat* gradhist = nullptr);
  void intermodel_sum_quantized(lbann_comm* comm, DistMat& mat, Mat& qerror,
                                Mat& im_qerror, bool do_adagrad = false,
                                Mat* gradhist = nullptr);
  void intermodel_sum_quantized2(lbann_comm* comm, Mat& mat, Mat& qerror,
                                 Mat& im_qerror);
  void intermodel_sum_quantized2(lbann_comm* comm, DistMat& mat, Mat& qerror,
                                 Mat& im_qerror);

  /**
   * Threshold and quantize a matrix.
   * @param mat The matrix to quantize.
   * @param q The output list of quantized entries.
   * @param qerror Running quantization error.
   * @param pos_thresh The positive threshold level.
   * @param neg_thresh The negative threshold level.
   * @param pos_avg The positive quantization value (0 for default).
   * @param neg_avg The negative quantization value (0 for default).
   */
  void threshold_quantize(const Mat& mat, ThreshQuantized& q, Mat& qerror,
                          DataType pos_thresh, DataType neg_thresh,
                          DataType pos_avg = 0.0f, DataType neg_avg = 0.0f);
  void threshold_quantize(const DistMat& mat, ThreshQuantized& q, Mat& qerror,
                          DataType pos_thresh, DataType neg_thresh,
                          DataType pos_avg = 0.0f, DataType neg_avg = 0.0f);
  /**
   * Unquantize a thresholded-and-quantized matrix.
   * @param q The quantized matrix.
   * @param mat The output unquantized matrix.
   * @param pos_avg The positive quantization value.
   * @param neg_avg The negative quantization value.
   * @param apply Whether to add unquantized data to existing entries or replace
   * existing entries.
   */
  void threshold_unquantize(const ThreshQuantized& q, Mat& mat,
                            DataType pos_avg, DataType neg_avg,
                            bool apply = false);
  void threshold_unquantize(const ThreshQuantized& q, DistMat& mat,
                            DataType pos_avg, DataType neg_avg,
                            bool apply = false);
  /**
   * Threshold and quantize a matrix, dynamically choosing the threshold and
   * quantization values.
   * @param mat The matrix to quantize.
   * @param q The output list of quantized entries.
   * @param qerror Running quantization error.
   * @param proportion Quantize one in proportion of the values.
   */
  void adaptive_threshold_quantize(const Mat& mat, ThreshQuantized& q, Mat& qerror,
                                   int proportion);
  void adaptive_threshold_quantize(const DistMat& mat, ThreshQuantized& q,
                                   Mat& qerror, int proportion);
  /**
   * Unquantize an adaptively-thresholded-and-quantized matrix.
   * @param q The quantizd matrix.
   * @param mat The output unquantized matrix.
   * @param apply Whether to add unquantized data to existing entries or replace
   * existing entries.
   */
  void adaptive_threshold_unquantize(const ThreshQuantized& q, Mat& mat,
                                     bool apply = false);
  void adaptive_threshold_unquantize(const ThreshQuantized& q, DistMat& mat,
                                     bool apply = false);
  /**
   * As with intermodel_sum_quantized, but use threshold quantization.
   */
  void intermodel_sum_threshold_quantized(lbann_comm* comm, Mat& mat,
                                          Mat& qerror, DataType pos_thresh,
                                          DataType neg_thresh, Mat& im_qerror,
                                          bool compress=true);
  void intermodel_sum_threshold_quantized(lbann_comm* comm, DistMat& mat,
                                          Mat& qerror, DataType pos_thresh,
                                          DataType neg_thresh, Mat& im_qerror,
                                          bool compress=true);

  /**
   * As with intermodel_sum_quantized, but use adaptive threshold quantization.
   */
  void intermodel_sum_adaptive_threshold_quantized(
    lbann_comm* comm, Mat& mat, Mat& qerror, int proportion, Mat& im_qerror,
    bool compress=true);
  void intermodel_sum_adaptive_threshold_quantized(
    lbann_comm* comm, DistMat& mat, Mat& qerror, int proportion, Mat& im_qerror,
    bool compress=true);

  /**
   * Compress the output of threshold_quantize.
   * This uses Golumb-Rice coding, with the quotient stored first, followed by
   * the remainder.
   */
  void compress_thresholds(const ThreshQuantized& q,
                           ThreshQuantized& cq);
  void compress_adaptive_thresholds(const ThreshQuantized& q,
                                    ThreshQuantized& cq);
  /** Corresponding uncompress. */
  void uncompress_thresholds(const ThreshQuantized& cq,
                             ThreshQuantized& q);
  void uncompress_adaptive_thresholds(const ThreshQuantized& cq,
                                      ThreshQuantized& q);

  /**
   * Compute positive and negative threshold values such that only one in
   * proportion of positive and negative entries are greater than or equal to
   * the threshold. Additionally, compute the average value of the positive and
   * negative values greater than this threshold.
   * @param mat The matrix to compute threshold values for.
   * @param proportion Proportion of entries to keep.
   * @return In this order: The positive and negative threshold values, then the
   * positive and negative averages.
   */
  std::tuple<DataType, DataType, DataType, DataType> proportion_threshold_average(
    const Mat& mat, int proportion);

  size_t get_bytes_sent() const { return bytes_sent; }
  size_t get_bytes_received() const { return bytes_received; }
  void reset_bytes_counters() {
    bytes_sent = 0;
    bytes_received = 0;
  }

private:
  /** Number of bits per quantized word. */
  static const size_t NUM_BITS = sizeof(qtype) * 8;
  /**
   * Golumb-Rice M parameter, a power of 2. Should be large-ish relative to the
   * data being encoded, but log_2(GR_M) should be <= 31.
   */
  static const uqtype GR_M = 16;
  /** log_2(GR_M). */
  static const uqtype GR_K = 4;

  size_t bytes_sent;
  size_t bytes_received;

  /** Return the height of mat after quantization with quantize(). */
  inline int get_quantized_matrix_height(const Mat& mat) const {
    return (mat.Height() + (NUM_BITS-1)) / NUM_BITS + 2;
  }

  /** Handle threshold unquantization from arbitrary locations. */
  void threshold_unquantize(const ThreshQuantized& q,
                            ThreshQuantized::const_iterator qstart, Mat& mat,
                            DataType pos_avg, DataType neg_avg,
                            bool apply = false);
  /** Handle compression starting from arbitrary locations. */
  void compress_thresholds(const ThreshQuantized& q,
                           ThreshQuantized::const_iterator qstart,
                           ThreshQuantized& cq);
  /** Handle uncompression starting from arbitrary locations. */
  void uncompress_thresholds(const ThreshQuantized& cq,
                             ThreshQuantized::const_iterator cqstart,
                             ThreshQuantized& q);

  template <typename T>
  void intermodel_ring_reduce_scatter(
    lbann_comm* comm, Mat& mat, bool var_recv,
    std::function<T*(Mat&, IR, IR, int&)> send_trans,
    std::function<T*(Mat&, int&)> get_recv_buf,
    std::function<void(T*, Mat&)> recv_trans);

  template <typename T>
  void intermodel_ring_allgather(
    lbann_comm* comm, Mat& mat, bool var_recv,
    std::function<void(Mat&)> reduced_trans,
    std::function<T*(int&)> get_send_buf,
    std::function<T*(Mat&, int&)> get_recv_buf,
    std::function<void(T*, Mat&)> recv_trans,
    std::function<void(T*, T*)> swap_bufs);
};

template <typename T>
void lbann_quantizer::intermodel_ring_reduce_scatter(
  lbann_comm* comm, Mat& mat, bool var_recv,
  std::function<T*(Mat&, IR, IR, int&)> send_trans,
  std::function<T*(Mat&, int&)> get_recv_buf,
  std::function<void(T*, Mat&)> recv_trans) {
  int rank = comm->get_model_rank();
  int nprocs = comm->get_num_models();
  // Compute the number of columns each processor sends.
  // The last processor handles the excess.
  int cols_per_proc = mat.Width() / nprocs;
  int cols_remainder = mat.Width() % nprocs;
  int local_col_width = cols_per_proc;
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
    T* send_buf = send_trans(
      mat, IR(0, mat.Height()),
      IR(dst * cols_per_proc, dst * cols_per_proc + send_col_width), send_size);
    // Send.
    lbann_mpi_req<T> req;
    comm->nb_send(send_buf, send_size, dst, req);
    bytes_sent += send_size * sizeof(T);
    // Get receive buffer.
    int recv_size = 0;
    if (var_recv) {
      recv_size = comm->get_count<T>(src);
    }
    T* recv_buf = get_recv_buf(accum_view, recv_size);
    // Receive.
    comm->recv(recv_buf, recv_size, src);
    bytes_received += recv_size * sizeof(T);
    // Transform the received portion.
    recv_trans(recv_buf, accum_view);
    comm->wait<T>(req);
  }
}

template <typename T>
void lbann_quantizer::intermodel_ring_allgather(
    lbann_comm* comm, Mat& mat, bool var_recv,
    std::function<void(Mat&)> reduced_trans,
    std::function<T*(int&)> get_send_buf,
    std::function<T*(Mat&, int&)> get_recv_buf,
    std::function<void(T*, Mat&)> recv_trans,
    std::function<void(T*, T*)> swap_bufs) {
  int rank = comm->get_model_rank();
  int nprocs = comm->get_num_models();
  // Compute the number of columns each processor sends.
  // The last processor handles the excess.
  int cols_per_proc = mat.Width() / nprocs;
  int cols_remainder = mat.Width() % nprocs;
  int local_col_width = cols_per_proc;
  if (rank == nprocs - 1) local_col_width += cols_remainder;
  // Get the portion of mat that was reduced.
  auto reduced = mat(IR(0, mat.Height()),
                     IR(rank * cols_per_proc,
                        rank * cols_per_proc + local_col_width));
  // Transform the reduced data.
  reduced_trans(reduced);
  // Compute the previous/next ranks in the ring.
  int src = rank - 1;
  if (src < 0) src = nprocs - 1;
  int dst = (rank + 1) % nprocs;
  // Do the allgather.
  for (int step = 0; step < nprocs - 1; ++step) {
    // Send our data or forward received data.
    lbann_mpi_req<T> req;
    int send_size;
    T* send_buf = get_send_buf(send_size);
    comm->nb_send(send_buf, send_size, dst, req);
    bytes_sent += send_size * sizeof(T);
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
    int recv_size = 0;
    if (var_recv) {
      recv_size = comm->get_count<T>(src);
    }
    T* recv_buf = get_recv_buf(recv_view, recv_size);
    // Receive data.
    comm->recv(recv_buf, recv_size, src);
    bytes_received += recv_size * sizeof(T);
    // Transform the received portion.
    recv_trans(recv_buf, recv_view);
    comm->wait<T>(req);
    // Swap so we forward the data we just received.
    swap_bufs(send_buf, recv_buf);
    send_size = recv_size;
  }
}

}  // namespace lbann

#endif  // LBANN_QUANTIZER_HPP_INCLUDED
