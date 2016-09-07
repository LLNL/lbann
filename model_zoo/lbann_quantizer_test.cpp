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
// lbann_quantizer_test.cpp - Tests lbann_quantizer
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include "lbann/lbann_comm.hpp"
#include "lbann/utils/lbann_quantizer.hpp"
#include "lbann_test_utils.hpp"

using namespace lbann;

/** Test quantization and unquantization. */
void test_quantize() {
  Mat mat;
  El::Uniform(mat, 10, 10, 0.0f, 10.0f);
  lbann_quantizer::QuantizedMatrix qmat;
  Mat qerror;
  El::Zeros(qerror, mat.Height(), mat.Width());
  lbann_quantizer quantizer;
  quantizer.quantize(mat, qmat, qerror);
  Mat uqmat(mat.Height(), mat.Width());
  quantizer.unquantize(qmat, uqmat);
  // Ensure there's some quantization error.
  Mat z;
  El::Zeros(z, mat.Height(), mat.Width());
  ASSERT_MAT_NEQ(qerror, z);
  ASSERT_MAT_NEQ(mat, uqmat);
  // With quantization error, we should approximately have the original matrix.
  Mat with_qerror(uqmat);
  with_qerror += qerror;
  ASSERT_MAT_EQ(mat, with_qerror);
}

/**
 * Test quantization/unquantization with one positive and negative value.
 * This should have no error.
 */
void test_2value_quantize() {
  Mat mat;
  El::Rademacher(mat, 10, 10);
  lbann_quantizer::QuantizedMatrix qmat;
  Mat qerror;
  El::Zeros(qerror, mat.Height(), mat.Width());
  lbann_quantizer quantizer;
  quantizer.quantize(mat, qmat, qerror);
  Mat uqmat(mat.Height(), mat.Width());
  quantizer.unquantize(qmat, uqmat);
  // Should have no error.
  Mat z;
  El::Zeros(z, mat.Height(), mat.Width());
  ASSERT_MAT_EQ(qerror, z);
  ASSERT_MAT_EQ(mat, uqmat);
}

/** Test threshold_quantize/unquantize. */
void test_threshold_quantize() {
  Mat mat;
  El::Uniform(mat, 10, 10, 0.0f, 10.0f);
  lbann_quantizer::ThreshQuantized qmat;
  Mat qerror;
  El::Zeros(qerror, mat.Height(), mat.Width());
  lbann_quantizer quantizer;
  quantizer.threshold_quantize(mat, qmat, qerror, 2.0f, -2.0f);
  Mat uqmat;
  El::Zeros(uqmat, mat.Width(), mat.Height());
  quantizer.threshold_unquantize(qmat, uqmat, 2.0f, -2.0f);
  // Ensure there's some quantization error.
  Mat z;
  El::Zeros(z, mat.Height(), mat.Width());
  ASSERT_MAT_NEQ(qerror, z);
  ASSERT_MAT_NEQ(mat, uqmat);
  // Ensure reconstruction is decent.
  Mat with_qerror(uqmat);
  with_qerror += qerror;
  ASSERT_MAT_EQ(mat, with_qerror);
}

/** Test compression with manual inputs. */
void test_compression() {
  lbann_quantizer::ThreshQuantized in = {1000, 0, 1, 2, 1000, 137};
  lbann_quantizer::ThreshQuantized comp;
  lbann_quantizer::ThreshQuantized out;
  lbann_quantizer quantizer;
  quantizer.compress_thresholds(in, comp);
  quantizer.uncompress_thresholds(comp, out);
  ASSERT_VECTOR_EQ(in, out);
}

/** Test threshold compression/uncompression. */
void test_threshold_compression() {
  Mat mat;
  El::Uniform(mat, 10, 10, 0.0f, 10.0f);
  lbann_quantizer::ThreshQuantized qmat;
  Mat qerror;
  El::Zeros(qerror, mat.Height(), mat.Width());
  lbann_quantizer quantizer;
  quantizer.threshold_quantize(mat, qmat, qerror, 2.0f, -2.0f);
  lbann_quantizer::ThreshQuantized compressed_qmat;
  quantizer.compress_thresholds(qmat, compressed_qmat);
  lbann_quantizer::ThreshQuantized uncompressed_qmat;
  quantizer.uncompress_thresholds(compressed_qmat, uncompressed_qmat);
  ASSERT_VECTOR_EQ(qmat, uncompressed_qmat);
  Mat uqmat;
  El::Zeros(uqmat, mat.Width(), mat.Height());
  quantizer.threshold_unquantize(qmat, uqmat, 2.0f, -2.0f);
  // Ensure there's some quantization error.
  Mat z;
  El::Zeros(z, mat.Height(), mat.Width());
  ASSERT_MAT_NEQ(qerror, z);
  ASSERT_MAT_NEQ(mat, uqmat);
  // Ensure reconstruction is decent.
  Mat with_qerror(uqmat);
  with_qerror += qerror;
  ASSERT_MAT_EQ(mat, with_qerror);
}

/** Test adaptive threshold quantization/unquantization. */
void test_adaptive_threshold_quantize() {
  Mat mat;
  El::Uniform(mat, 10, 10, 0.0f, 10.0f);
  lbann_quantizer::ThreshQuantized qmat;
  Mat qerror;
  El::Zeros(qerror, mat.Height(), mat.Width());
  lbann_quantizer quantizer;
  quantizer.adaptive_threshold_quantize(mat, qmat, qerror, 3);
  Mat uqmat;
  El::Zeros(uqmat, mat.Width(), mat.Height());
  quantizer.adaptive_threshold_unquantize(qmat, uqmat);
  // Ensure there's some quantization error.
  Mat z;
  El::Zeros(z, mat.Height(), mat.Width());
  ASSERT_MAT_NEQ(qerror, z);
  ASSERT_MAT_NEQ(mat, uqmat);
  // Ensure reconstruction is decent.
  Mat with_qerror(uqmat);
  with_qerror += qerror;
  ASSERT_MAT_EQ(mat, with_qerror);
}

/** Test adaptive threshold compression/uncompression. */
void test_adaptive_threshold_compression() {
  Mat mat;
  El::Uniform(mat, 10, 10, 0.0f, 10.0f);
  lbann_quantizer::ThreshQuantized qmat, comp_qmat;
  Mat qerror;
  El::Zeros(qerror, mat.Height(), mat.Width());
  lbann_quantizer quantizer;
  quantizer.adaptive_threshold_quantize(mat, comp_qmat, qerror, 3, true);
  Mat uqmat;
  El::Zeros(uqmat, mat.Width(), mat.Height());
  quantizer.adaptive_threshold_unquantize(comp_qmat, uqmat, true);
  // Ensure there's some quantization error.
  Mat z;
  El::Zeros(z, mat.Height(), mat.Width());
  ASSERT_MAT_NEQ(qerror, z);
  ASSERT_MAT_NEQ(mat, uqmat);
  // Ensure reconstruction is decent.
  Mat with_qerror(uqmat);
  with_qerror += qerror;
  ASSERT_MAT_EQ(mat, with_qerror);
}

/** Test the inter-model quantize-and-allreduce (high-bandwidth version). */
void test_quantize_allreduce2() {
  lbann_comm* comm = new lbann_comm(2);
  DistMat mat(comm->get_model_grid());
  if (comm->get_model_rank() == 0) {
    El::Rademacher(mat, 10, 10);
    comm->intermodel_broadcast_matrix(mat, 0);
  } else {
    El::Zeros(mat, 10, 10);
    comm->intermodel_broadcast_matrix(mat, 0);
  }
  if (comm->get_model_rank() % 2 == 1) {
    El::Scale(-1, mat);
  }
  DistMat exact_sum(mat);
  Mat qerror;
  El::Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  Mat im_qerror;
  lbann_quantizer quantizer;
  quantizer.intermodel_sum_quantized2(comm, mat, qerror, im_qerror);
  comm->intermodel_sum_matrix(exact_sum);
  Mat abs_elemerr;
  DataType abs_err = absolute_error(mat.Matrix(), exact_sum.Matrix(),
                                    abs_elemerr);
  // Should have no error.
  Mat z;
  El::Zeros(z, mat.LocalHeight(), mat.LocalWidth());
  ASSERT_MAT_EQ(qerror, z);
  ASSERT_MAT_EQ(mat, exact_sum);
  ASSERT_MAT_EQ(abs_elemerr, z);
}

/** Test the inter-model quantize-and-allreduce. */
void test_quantize_allreduce() {
  lbann_comm* comm = new lbann_comm(2);
  DistMat mat(comm->get_model_grid());
  if (comm->get_model_rank() == 0) {
    El::Rademacher(mat, 10, 10);
    comm->intermodel_broadcast_matrix(mat, 0);
  } else {
    El::Zeros(mat, 10, 10);
    comm->intermodel_broadcast_matrix(mat, 0);
  }
  if (comm->get_model_rank() % 2 == 1) {
    El::Scale(-1, mat);
  }
  DistMat exact_sum(mat);
  Mat qerror;
  El::Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  Mat im_qerror;
  lbann_quantizer quantizer;
  quantizer.intermodel_sum_quantized(comm, mat, qerror, im_qerror);
  comm->intermodel_sum_matrix(exact_sum);  // Compute exact sum.
  Mat abs_elemerr;
  DataType abs_err = absolute_error(mat.Matrix(), exact_sum.Matrix(),
                                    abs_elemerr);
  // Should have no error.
  Mat z;
  El::Zeros(z, mat.LocalHeight(), mat.LocalWidth());
  ASSERT_MAT_EQ(qerror, z);
  ASSERT_MAT_EQ(mat, exact_sum);
  ASSERT_MAT_EQ(abs_elemerr, z);
  delete comm;
}

/** Test the inter-model threshold quantize-and-allreduce. */
void test_threshold_quantize_allreduce() {
  lbann_comm* comm = new lbann_comm(2);
  DistMat mat(comm->get_model_grid());
  if (comm->get_model_rank() == 0) {
    El::Rademacher(mat, 10, 10);
    comm->intermodel_broadcast_matrix(mat, 0);
  } else {
    El::Zeros(mat, 10, 10);
    comm->intermodel_broadcast_matrix(mat, 0);
  }
  if (comm->get_model_rank() % 2 == 1) {
    El::Scale(-1, mat);
  }
  DistMat exact_sum(mat);
  Mat qerror;
  El::Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  Mat im_qerror;
  lbann_quantizer quantizer;
  // Thresholds such that everything is sent.
  quantizer.intermodel_sum_threshold_quantized(comm, mat, qerror, 1.0f, -1.0f,
                                               im_qerror, false);
  comm->intermodel_sum_matrix(exact_sum);
  Mat abs_elemerr;
  DataType abs_err = absolute_error(mat.Matrix(), exact_sum.Matrix(),
                                    abs_elemerr);
  // Since this can't change thresholds, the 0s get requantized as 1s. So we
  // should have no intermediate error, but some final error.
  // Specifically, it should be within 1.
  Mat z;
  El::Zeros(z, mat.LocalHeight(), mat.LocalWidth());
  ASSERT_MAT_EQ(qerror, z);
  ASSERT_MAT_EQ_TOL(mat, exact_sum, 1.0f);
  ASSERT_MAT_EQ_TOL(abs_elemerr, z, 1.0f);
  delete comm;
}

/** Test the inter-model threshold quantize-and-allreduce with compression. */
void test_compressed_threshold_quantize_allreduce() {
    lbann_comm* comm = new lbann_comm(2);
  DistMat mat(comm->get_model_grid());
  if (comm->get_model_rank() == 0) {
    El::Rademacher(mat, 10, 10);
    comm->intermodel_broadcast_matrix(mat, 0);
  } else {
    El::Zeros(mat, 10, 10);
    comm->intermodel_broadcast_matrix(mat, 0);
  }
  if (comm->get_model_rank() % 2 == 1) {
    El::Scale(-1, mat);
  }
  DistMat exact_sum(mat);
  Mat qerror;
  El::Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  Mat im_qerror;
  lbann_quantizer quantizer;
  // Thresholds such that everything is sent.
  quantizer.intermodel_sum_threshold_quantized(comm, mat, qerror, 1.0f, -1.0f,
                                               im_qerror);
  comm->intermodel_sum_matrix(exact_sum);
  Mat abs_elemerr;
  DataType abs_err = absolute_error(mat.Matrix(), exact_sum.Matrix(),
                                    abs_elemerr);
  // Since this can't change thresholds, the 0s get requantized as 1s. So we
  // should have no intermediate error, but some final error.
  // Specifically, it should be within 1.
  Mat z;
  El::Zeros(z, mat.LocalHeight(), mat.LocalWidth());
  ASSERT_MAT_EQ(qerror, z);
  ASSERT_MAT_EQ_TOL(mat, exact_sum, 1.0f);
  ASSERT_MAT_EQ_TOL(abs_elemerr, z, 1.0f);
  delete comm;
}

/** Test the inter-model adaptive threshold quantize-and-allreduce. */
void test_adaptive_threshold_quantize_allreduce() {
  lbann_comm* comm = new lbann_comm(2);
  DistMat mat(comm->get_model_grid());
  if (comm->get_model_rank() == 0) {
    El::Rademacher(mat, 10, 10);
    comm->intermodel_broadcast_matrix(mat, 0);
  } else {
    El::Zeros(mat, 10, 10);
    comm->intermodel_broadcast_matrix(mat, 0);
  }
  if (comm->get_model_rank() % 2 == 1) {
    El::Scale(-1, mat);
  }
  DistMat exact_sum(mat);
  Mat qerror;
  El::Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  Mat im_qerror;
  lbann_quantizer quantizer;
  // Proportion such that everything is sent.
  quantizer.intermodel_sum_adaptive_threshold_quantized(comm, mat, qerror, 1,
                                                        im_qerror, false);
  comm->intermodel_sum_matrix(exact_sum);
  Mat abs_elemerr;
  DataType abs_err = absolute_error(mat.Matrix(), exact_sum.Matrix(),
                                    abs_elemerr);
  Mat z;
  El::Zeros(z, mat.LocalHeight(), mat.LocalWidth());
  ASSERT_MAT_EQ(qerror, z);
  ASSERT_MAT_EQ(mat, exact_sum);
  ASSERT_MAT_EQ(abs_elemerr, z);
  delete comm;
}

/**
 * Test the inter-model adaptive threshold quantize-and-allreduce with
 * compression.
 */
void test_compressed_adaptive_threshold_quantize_allreduce() {
  lbann_comm* comm = new lbann_comm(2);
  DistMat mat(comm->get_model_grid());
  if (comm->get_model_rank() == 0) {
    El::Rademacher(mat, 10, 10);
    comm->intermodel_broadcast_matrix(mat, 0);
  } else {
    El::Zeros(mat, 10, 10);
    comm->intermodel_broadcast_matrix(mat, 0);
  }
  if (comm->get_model_rank() % 2 == 1) {
    El::Scale(-1, mat);
  }
  DistMat exact_sum(mat);
  Mat qerror;
  El::Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  Mat im_qerror;
  lbann_quantizer quantizer;
  // Proportion such that everything is sent.
  quantizer.intermodel_sum_adaptive_threshold_quantized(comm, mat, qerror, 1,
                                                        im_qerror);
  comm->intermodel_sum_matrix(exact_sum);
  Mat abs_elemerr;
  DataType abs_err = absolute_error(mat.Matrix(), exact_sum.Matrix(),
                                    abs_elemerr);
  Mat z;
  El::Zeros(z, mat.LocalHeight(), mat.LocalWidth());
  ASSERT_MAT_EQ(qerror, z);
  ASSERT_MAT_EQ(mat, exact_sum);
  ASSERT_MAT_EQ(abs_elemerr, z);
  delete comm;
}

int main(int argc, char** argv) {
  El::Initialize(argc, argv);
  test_quantize();
  test_2value_quantize();
  test_threshold_quantize();
  test_compression();
  test_threshold_compression();
  test_adaptive_threshold_quantize();
  test_adaptive_threshold_compression();
  test_quantize_allreduce2();
  test_quantize_allreduce();
  test_threshold_quantize_allreduce();
  test_compressed_threshold_quantize_allreduce();
  test_adaptive_threshold_quantize_allreduce();
  test_compressed_adaptive_threshold_quantize_allreduce();
  El::Finalize();
  return 0;
}
