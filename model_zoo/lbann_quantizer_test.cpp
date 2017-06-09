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

// Test local quantization/unquantization.

/** Do some checks on the unquantized matrix. */
void check_quantized_mat(const Mat& orig, const Mat& qerror, const Mat& uqmat,
                         bool exact) {
  if (!exact) {
    // Ensure there is some quantization error.
    Mat z;
    El::Zeros(z, orig.Height(), orig.Width());
    ASSERT_MAT_NEQ(qerror, z);
    ASSERT_MAT_NEQ(orig, uqmat);
  }
  // Ensure we can use qerror to recover the original matrix.
  Mat with_qerror(uqmat);
  with_qerror += qerror;
  Mat err(with_qerror);
  err -= orig;
  ASSERT_MAT_EQ(orig, with_qerror);
}

/** Test onebit quantization. */
void test_onebit_quantization(const Mat& mat, bool exact) {
  std::cout << "Testing onebit" << std::endl;
  Mat qerror, uqmat;
  El::Zeros(qerror, mat.Height(), mat.Width());
  El::Zeros(uqmat, mat.Height(), mat.Width());
  lbann_quantizer quantizer;
  lbann_quantizer::QuantizedMatrix qmat;
  quantizer.onebit_quantize(mat, qmat, qerror);
  quantizer.onebit_unquantize(qmat, uqmat);
  check_quantized_mat(mat, qerror, uqmat, exact);
}

/** Test threshold quantization. */
void test_threshold_quantization(const Mat& mat, bool exact) {
  std::cout << "Testing threshold" << std::endl;
  Mat qerror, uqmat;
  El::Zeros(qerror, mat.Height(), mat.Width());
  El::Zeros(uqmat, mat.Height(), mat.Width());
  lbann_quantizer quantizer;
  lbann_quantizer::ThreshQuantized qmat;
  quantizer.threshold_quantize(mat, qmat, qerror, DataType(2.0), DataType(-2.0));
  quantizer.threshold_unquantize(qmat, uqmat, DataType(2.0), DataType(-2.0));
  check_quantized_mat(mat, qerror, uqmat, exact);
}

/** Test adaptive quantization. */
void test_adaptive_quantization(const Mat& mat, bool exact) {
  std::cout << "Testing adaptive" << std::endl;
  Mat qerror, uqmat;
  El::Zeros(qerror, mat.Height(), mat.Width());
  El::Zeros(uqmat, mat.Height(), mat.Width());
  lbann_quantizer quantizer;
  std::vector<uint16_t> qmat;
  // Handle different datatype sizes.
  typedef std::conditional<sizeof(DataType) <= sizeof(uint32_t),
          uint32_t, uint64_t>::type colT;
  quantizer.adaptive_quantize<colT, uint16_t>(mat, qmat, qerror, 3);
  quantizer.adaptive_unquantize<colT, uint16_t>(qmat.data(), uqmat);
  check_quantized_mat(mat, qerror, uqmat, exact);
}

// Test quantized allreduces.

void check_allreduced_mat(lbann_comm *comm, const DistMat& mat,
                          const DistMat& exact_sum, const Mat& qerror,
                          bool exact) {
  if (exact) {
    ASSERT_MAT_EQ(mat.LockedMatrix(), exact_sum.LockedMatrix());
  }
  // Compute the global error and compare.
  Mat global_qerror(qerror);
  comm->intermodel_sum_matrix(global_qerror);
  Mat with_qerror(mat.LockedMatrix());
  with_qerror += global_qerror;
  ASSERT_MAT_EQ(exact_sum.LockedMatrix(), with_qerror);
}

/** Test onebit quantized allreduce. */
void test_onebit_quantize_allreduce(lbann_comm *comm, DistMat& mat,
                                    bool exact) {
  if (comm->am_world_master()) {
    std::cout << "Testing onebit" << std::endl;
  }
  DistMat exact_sum(mat);
  Mat qerror;
  El::Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  lbann_quantizer quantizer;
  quantizer.intermodel_sum_onebit_quantized(comm, mat, qerror);
  comm->intermodel_sum_matrix(exact_sum);
  check_allreduced_mat(comm, mat, exact_sum, qerror, exact);
}

/** Test threshold quantized allreduce. */
void test_threshold_quantize_allreduce(lbann_comm *comm, DistMat& mat,
                                       bool exact) {
  if (comm->am_world_master()) {
    std::cout << "Testing threshold" << std::endl;
  }
  DistMat exact_sum(mat);
  Mat qerror;
  El::Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  lbann_quantizer quantizer;
  quantizer.intermodel_sum_threshold_quantized(comm, mat, qerror,
      DataType(2.0), DataType(-2.0));
  comm->intermodel_sum_matrix(exact_sum);
  check_allreduced_mat(comm, mat, exact_sum, qerror, exact);
}

/** Test adaptively quantized allreduce. */
void test_adaptive_quantize_allreduce(lbann_comm *comm, DistMat& mat,
                                      bool exact) {
  if (comm->am_world_master()) {
    std::cout << "Testing adaptive" << std::endl;
  }
  DistMat exact_sum(mat);
  Mat qerror;
  El::Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  lbann_quantizer quantizer;
  quantizer.intermodel_sum_adaptive_quantized(comm, mat, qerror, 1);
  comm->intermodel_sum_matrix(exact_sum);
  check_allreduced_mat(comm, mat, exact_sum, qerror, exact);
}

/** Test local operations. */
void test_local() {
  std::cout << "Testing local quantization" << std::endl;
  for (Int mat_size = 1; mat_size <= 4096; mat_size *= 2) {
    // Test uniform matrix.
    Mat uniform_mat;
    El::Uniform(uniform_mat, mat_size, mat_size, DataType(0), DataType(4));
    std::cout << "Uniform " << mat_size << "x" << mat_size << std::endl;
    test_onebit_quantization(uniform_mat, false || mat_size <= 2);
    test_threshold_quantization(uniform_mat, false);
    test_adaptive_quantization(uniform_mat, false || mat_size <= 2);
    // Test Gaussian matrix.
    Mat gaussian_mat;
    El::Gaussian(gaussian_mat, mat_size, mat_size, DataType(0), DataType(2));
    std::cout << "Gaussian " << mat_size << "x" << mat_size << std::endl;
    test_onebit_quantization(gaussian_mat, false || mat_size <= 2);
    test_threshold_quantization(gaussian_mat, false);
    test_adaptive_quantization(gaussian_mat, false || mat_size <= 2);
    // Test Rademacher matrix (should be exact).
    Mat rademacher_mat;
    El::Rademacher(rademacher_mat, mat_size, mat_size);
    std::cout << "Rademacher " << mat_size << "x" << mat_size << std::endl;
    test_onebit_quantization(rademacher_mat, true);
    // Threshold can't guarantee exact reconstruction.
    test_threshold_quantization(rademacher_mat, false);
    test_adaptive_quantization(rademacher_mat, true);
  }
}

/** Test global allreduce operations. */
void test_allreduces() {
  lbann_comm *comm = new lbann_comm(1);
  if (comm->am_world_master()) {
    std::cout << "Testing quantized allreduces" << std::endl;
  }
  // Note: Threshold quantized allreduce not currently supported.
  for (Int mat_size = 1; mat_size <= 4096; mat_size *= 2) {
    // Test Rademacher matrix (should be exact);
    DistMat rademacher_mat(comm->get_model_grid());
    if (comm->get_model_rank() == 0) {
      El::Rademacher(rademacher_mat, mat_size, mat_size);
      comm->intermodel_broadcast_matrix(rademacher_mat, 0);
    } else {
      rademacher_mat.Resize(mat_size, mat_size);
      comm->intermodel_broadcast_matrix(rademacher_mat, 0);
    }
    if (comm->get_model_rank() % 2 == 1) {
      El::Scale(-1, rademacher_mat);
    }
    DistMat onebit_rademacher(rademacher_mat),
            threshold_rademacher(rademacher_mat),
            adaptive_rademacher(rademacher_mat);
    if (comm->get_model_rank() % 2 == 1) {
      // Adaptive quantization disregards 0s, so we need this to sum to a
      // different value instead.
      // In the case of 3 models, don't scale by 2 or else elements sum to 0.
      if (comm->get_num_models() != 3) {
        El::Scale(2, adaptive_rademacher);
      }
    }
    if (comm->am_world_master()) {
      std::cout << "Rademacher " << mat_size << "x" << mat_size << std::endl;
    }
    test_onebit_quantize_allreduce(comm, onebit_rademacher, true);
    //test_threshold_quantize_allreduce(comm, threshold_rademacher, false);
    test_adaptive_quantize_allreduce(comm, adaptive_rademacher, true);
  }
}

int main(int argc, char **argv) {
  El::Initialize(argc, argv);
  if (El::mpi::Rank() == 0) {
    test_local();
  }
  test_allreduces();
  El::Finalize();
  return 0;
}
