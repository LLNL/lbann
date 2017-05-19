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
// lbann_collective_test.cpp - Tests custom LBANN collective implementations
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include "lbann/lbann_comm.hpp"
#include "lbann/utils/lbann_timer.hpp"
#include "lbann_test_utils.hpp"

using namespace lbann;

const int num_trials = 20;

void test_rd_allreduce(lbann_comm* comm, DistMat& dmat) {
  auto send_transform =
    [] (Mat& mat, IR h, IR w, int& send_size, bool const_data) {
    auto to_send = mat(h, w);
    send_size = sizeof(DataType) * to_send.Height() * to_send.Width();
    return (uint8_t*) to_send.Buffer();
  };
  auto recv_apply_transform =
    [] (uint8_t* recv_buf, Mat& accum) {
    Mat recv_mat;
    recv_mat.LockedAttach(accum.Height(), accum.Width(), (DataType*) recv_buf,
                          accum.LDim());
    accum += recv_mat;
    return sizeof(DataType) * recv_mat.Height() * recv_mat.Width();
  };
  Mat& mat = dmat.Matrix();
  int max_recv_count = sizeof(DataType) * mat.Height() * mat.Width();
  comm->recursive_doubling_allreduce_pow2(
    comm->get_intermodel_comm(), mat, max_recv_count,
    std::function<uint8_t*(Mat&, IR, IR, int&, bool)>(send_transform),
    std::function<int(uint8_t*, Mat&)>(recv_apply_transform));
}

void test_pe_ring_allreduce(lbann_comm* comm, DistMat& dmat) {
  auto send_transform =
    [] (Mat& mat, IR h, IR w, int& send_size, bool const_data) {
    auto to_send = mat(h, w);
    send_size = sizeof(DataType) * to_send.Height() * to_send.Width();
    return (uint8_t*) to_send.Buffer();
  };
  auto recv_transform =
    [] (uint8_t* recv_buf, Mat& accum) {
    Mat recv_mat;
    recv_mat.LockedAttach(accum.Height(), accum.Width(), (DataType*) recv_buf,
                          accum.LDim());
    accum = recv_mat;
    return sizeof(DataType) * recv_mat.Height() * recv_mat.Width();
  };
  auto recv_apply_transform =
    [] (uint8_t* recv_buf, Mat& accum) {
    Mat recv_mat;
    recv_mat.LockedAttach(accum.Height(), accum.Width(), (DataType*) recv_buf,
                          accum.LDim());
    accum += recv_mat;
    return sizeof(DataType) * recv_mat.Height() * recv_mat.Width();
  };
  Mat& mat = dmat.Matrix();
  int max_recv_count = sizeof(DataType) * mat.Height() * mat.Width();
  comm->pe_ring_allreduce(
    comm->get_intermodel_comm(), mat, max_recv_count,
    std::function<uint8_t*(Mat&, IR, IR, int&, bool)>(send_transform),
    std::function<int(uint8_t*, Mat&)>(recv_transform),
    std::function<int(uint8_t*, Mat&)>(recv_apply_transform), true);
}

void test_ring_allreduce(lbann_comm* comm, DistMat& dmat) {
  auto send_transform =
    [] (Mat& mat, IR h, IR w, int& send_size, bool const_data) {
    auto to_send = mat(h, w);
    send_size = sizeof(DataType) * to_send.Height() * to_send.Width();
    return (uint8_t*) to_send.Buffer();
  };
  auto recv_transform =
    [] (uint8_t* recv_buf, Mat& accum) {
    Mat recv_mat;
    recv_mat.LockedAttach(accum.Height(), accum.Width(), (DataType*) recv_buf,
                          accum.LDim());
    accum = recv_mat;
    return sizeof(DataType) * recv_mat.Height() * recv_mat.Width();
  };
  auto recv_apply_transform =
    [] (uint8_t* recv_buf, Mat& accum) {
    Mat recv_mat;
    recv_mat.LockedAttach(accum.Height(), accum.Width(), (DataType*) recv_buf,
                          accum.LDim());
    accum += recv_mat;
    return sizeof(DataType) * recv_mat.Height() * recv_mat.Width();
  };
  Mat& mat = dmat.Matrix();
  int max_recv_count = sizeof(DataType) * mat.Height() * mat.Width();
  comm->ring_allreduce(
    comm->get_intermodel_comm(), mat, max_recv_count,
    std::function<uint8_t*(Mat&, IR, IR, int&, bool)>(send_transform),
    std::function<int(uint8_t*, Mat&)>(recv_transform),
    std::function<int(uint8_t*, Mat&)>(recv_apply_transform));
}

void print_stats(const std::vector<double>& times) {
  double sum = std::accumulate(times.begin() + 1, times.end(), 0.0);
  double mean = sum / (times.size() - 1);
  auto minmax = std::minmax_element(times.begin() + 1, times.end());
  double sqsum = 0.0;
  for (auto t = times.begin() + 1; t != times.end(); ++t) {
    sqsum += (*t - mean) * (*t - mean);
  }
  double stdev = std::sqrt(sqsum / (times.size() - 1));
  std::cout << "\tMean: " << mean << std::endl;
  std::cout << "\tMin: " << *(minmax.first) << std::endl;
  std::cout << "\tMax: " << *(minmax.second) << std::endl;
  std::cout << "\tStdev: " << stdev << std::endl;
  std::cout << "\tRaw: ";
  for (const auto& t : times) {
    std::cout << t << ", ";
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  El::Initialize(argc, argv);
  lbann_comm* comm = new lbann_comm(1);
  for (Int mat_size = 1; mat_size <= 16384; mat_size *= 2) {
    std::vector<double> mpi_times, rd_times, pe_ring_times, ring_times;
    // First trial is a warmup.
    for (int trial = 0; trial < num_trials + 1; ++trial) {
      DistMat rd_mat(comm->get_model_grid());
      El::Uniform(rd_mat, mat_size, mat_size, 0.0f, 1.0f);
      DistMat exact_mat(rd_mat);
      DistMat pe_ring_mat(rd_mat);
      DistMat ring_mat(rd_mat);
      comm->global_barrier();
      // Baseline.
      double start = get_time();
      comm->intermodel_sum_matrix(exact_mat);
      mpi_times.push_back(get_time() - start);
      comm->global_barrier();
      // Recursive doubling.
      start = get_time();
      test_rd_allreduce(comm, rd_mat);
      rd_times.push_back(get_time() - start);
      ASSERT_MAT_EQ(rd_mat.Matrix(), exact_mat.Matrix());
      comm->global_barrier();
      // Pairwise-exchange/ring.
      start = get_time();
      test_pe_ring_allreduce(comm, pe_ring_mat);
      pe_ring_times.push_back(get_time() - start);
      ASSERT_MAT_EQ(pe_ring_mat.Matrix(), exact_mat.Matrix());
      // Ring.
      start = get_time();
      test_ring_allreduce(comm, ring_mat);
      ring_times.push_back(get_time() - start);
      ASSERT_MAT_EQ(ring_mat.Matrix(), exact_mat.Matrix());
    }
    if (comm->am_world_master()) {
      std::cout << "MPI (" << mat_size << "x" << mat_size << "):" << std::endl;
      print_stats(mpi_times);
      std::cout << "RD (" << mat_size << "x" << mat_size << "):" << std::endl;
      print_stats(rd_times);
      std::cout << "PE/ring (" << mat_size << "x" << mat_size << "):" <<
        std::endl;
      print_stats(pe_ring_times);
      std::cout << "Ring (" << mat_size << "x" << mat_size << "):" << std::endl;
      print_stats(ring_times);
    }    
  }
  delete comm;
  El::Finalize();
  return 0;
}
