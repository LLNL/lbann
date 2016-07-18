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
// lbann_comm_test.cpp - Tests lbann_comm
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include "lbann/lbann_comm.hpp"
#include "lbann_test_utils.hpp"

using namespace lbann;

// Configuration.
#define LBANN_COMM_TEST_PPM 2
#define LBANN_COMM_TEST_PROCS 8
#define LBANN_COMM_TEST_NUM_MODELS (LBANN_COMM_TEST_PROCS / LBANN_COMM_TEST_PPM)
#define LBANN_COMM_TEST_NROWS (2*LBANN_COMM_TEST_PPM)
#define LBANN_COMM_TEST_NCOLS (2*LBANN_COMM_TEST_PPM)

/** Initialize a new communicator. */
lbann_comm* init_comm() {
  return new lbann_comm(LBANN_COMM_TEST_PPM);
}

/** Destroy a communicator. */
void fini_comm(lbann_comm* comm) {
  delete comm;
}

/** Create a new matrix. */
void create_mat(DistMat& mat, DataType def = 1.0) {
  mat.Resize(LBANN_COMM_TEST_NROWS, LBANN_COMM_TEST_NCOLS);
  El::Fill(mat, def);
}

/** Validate every local entry of a matrix has a given value. */
void validate_mat(DistMat& mat, DataType expected) {
  for (int i = 0; i < mat.LocalHeight(); ++i) {
    for (int j = 0; j < mat.LocalWidth(); ++j) {
      ASSERT_EQ(mat.GetLocal(i, j), expected);
    }
  }
}

/** Verify we can successfully create/destroy a lbann_comm. */
void test_creation() {
  lbann_comm* comm = init_comm();
  ASSERT_TRUE(comm);
  fini_comm(comm);
}

/** Verify ranks are all reasonable. */
void test_ranks() {
  lbann_comm* comm = init_comm();
  int model_rank = comm->get_model_rank();
  ASSERT_TRUE(model_rank >= 0);
  ASSERT_TRUE(model_rank < LBANN_COMM_TEST_PROCS);
  int rank_in_model = comm->get_rank_in_model();
  ASSERT_TRUE(rank_in_model >= 0);
  ASSERT_TRUE(rank_in_model < LBANN_COMM_TEST_PPM);
  ASSERT_TRUE(comm->get_num_models() == LBANN_COMM_TEST_NUM_MODELS);
  fini_comm(comm);
}

/** Verify the grid is valid. */
void test_grid() {
  lbann_comm* comm = init_comm();
  Grid& grid = comm->get_model_grid();
  ASSERT_TRUE(grid.Size() == LBANN_COMM_TEST_PPM);
  fini_comm(comm);
}

/** Verify matrices work properly. */
void test_mat() {
  lbann_comm* comm = init_comm();
  DistMat mat(comm->get_model_grid());
  create_mat(mat, 42.0);
  ASSERT_EQ(mat.DistRank(), comm->get_rank_in_model());
  // Only validates the local portion.
  validate_mat(mat, 42.0);
  fini_comm(comm);
}

/** Verify inter-model matrix summation works. */
void test_intermodel_sum_matrix() {
  lbann_comm* comm = init_comm();
  DistMat mat(comm->get_model_grid());
  create_mat(mat);
  comm->intermodel_barrier();
  comm->intermodel_sum_matrix(mat);
  validate_mat(mat, (float) LBANN_COMM_TEST_NUM_MODELS);
  fini_comm(comm);
}

/** Verify inter-model matrix broadcast works. */
void test_intermodel_broadcast_matrix() {
  lbann_comm* comm = init_comm();
  DistMat mat(comm->get_model_grid());
  create_mat(mat, (float) comm->get_model_rank());
  comm->intermodel_barrier();
  comm->intermodel_broadcast_matrix(mat, 0);
  validate_mat(mat, (float) 0);  // Should come from the 0'th model.
  fini_comm(comm);
}

/** Verify sends/receives of blob data work. */
void test_send_recv_blob() {
  lbann_comm* comm = init_comm();
  int send_model = (comm->get_model_rank() + 1) % LBANN_COMM_TEST_NUM_MODELS;
  int recv_model = (comm->get_model_rank() + LBANN_COMM_TEST_NUM_MODELS - 1) %
    LBANN_COMM_TEST_NUM_MODELS;
  int send_data = 42;
  int recv_data = 0;
  // Test sends/recvs with full model/rank spec.
  comm->send(&send_data, 1, send_model, comm->get_rank_in_model());
  comm->recv(&recv_data, 1, recv_model, comm->get_rank_in_model());
  ASSERT_EQ(send_data, recv_data);
  // Test sends/recvs with only the model.
  recv_data = 0;
  comm->send(&send_data, 1, send_model);
  comm->recv(&recv_data, 1, recv_model);
  ASSERT_EQ(send_data, recv_data);
  // Test with receiving from anywhere.
  recv_data = 0;
  comm->send(&send_data, 1, send_model, comm->get_rank_in_model());
  comm->recv(&recv_data, 1);
  ASSERT_EQ(send_data, recv_data);
  fini_comm(comm);
}

/** Verify sends/receives of matrices work. */
void test_send_recv_mat() {
  lbann_comm* comm = init_comm();
  int send_model = (comm->get_model_rank() + 1) % LBANN_COMM_TEST_NUM_MODELS;
  int recv_model = (comm->get_model_rank() + LBANN_COMM_TEST_NUM_MODELS - 1) %
    LBANN_COMM_TEST_NUM_MODELS;
  DistMat send_mat(comm->get_model_grid());
  DistMat recv_mat(comm->get_model_grid());
  create_mat(send_mat, (float) comm->get_model_rank());
  create_mat(recv_mat, (float) 42.0);
  comm->send(send_mat, send_model, comm->get_rank_in_model());
  comm->recv(recv_mat, recv_model, comm->get_rank_in_model());
  validate_mat(recv_mat, (float) recv_model);
  El::Fill(recv_mat, (float) 42.0);
  comm->send(send_mat, send_model);
  comm->recv(recv_mat, recv_model);
  validate_mat(recv_mat, (float) recv_model);
  El::Fill(recv_mat, (float) 42.0);
  comm->send(send_mat, send_model, comm->get_rank_in_model());
  comm->recv(recv_mat);
  validate_mat(recv_mat, (float) recv_model);
  fini_comm(comm);
}

/** Verify broadcasting blob data works. */
void test_broadcast_blob() {
  lbann_comm* comm = init_comm();
  int send_data = 0;
  if (comm->get_model_rank() == 0) {
    send_data = 42;
  }
  std::vector<int> dests(3);
  dests[0] = comm->get_world_rank(1, comm->get_rank_in_model());
  dests[1] = comm->get_world_rank(2, comm->get_rank_in_model());
  dests[2] = comm->get_world_rank(3, comm->get_rank_in_model());
  comm->broadcast(&send_data, 1, dests,
                  comm->get_world_rank(0, comm->get_rank_in_model()));
  ASSERT_EQ(send_data, 42);
  fini_comm(comm);
}

/** Verify broadcasting matrices works. */
void test_broadcast_mat() {
  lbann_comm* comm = init_comm();
  DistMat mat(comm->get_model_grid());
  if (comm->get_model_rank() == 0) {
    create_mat(mat, (float) 42.0);
  } else {
    create_mat(mat, (float) 0.0);
  }
  std::vector<int> dests(3);
  dests[0] = comm->get_world_rank(1, comm->get_rank_in_model());
  dests[1] = comm->get_world_rank(2, comm->get_rank_in_model());
  dests[2] = comm->get_world_rank(3, comm->get_rank_in_model());
  comm->broadcast(mat, dests,
                  comm->get_world_rank(0, comm->get_rank_in_model()));
  validate_mat(mat, (float) 42.0);
  fini_comm(comm);
}

// Run with srun -n8 --tasks-per-node=12 
int main(int argc, char** argv) {
  El::Initialize(argc, argv);
  ASSERT_EQ(El::mpi::Size(El::mpi::COMM_WORLD), LBANN_COMM_TEST_PROCS);
  try {
    test_creation();
    test_ranks();
    test_grid();
    test_mat();
    test_intermodel_sum_matrix();
    test_intermodel_broadcast_matrix();
    test_send_recv_blob();
    test_send_recv_mat();
    test_broadcast_blob();
    test_broadcast_mat();
    El::mpi::Barrier(El::mpi::COMM_WORLD);
    if (El::mpi::Rank(El::mpi::COMM_WORLD) == 0) {
      std::cout << "All tests passed" << std::endl;
    }
  } catch (exception& e) {
    ReportException(e);
  }
  El::Finalize();
  return 0;
}
