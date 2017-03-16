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
// lbann_comm .hpp .cpp - LBANN communication utilities
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann_comm.hpp"
#include "lbann/utils/lbann_exception.hpp"
#include "mpi.h"
#include <sstream>

using namespace std;
using namespace El;

lbann::lbann_comm::lbann_comm(int _procs_per_model) :
  procs_per_model(_procs_per_model), num_model_barriers(0),
  num_intermodel_barriers(0), num_global_barriers(0), bytes_sent(0),
  bytes_received(0) {

  // Initialize parameters
  int world_size = mpi::Size(mpi::COMM_WORLD);
  if (procs_per_model == 0) {
    procs_per_model = world_size;
  }
  num_models = world_size / procs_per_model;
  model_rank = mpi::Rank(mpi::COMM_WORLD) / procs_per_model;
  rank_in_model = mpi::Rank(mpi::COMM_WORLD) % procs_per_model;

  // Check if parameters are valid
  if (procs_per_model > world_size) {
    stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: Not enough processes to create one model; procs_per_model: "
        << procs_per_model << " is larger than world_size: " << world_size;
    throw lbann_exception(err.str());
  }
  if (world_size % procs_per_model != 0) {
    stringstream err;
    err << __FILE__ << " " << __LINE__ 
        << " :: Procs per model does not divide total number of procs; procs_per_model: " 
        << procs_per_model << " total number of procs (world size): " << world_size;
    throw lbann_exception(err.str());
  }

  // Initialize model and intermodel communicators
  mpi::Split(mpi::COMM_WORLD, model_rank, rank_in_model, model_comm);
  mpi::Split(mpi::COMM_WORLD, rank_in_model, model_rank, intermodel_comm);

  // Initialize Elemental grid
  grid = new Grid(model_comm);

  // Initialize node communicators
  setup_node_comm();
  procs_per_node = mpi::Size(node_comm);
  rank_in_node = mpi::Rank(node_comm);
  
}

lbann::lbann_comm::~lbann_comm() {
  delete grid;
  mpi::Free(model_comm);
  mpi::Free(intermodel_comm);
}

void lbann::lbann_comm::intermodel_sum_matrix(Mat& mat) {
  bytes_sent += sizeof(DataType) * mat.Height() * mat.Width();
  AllReduce(mat, intermodel_comm, mpi::SUM);
  bytes_received += sizeof(DataType) * mat.Height() * mat.Width();
}

void lbann::lbann_comm::intermodel_sum_matrix(DistMat& mat) {
  bytes_sent += sizeof(DataType) * mat.LocalHeight() * mat.LocalWidth();
  AllReduce(mat, intermodel_comm, mpi::SUM);
  bytes_received += sizeof(DataType) * mat.LocalHeight() * mat.LocalWidth();
}

/*void lbann::lbann_comm::nb_intermodel_sum_matrix(Mat& mat, mpi::Request& req) {
  MPI_Iallreduce(MPI_IN_PLACE, mat.Buffer(),
                 mat.Height() * mat.Width(), DataTypeMPI, MPI_SUM,
                 intermodel_comm.comm, &req);
}

void lbann::lbann_comm::nb_intermodel_sum_matrix(DistMat& mat,
                                                 mpi::Request& req) {
  // Note: This reaches into the Elemental internals where presently
  // mpi::Request is a typedef of MPI_Request and the MPI communicator
  // is mpi::Comm::comm.
  MPI_Iallreduce(MPI_IN_PLACE, mat.Buffer(),
                 mat.LocalHeight() * mat.LocalWidth(), DataTypeMPI, MPI_SUM,
                 intermodel_comm.comm, &req);
                 }*/

void lbann::lbann_comm::intermodel_broadcast_matrix(Mat& mat, int root) {
  Broadcast(mat, intermodel_comm, root);
}

void lbann::lbann_comm::intermodel_broadcast_matrix(DistMat& mat, int root) {
  Broadcast(mat, intermodel_comm, root);
}

/*void lbann::lbann_comm::nb_intermodel_broadcast_matrix(Mat& mat, int root,
                                                       mpi::Request& req) {
  MPI_Ibcast(mat.Buffer(), mat.Height() * mat.Width(), DataTypeMPI, root,
             intermodel_comm.comm, &req);
}

void lbann::lbann_comm::nb_intermodel_broadcast_matrix(DistMat& mat, int root,
                                                       mpi::Request& req) {
  MPI_Ibcast(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), DataTypeMPI,
             root, intermodel_comm.comm, &req);
             }*/

void lbann::lbann_comm::intermodel_barrier() {
  ++num_intermodel_barriers;
  mpi::Barrier(intermodel_comm);
}

void lbann::lbann_comm::model_barrier() {
  ++num_model_barriers;
  mpi::Barrier(model_comm);
}

void lbann::lbann_comm::global_barrier() {
  ++num_global_barriers;
  mpi::Barrier(mpi::COMM_WORLD);
}

void lbann::lbann_comm::send(Mat& mat, int model, int rank) {
  send(mat.Buffer(), mat.Height() * mat.Width(), model, rank);
}

void lbann::lbann_comm::send(DistMat& mat, int model, int rank) {
  send(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), model, rank);
}

void lbann::lbann_comm::nb_send(Mat& mat, int model, int rank,
                                mpi::Request<DataType>& req) {
  nb_send(mat.Buffer(), mat.Height() * mat.Width(), model, rank, req);
}

void lbann::lbann_comm::nb_send(DistMat& mat, int model, int rank,
                                mpi::Request<DataType>& req) {
  nb_send(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), model, rank, req);
}

void lbann::lbann_comm::recv(Mat& mat, int model, int rank) {
  recv(mat.Buffer(), mat.Height() * mat.Width(), model, rank);
}

void lbann::lbann_comm::recv(DistMat& mat, int model, int rank) {
  recv(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), model, rank);
}

void lbann::lbann_comm::recv(Mat& mat) {
  recv(mat.Buffer(), mat.Height() * mat.Width());
}

void lbann::lbann_comm::recv(DistMat& mat) {
  recv(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth());
}

void lbann::lbann_comm::nb_recv(Mat& mat, int model, int rank,
                                mpi::Request<DataType>& req) {
  nb_recv(mat.Buffer(), mat.Height() * mat.Width(), model, rank, req);
}

void lbann::lbann_comm::nb_recv(DistMat& mat, int model, int rank,
                                mpi::Request<DataType>& req) {
  nb_recv(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), model, rank, req);
}

void lbann::lbann_comm::nb_recv(Mat& mat, mpi::Request<DataType>& req) {
  nb_recv(mat.Buffer(), mat.Height() * mat.Width(), req);
}

void lbann::lbann_comm::nb_recv(DistMat& mat, mpi::Request<DataType>& req) {
  nb_recv(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), req);
}

void lbann::lbann_comm::broadcast(Mat& mat,
                                  std::vector<int>& dests, int root) {
  broadcast(mat.Buffer(), mat.Height() * mat.Width(), dests, root);
}

void lbann::lbann_comm::broadcast(DistMat& mat,
                                  std::vector<int>& dests, int root) {
  broadcast(mat.Buffer(), mat.LocalHeight() * mat.LocalWidth(), dests, root);
}

void lbann::lbann_comm::setup_node_comm() {
 
  // Get string specifying compute node
  char node_name[MPI_MAX_PROCESSOR_NAME];
  int node_name_len;
  int status = MPI_Get_processor_name(node_name, &node_name_len);
  if(status != MPI_SUCCESS) {
    throw lbann_exception("lbann_comm: error in MPI_Get_processor_name");
  }

  // Hash the node names and split MPI processes
  // TODO: Generate random salt. The salt must be shared across all ranks.
  std::string node_string = "hXFgQFNrqyL1mIsq";
  node_string += node_name;
  int hash = std::hash<std::string>()(node_string);
  hash = hash & 0x7fffffff; // Make sure that this is a positive int to avoid crashing MPI
  mpi::Comm hash_comm;
  mpi::Split(mpi::COMM_WORLD, hash, mpi::Rank(mpi::COMM_WORLD), hash_comm);

  // Hash the node names again and split MPI processes
  // TODO: Generate random salt. The salt must be shared across all ranks.
  node_string = "BGqXbNZqxuXBV5lm";
  node_string += node_name;
  hash = std::hash<std::string>()(node_string);
  hash = hash & 0x7fffffff; // Make sure that this is a positive int to avoid crashing MPI
  mpi::Split(hash_comm, hash, mpi::Rank(mpi::COMM_WORLD), node_comm);

}
