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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/distconv.hpp"
#include "lbann/utils/cudnn.hpp"
#include <cstdlib>

#ifdef LBANN_HAS_DISTCONV

using namespace distconv;

namespace lbann {
namespace dc {
namespace {
bool initialized = false;
MPI_Comm mpi_comm = MPI_COMM_NULL;
p2p::P2P *p2p_instance = nullptr;
Backend *backend_instance = nullptr;

bool options_set = false;
int opt_rank_stride = 1;
bool opt_enable_profile = false;
bool opt_skip_metrics_while_training = false;
bool opt_use_partial_aggregation_in_bn = false;
std::string opt_convolution_fwd_algorithm("DEFAULT");
std::string opt_convolution_bwd_data_algorithm("DEFAULT");
std::string opt_convolution_bwd_filter_algorithm("DEFAULT");

void set_options() {
  if (options_set) return;
  char *env = std::getenv("LBANN_DISTCONV_RANK_STRIDE");
  if (env) {
    opt_rank_stride = std::atoi(env);
  }
  if (std::getenv("LBANN_DISTCONV_PROFILE")) {
    opt_enable_profile = true;
  }
  if (std::getenv("LBANN_DISTCONV_SKIP_METRICS_WHILE_TRAINING")) {
    opt_skip_metrics_while_training = true;
  }
  if (std::getenv("LBANN_DISTCONV_USE_PARTIAL_AGGREGATION_IN_BN")) {
    opt_use_partial_aggregation_in_bn = true;
  }
  env = getenv("LBANN_DISTCONV_CONVOLUTION_FWD_ALGORITHM");
  if (env) {
    opt_convolution_fwd_algorithm = env;
  }
  env = getenv("LBANN_DISTCONV_CONVOLUTION_BWD_DATA_ALGORITHM");
  if (env) {
    opt_convolution_bwd_data_algorithm = env;
  }
  env = getenv("LBANN_DISTCONV_CONVOLUTION_BWD_FILTER_ALGORITHM");
  if (env) {
    opt_convolution_bwd_filter_algorithm = env;
  }
  options_set = true;
}

void print_options(std::ostream &os) {
  if (is_mpi_root()) {
    std::stringstream ss;
    ss << "LBANN/Distconv options\n";
    ss << "  rank_stride:" << opt_rank_stride << "\n";
    ss << "  enable_profile: " << opt_enable_profile << "\n";
    ss << "  skip_metrics_while_training: "
       << opt_skip_metrics_while_training << "\n";
    ss << "  use_partial_aggregation_in_bn: "
       << opt_use_partial_aggregation_in_bn
       << std::endl;
    os << ss.str();
  }
}

int get_number_of_local_ranks(MPI_Comm comm) {
  MPI_Comm local_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL, &local_comm);
  int local_comm_size;
  MPI_Comm_size(local_comm, &local_comm_size);
  MPI_Comm_free(&local_comm);
  return local_comm_size;
}

// P2P is only supported intra-node shuffling.
bool is_p2p_shuffle_feasible(const TensorDev &tensor) {
  const auto &dist = tensor.get_distribution();
  auto sample_proc_groups = dist.get_locale_shape().back();
  auto sample_size = tensor.get_shape().back();
  // Condition: The number of samples must be divisible by the size of
  // sample process groups
  if (sample_size % sample_proc_groups != 0) {
    return false;
  }
  // Condition: The number of local processes must be greater than or
  // equal to the number of processes of the spatial domain
  auto local_comm_size = get_number_of_local_ranks(
      tensor.get_locale().get_comm());
  auto spatial_proc_size = 1;
  for (int i = 0; i < TensorDev::num_spatial_dims; ++i) {
    spatial_proc_size *= dist.get_locale_shape()[i];
  }
  if (local_comm_size < spatial_proc_size) {
    return false;
  }
  // Condition: The number of local processes must be divisible by the
  // number of processes for the spatial domain
  if (local_comm_size % spatial_proc_size != 0) {
    return false;
  }
  return true;
}
} // namespace

MPI_Comm get_strided_mpi_comm(MPI_Comm comm) {
  // Assumes comm is in the packed order of nodes, i.e., let PPN be
  // the number of processes per node, the local rank is rank % PPN,
  // and the node rank is rank / PPN.
  set_options();
  int stride = opt_rank_stride;
  if (stride == 1) return comm;
  int rank;
  MPI_Comm_rank(comm, &rank);
  int num_ranks;
  MPI_Comm_size(comm, &num_ranks);
  int num_local_ranks = get_number_of_local_ranks(comm);
  assert_always(stride >= 1);
  assert0(num_local_ranks % stride);
  assert0(num_ranks % num_local_ranks);
  assert0(num_ranks % stride);
  int new_rank = rank / stride + (rank % stride) * (num_ranks / stride);
  MPIPrintStreamInfo() << "Mapping rank " << rank << " to " << new_rank;
  MPI_Comm new_comm;
  MPI_Comm_split(comm, 0, new_rank, &new_comm);
  return new_comm;
}

void initialize(MPI_Comm comm) {
  assert_always(!initialized);
  set_options();
  mpi_comm = comm;
  p2p_instance = new p2p::P2P(mpi_comm);
  auto &cudnn_h = lbann::cudnn::get_handle();
  cudaStream_t s;
  CHECK_CUDNN(cudnnGetStream(cudnn_h, &s));
  backend_instance = new Backend(mpi_comm, cudnn_h, s);
  print_options(std::cout);
  initialized = true;
}

void finalize() {
  if (initialized) {
    delete p2p_instance;
    p2p_instance = nullptr;
    delete backend_instance;
    backend_instance = nullptr;
    initialized = false;
  }
}

MPI_Comm get_mpi_comm() {
  return mpi_comm;
}

int get_mpi_rank() {
  int rank;
  MPI_Comm_rank(get_mpi_comm(), &rank);
  return rank;
}

int get_mpi_num_ranks() {
  int num_ranks;
  MPI_Comm_size(get_mpi_comm(), &num_ranks);
  return num_ranks;
}

bool is_mpi_root() {
  return get_mpi_rank() == 0;
}

int get_rank_stride() {
  return opt_rank_stride;
}

bool is_profiling_enabled() {
  return opt_enable_profile;
}

bool skip_metrics_while_training() {
  return opt_skip_metrics_while_training;
}

bool use_partial_aggregation_in_bn() {
  return opt_use_partial_aggregation_in_bn;
}

std::string get_convolution_fwd_algorithm() {
  return opt_convolution_fwd_algorithm;
}

std::string get_convolution_bwd_data_algorithm() {
  return opt_convolution_bwd_data_algorithm;
}

std::string get_convolution_bwd_filter_algorithm() {
  return opt_convolution_bwd_filter_algorithm;
}

p2p::P2P &get_p2p() {
  return *p2p_instance;
}

Backend &get_backend() {
  return *backend_instance;
}

cudaStream_t get_stream() {
  return get_backend().get_stream();
}

HaloExchangeMethod get_halo_exchange_method() {
  char *env = std::getenv("DISTCONV_HALO_EXCHANGE");
  if (!env) {
    // not specified
    return HaloExchangeMethod::MPI_DERIVED_TYPE;
  }
  std::string s(env);
  if (s == "P2P") {
    return HaloExchangeMethod::P2P;
  } else if (s == "MPI") {
    return HaloExchangeMethod::MPI;
  } else if (s == "MPI_DERIVED_TYPE") {
    return HaloExchangeMethod::MPI_DERIVED_TYPE;
  } else {
    LBANN_ERROR("Unknown value of environment variable DISTCONV_HALO_EXCHANGE");
  }
}

TensorShuffler *get_tensor_shuffler(const TensorDev &src,
                                    const TensorDev &dst) {
  // Use the P2P shuffler if possible. Otherwise, the default
  // MPI-based shuffler is returned.
  char *env = std::getenv("LBANN_DISTCONV_TENSOR_SHUFFLER");
  if (env && std::string(env) == "P2P") {
    bool src_feasible = is_p2p_shuffle_feasible(src);
    bool dst_feasible = is_p2p_shuffle_feasible(dst);
    if (!src_feasible) {
      MPIRootPrintStreamInfo()
          << "Unable to use P2P shuffler for source tensor";
    }
    if (!dst_feasible) {
      MPIRootPrintStreamInfo()
          << "Unable to use P2P shuffler for destination tensor";
    }
    if (src_feasible && dst_feasible) {
      MPIRootPrintStreamInfo() << "Using P2P shuffler";
      return new TensorShufflerP2P(src, dst, get_p2p());
    }
  }
  MPIRootPrintStreamInfo() << "Using MPI-based shuffler";
  return new TensorShuffler(src, dst);
}

} // namespace dc
} // namespace lbann

#endif // LBANN_HAS_DISTCONV
