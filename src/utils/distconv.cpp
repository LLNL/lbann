////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#define LBANN_UTILS_DISTCONV_INSTANTIATE
#include "lbann/utils/distconv.hpp"
#ifdef LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/helpers.hpp"
#endif // LBANN_HAS_DNN_LIB
#include "lbann/layers/layer.hpp"
#include <cstdlib>

#ifdef LBANN_HAS_DISTCONV

using namespace distconv;

namespace lbann {
namespace dc {
namespace {
bool initialized = false;
MPI_Comm mpi_comm = MPI_COMM_NULL;
#ifdef DISTCONV_HAS_P2P
p2p::P2P* p2p_instance = nullptr;
#endif // DISTCONV_HAS_P2P
AlCommType* hosttransfer_comm_instance = nullptr;
Backend* backend_instance = nullptr;

bool options_set = false;
#ifdef DISTCONV_HAS_P2P
HaloExchangeMethod opt_halo_exchange = HaloExchangeMethod::HYBRID;
ShuffleMethod opt_tensor_shuffler = ShuffleMethod::HYBRID;
#else
HaloExchangeMethod opt_halo_exchange = HaloExchangeMethod::AL;
ShuffleMethod opt_tensor_shuffler = ShuffleMethod::AL;
#endif // DISTCONV_HAS_P2P
int opt_rank_stride = 1;
bool opt_evaluate_performance = false;
std::string opt_convolution_fwd_algorithm("DEFAULT");
std::string opt_convolution_bwd_data_algorithm("DEFAULT");
std::string opt_convolution_bwd_filter_algorithm("DEFAULT");
// Allowed values: MINSTD, MT, and ONE
std::string opt_synthetic_data_reader_randgen("MINSTD");
int opt_num_pre_generated_synthetic_data = 0;
bool opt_deterministic = false;
int opt_num_io_partitions = 1;
bool opt_cosmoflow_parallel_io = false;

void set_options()
{
  if (options_set)
    return;
  char* env = nullptr;
  env = std::getenv("LBANN_DISTCONV_HALO_EXCHANGE");
  if (env) {
    opt_halo_exchange = ::distconv::GetHaloExchangeMethod(env);
  }
  env = std::getenv("LBANN_DISTCONV_TENSOR_SHUFFLER");
  if (env) {
    opt_tensor_shuffler = ::distconv::GetShuffleMethod(env);
  }
  env = std::getenv("LBANN_DISTCONV_RANK_STRIDE");
  if (env) {
    opt_rank_stride = std::atoi(env);
  }
  if (std::getenv("LBANN_DISTCONV_EVALUATE_PERFORMANCE")) {
    opt_evaluate_performance = true;
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
  env = getenv("LBANN_DISTCONV_SYNTHETIC_DATA_READER_RANDGEN");
  if (env) {
    opt_synthetic_data_reader_randgen = env;
  }
  env = getenv("LBANN_DISTCONV_NUM_PRE_GENERATED_SYNTHETIC_DATA");
  if (env) {
    opt_num_pre_generated_synthetic_data = atoi(env);
  }
  env = getenv("LBANN_DISTCONV_DETERMINISTIC");
  if (env) {
    opt_deterministic = true;
  }
  env = getenv("LBANN_DISTCONV_NUM_IO_PARTITIONS");
  if (env) {
    opt_num_io_partitions = std::atoi(env);
  }
  env = getenv("LBANN_DISTCONV_COSMOFLOW_PARALLEL_IO");
  if (env) {
    opt_cosmoflow_parallel_io = true;
  }
  options_set = true;
}

void print_options(std::ostream& os)
{
  if (is_mpi_root()) {
    std::stringstream ss;
    ss << "LBANN/Distconv options\n";
    ss << "  halo_exchange:" << opt_halo_exchange << "\n";
    ss << "  tensor_shuffler:" << opt_tensor_shuffler << "\n";
    ss << "  rank_stride:" << opt_rank_stride << "\n";
    ss << "  evaluate_performance: " << opt_evaluate_performance << "\n";
    ss << "  convolution_fwd_algorithm: " << opt_convolution_fwd_algorithm
       << std::endl;
    ss << "  convolution_bwd_data_algorithm: "
       << opt_convolution_bwd_data_algorithm << std::endl;
    ss << "  convolution_bwd_filter_algorithm: "
       << opt_convolution_bwd_filter_algorithm << std::endl;
    ss << "  synthetic_data_reader_randgen: "
       << opt_synthetic_data_reader_randgen << std::endl;
    ss << "  num_pre_generated_synthetic_data: "
       << opt_num_pre_generated_synthetic_data << std::endl;
    ss << "  deterministic: " << opt_deterministic << std::endl;
    ss << "  num_io_partitions: " << opt_num_io_partitions << std::endl;
    ss << "  cosmoflow_parallel_io: " << opt_cosmoflow_parallel_io << std::endl;
    os << ss.str();
  }
}

int get_number_of_local_ranks(MPI_Comm comm)
{
  MPI_Comm local_comm;
  MPI_Comm_split_type(comm,
                      MPI_COMM_TYPE_SHARED,
                      0,
                      MPI_INFO_NULL,
                      &local_comm);
  int local_comm_size;
  MPI_Comm_size(local_comm, &local_comm_size);
  MPI_Comm_free(&local_comm);
  return local_comm_size;
}

// P2P is only supported intra-node shuffling.
template <typename Tensor>
bool is_p2p_shuffle_feasible(const Tensor& tensor)
{
  const auto& dist = tensor.get_distribution();
  auto sample_proc_groups = dist.get_locale_shape()[dc::get_sample_dim()];
  auto sample_size = tensor.get_shape().back();
  // Condition: The number of samples must be divisible by the size of
  // sample process groups
  if (sample_size % sample_proc_groups != 0) {
    return false;
  }
  // Condition: The number of local processes must be greater than or
  // equal to the number of processes of the spatial domain
  auto local_comm_size =
    get_number_of_local_ranks(tensor.get_locale().get_comm());
  auto spatial_proc_size = 1;
  for (int i = 0; i < tensor.get_num_spatial_dims(); ++i) {
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

#ifdef DISTCONV_HAS_P2P
void* shuffler_src_buf = nullptr;
size_t shuffler_src_buf_size = 0;
void* shuffler_dst_buf = nullptr;
size_t shuffler_dst_buf_size = 0;

template <typename TensorDataType>
TensorDataType* get_shuffler_src_buf(const TensorDev<TensorDataType>& tensor)
{
  // Allocate if null
  if (shuffler_src_buf == nullptr) {
    shuffler_src_buf_size =
      TensorShuffler<TensorDataType>::get_buf_size(tensor);
    MPIPrintStreamDebug() << "Allocating shared shuffler buffer of size "
                          << shuffler_src_buf_size;
    DISTCONV_GPU_MALLOC(&shuffler_src_buf, shuffler_src_buf_size);
  }
  // Returns the pre-allocated memory if it's large enough
  size_t required_size = TensorShuffler<TensorDataType>::get_buf_size(tensor);
  if (required_size <= shuffler_src_buf_size) {
    MPIPrintStreamDebug() << "Using shared shuffler buffer";
    return static_cast<TensorDataType*>(shuffler_src_buf);
  }
  else {
    return nullptr;
  }
}

template <typename TensorDataType>
TensorDataType* get_shuffler_dst_buf(const TensorDev<TensorDataType>& tensor)
{
  // Allocate if null
  if (shuffler_dst_buf == nullptr) {
    shuffler_dst_buf_size =
      TensorShuffler<TensorDataType>::get_buf_size(tensor);
    MPIPrintStreamDebug() << "Allocating shared shuffler buffer of size "
                          << shuffler_src_buf_size;
    DISTCONV_GPU_MALLOC(&shuffler_dst_buf, shuffler_dst_buf_size);
  }
  size_t required_size = TensorShuffler<TensorDataType>::get_buf_size(tensor);
  // Returns the pre-allocated memory if it's large enough
  if (required_size <= shuffler_dst_buf_size) {
    MPIPrintStreamDebug() << "Using shared shuffler buffer";
    return static_cast<TensorDataType*>(shuffler_dst_buf);
  }
  else {
    return nullptr;
  }
}
void delete_shuffler_buffers()
{
  if (shuffler_src_buf) {
    DISTCONV_CHECK_GPU(GPU_FREE(shuffler_src_buf));
    shuffler_src_buf = nullptr;
  }
  if (shuffler_dst_buf) {
    DISTCONV_CHECK_GPU(GPU_FREE(shuffler_dst_buf));
    shuffler_dst_buf = nullptr;
  }
}
#endif
} // namespace

int get_strided_mpi_rank(MPI_Comm comm)
{
  // Assumes comm is in the packed order of nodes, i.e., let PPN be
  // the number of processes per node, the local rank is rank % PPN,
  // and the node rank is rank / PPN.
  set_options();
  int stride = opt_rank_stride;
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (stride == 1)
    return rank;
  int num_ranks;
  MPI_Comm_size(comm, &num_ranks);
  int num_local_ranks = get_number_of_local_ranks(comm);
  assert_always(stride >= 1);
  assert0(num_ranks % num_local_ranks);
  assert0(num_ranks % stride);
  int new_rank = rank / stride + (rank % stride) * (num_ranks / stride);
  return new_rank;
}

MPI_Comm get_strided_mpi_comm(MPI_Comm comm)
{
  set_options();
  int stride = opt_rank_stride;
  if (stride == 1)
    return comm;
  int rank;
  MPI_Comm_rank(comm, &rank);
  int new_rank = get_strided_mpi_rank(comm);
  MPIPrintStreamInfo() << "Mapping rank " << rank << " to " << new_rank;
  MPI_Comm new_comm;
  MPI_Comm_split(comm, 0, new_rank, &new_comm);
  return new_comm;
}

void initialize(MPI_Comm comm)
{
  assert_always(!initialized);
  set_options();
  mpi_comm = comm;
#ifdef DISTCONV_HAS_P2P
  p2p_instance = new p2p::P2P(mpi_comm);
#endif // DISTCONV_HAS_P2P
  hosttransfer_comm_instance =
    new AlCommType(mpi_comm, default_hydrogen_stream());
  ::distconv::backend::Options backend_opts;
  backend_opts.m_deterministic = opt_deterministic;
  backend_instance = new Backend(mpi_comm,
                                 lbann::dnn_lib::get_handle(),
                                 default_hydrogen_stream(),
                                 backend_opts);
  print_options(std::cout);
  initialized = true;
}

void finalize()
{
  if (initialized) {
#ifdef DISTCONV_HAS_P2P
    delete p2p_instance;
    p2p_instance = nullptr;
#endif // DISTCONV_HAS_P2P
    delete backend_instance;
    backend_instance = nullptr;
    initialized = false;
#ifdef DISTCONV_HAS_P2P
    delete_shuffler_buffers();
#endif
  }
}

MPI_Comm get_mpi_comm() { return mpi_comm; }

int get_mpi_rank()
{
  int rank;
  MPI_Comm_rank(get_mpi_comm(), &rank);
  return rank;
}

int get_mpi_num_ranks()
{
  int num_ranks;
  MPI_Comm_size(get_mpi_comm(), &num_ranks);
  return num_ranks;
}

bool is_mpi_root() { return get_mpi_rank() == 0; }

int get_rank_stride() { return opt_rank_stride; }

bool evaluate_performance() { return opt_evaluate_performance; }

std::string get_convolution_fwd_algorithm()
{
  return opt_convolution_fwd_algorithm;
}

std::string get_convolution_bwd_data_algorithm()
{
  return opt_convolution_bwd_data_algorithm;
}

std::string get_convolution_bwd_filter_algorithm()
{
  return opt_convolution_bwd_filter_algorithm;
}

std::string get_synthetic_data_reader_randgen()
{
  return opt_synthetic_data_reader_randgen;
}

int get_number_of_pre_generated_synthetic_data()
{
  return opt_num_pre_generated_synthetic_data;
}

bool is_deterministic() { return opt_deterministic; }

int get_number_of_io_partitions() { return opt_num_io_partitions; }

bool is_cosmoflow_parallel_io_enabled() { return opt_cosmoflow_parallel_io; }

AlCommType& get_hosttransfer() { return *hosttransfer_comm_instance; }

Backend& get_backend() { return *backend_instance; }

HaloExchangeMethod get_halo_exchange_method() { return opt_halo_exchange; }

template <typename TensorDataType>
TensorShuffler<TensorDataType>*
get_tensor_shuffler(const TensorDev<TensorDataType>& src,
                    const TensorDev<TensorDataType>& dst)
{
  if (opt_tensor_shuffler == ShuffleMethod::AL) {
    return new TensorShufflerAL<TensorDataType>(src, dst, get_hosttransfer());
  }
  else if (opt_tensor_shuffler == ShuffleMethod::MPI) {
    return new TensorShuffler<TensorDataType>(src, dst);
#ifdef DISTCONV_HAS_P2P
  }
  else if (opt_tensor_shuffler == ShuffleMethod::HYBRID) {
    return new TensorShufflerHybrid<TensorDataType>(src,
                                                    dst,
                                                    *p2p_instance,
                                                    get_hosttransfer(),
                                                    get_shuffler_src_buf(src),
                                                    get_shuffler_dst_buf(dst));
  }
  else if (opt_tensor_shuffler == ShuffleMethod::P2P) {
    bool src_feasible = is_p2p_shuffle_feasible(src);
    bool dst_feasible = is_p2p_shuffle_feasible(dst);
    if (!src_feasible) {
      LBANN_ERROR("P2P shuffler requested but not possible; ",
                  "inter-node communication is required for source tennsor");
    }
    if (!dst_feasible) {
      LBANN_ERROR(
        "P2P shuffler requested but not possible; ",
        "inter-node communication is required for destination tennsor");
    }
    return new TensorShufflerP2P<TensorDataType>(src,
                                                 dst,
                                                 *p2p_instance,
                                                 get_shuffler_src_buf(src),
                                                 get_shuffler_dst_buf(dst));
#endif // DISTCONV_HAS_P2P
  }
  else {
    LBANN_ERROR("Unsupported shuffler method: ", opt_tensor_shuffler);
  }
}

MPI_Comm get_input_comm(const lbann_comm& comm)
{
  if (!is_cosmoflow_parallel_io_enabled() || get_rank_stride() == 1) {
    return comm.get_trainer_comm().GetMPIComm();
  }
  else {
    return get_mpi_comm();
  }
}

int get_input_rank(const lbann_comm& comm)
{
  if (!is_cosmoflow_parallel_io_enabled() || get_rank_stride() == 1) {
    return comm.get_rank_in_trainer();
  }
  else {
    return get_mpi_rank();
  }
}

Dist get_hydrogen_data_parallel_distribution(int num_dims)
{
  using ::distconv::index_t;
  // When rank stride is 1, the distribution is just sample
  // distribution. When it's greater than 1, multiple consecutive
  // ranks of length rank stride share a split in the first
  // dimension. It is assumed that LBANN uses only the
  // NUM_RANKS/STRIDE ranks in a data-parallel input layer to read
  // training data.
  dc::Shape sample_locale_shape(num_dims, 1);
  sample_locale_shape[0] = static_cast<index_t>(dc::get_rank_stride());
  sample_locale_shape[-1] =
    static_cast<index_t>(dc::get_mpi_num_ranks() / dc::get_rank_stride());
  auto sample_split_shape = sample_locale_shape;
  sample_split_shape[0] = 1;
  auto sample_dist =
    dc::Dist::make_shared_distribution(sample_locale_shape, sample_split_shape);
  return sample_dist;
}

size_t get_workspace_capacity()
{
  size_t available, total;
#if H2_HAS_CUDA
  FORCE_CHECK_CUDA(cudaMemGetInfo(&available, &total));
#elif H2_HAS_ROCM
  FORCE_CHECK_ROCM(hipMemGetInfo(&available, &total));
#endif
  size_t workspace_capacity = available;
  // set aside some space for shuffling, halo exchange, etc.
  workspace_capacity -= 1 << 28;
  dc::MPIRootPrintStreamInfo()
    << "Current available memory: " << available << " ("
    << int(available / 1024.0 / 1024.0)
    << " MB), workspace: " << workspace_capacity << " ("
    << int(workspace_capacity / 1024.0 / 1024.0) << " MB)";
  return workspace_capacity;
}

int get_num_dims(const Layer& layer)
{
  // Use the dimension of either input or output data.
  auto nd = layer.get_num_parents() > 0 ? layer.get_input_dims().size()
                                        : layer.get_output_dims().size();
  nd += 1; // input and output dimensions do not have the sample dimension.
  if (!(nd == 4 || nd == 5)) {
    LBANN_ERROR(layer.get_name(), ": Invalid number of dimensions: ", nd);
  }
  return nd;
}

int get_num_spatial_dims(const Layer& layer) { return get_num_dims(layer) - 2; }

#define PROTO(T)                                                               \
  template TensorShuffler<T>* get_tensor_shuffler<T>(const TensorDev<T>&,      \
                                                     const TensorDev<T>&);

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#define LBANN_INSTANTIATE_DOUBLE
#include "lbann/macros/instantiate.hpp"

} // namespace dc
} // namespace lbann

#endif // LBANN_HAS_DISTCONV
