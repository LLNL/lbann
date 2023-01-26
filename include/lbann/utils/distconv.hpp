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

#ifndef LBANN_UTILS_DISTCONV_HPP
#define LBANN_UTILS_DISTCONV_HPP

#include "lbann_config.hpp"

#ifdef LBANN_HAS_DISTCONV

#include "El.hpp"
#include "lbann/comm_nb_request.hpp"
#include <vector>

#ifdef LBANN_DEBUG
#define DISTCONV_DEBUG
#endif

#include "distconv/distconv.hpp"
#include "distconv/dnn_backend/backend.hpp"
#include "distconv/tensor/algorithms.hpp"
#include "distconv/tensor/shuffle_mpi.hpp"
#include "distconv/tensor/shuffle_mpi_cuda.hpp"
#include "distconv/tensor/shuffle_mpi_cuda_al.hpp"
#include "distconv/tensor/tensor_mpi_cuda.hpp"
#include "distconv/util/util.hpp"
#ifdef DISTCONV_HAS_P2P
#include "distconv/tensor/shuffle_mpi_cuda_hybrid.hpp"
#include "distconv/tensor/shuffle_mpi_cuda_p2p.hpp"
#include "p2p/p2p.hpp"
#endif // DISTCONV_HAS_P2P

#include "lbann/layers/learning/distconv/distconv_layers.hpp"
#include "lbann/layers/math/distconv/distconv_matmul.hpp"

#ifdef LBANN_HAS_NVSHMEM
#include "lbann/layers/transform/distconv/distconv_scatter.hpp"
#include "lbann/layers/transform/distconv/distconv_gather.hpp"
#include "lbann/layers/transform/distconv/distconv_nvshmem_vector_addressing.hpp"
#endif // LBANN_HAS_NVSHMEM

#include "lbann/layers/misc/distconv/distconv_channelwise_softmax.hpp"

namespace lbann {

inline auto default_hydrogen_stream()
{
#if H2_HAS_CUDA
  return hydrogen::cuda::GetDefaultStream();
#elif H2_HAS_ROCM
  return hydrogen::rocm::GetDefaultStream();
#endif
}

// Forward Declarations
class lbann_comm;
class Layer;

namespace dc {

namespace tensor = ::distconv::tensor;
namespace util = ::distconv::util;

////////////////////////////////////////////////////////////
// Helper type aliases
////////////////////////////////////////////////////////////
using IntVector = ::distconv::IntVector;
using IndexVector = ::distconv::IndexVector;
using Shape = ::distconv::tensor::Shape;

using Dist = ::distconv::tensor::Distribution;

using LocaleMPI = ::distconv::tensor::LocaleMPI;

using AbsTensor = ::distconv::tensor::AbstractTensor;

template <typename TensorDataType>
using TensorHost = ::distconv::tensor::
  Tensor<TensorDataType, LocaleMPI, ::distconv::tensor::BaseAllocator>;

template <typename TensorDataType>
using TensorDev = ::distconv::tensor::
  Tensor<TensorDataType, LocaleMPI, ::distconv::tensor::CUDAAllocator>;

template <typename TensorDataType>
using TensorHostShuffler =
  ::distconv::tensor::TensorMPIShuffler<TensorDataType,
                                        ::distconv::tensor::BaseAllocator>;

template <typename TensorDataType>
using TensorShuffler =
  ::distconv::tensor::TensorMPICUDAShuffler<TensorDataType>;
template <typename TensorDataType>
using TensorShufflerAL =
  ::distconv::tensor::TensorMPICUDAShufflerAL<TensorDataType>;
#ifdef DISTCONV_HAS_P2P
template <typename TensorDataType>
using TensorShufflerP2P =
  ::distconv::tensor::TensorMPICUDAShufflerP2P<TensorDataType>;
template <typename TensorDataType>
using TensorShufflerHybrid =
  ::distconv::tensor::TensorMPICUDAShufflerHybrid<TensorDataType>;
#endif // DISTCONV_HAS_P2P

// Debug printing functions
using MPIPrintStreamDebug = ::distconv::util::MPIPrintStreamDebug;
using MPIPrintStreamError = ::distconv::util::MPIPrintStreamError;
using MPIPrintStreamInfo = ::distconv::util::MPIPrintStreamInfo;
using MPIPrintStreamWarning = ::distconv::util::MPIPrintStreamWarning;
using MPIRootPrintStreamDebug = ::distconv::util::MPIRootPrintStreamDebug;
using MPIRootPrintStreamError = ::distconv::util::MPIRootPrintStreamError;
using MPIRootPrintStreamInfo = ::distconv::util::MPIRootPrintStreamInfo;
using MPIRootPrintStreamWaning = ::distconv::util::MPIRootPrintStreamWarning;

// Distconv layer classes
using Backend = ::distconv::BackendDNNLib;
using ReLU = ::distconv::ReLU<Backend>;
using LeakyReLU = ::distconv::LeakyReLU<Backend>;
template <typename TensorDataType>
using Convolution = ::distconv::Convolution<Backend, TensorDataType>;
template <typename TensorDataType>
using ChannelwiseFullyConnected = ::distconv::ChannelwiseFullyConnected<Backend, TensorDataType>;
template <typename TensorDataType>
using Pooling = ::distconv::Pooling<Backend, TensorDataType>;
template <typename TensorDataType>
using BatchNormalization = ::distconv::BatchNormalization<Backend, TensorDataType>;
template <typename TensorDataType>
using MatMul = ::distconv::MatMul<Backend, TensorDataType>;
template <typename TensorDataType>
using ChannelwiseSoftmax = ::distconv::ChannelwiseSoftmax<Backend, TensorDataType>;
using Softmax = ::distconv::Softmax<Backend>;
using CrossEntropy = ::distconv::CrossEntropy<Backend>;
using MeanSquaredError = ::distconv::MeanSquaredError<Backend>;

using ::distconv::get_channel_dim;
using ::distconv::get_sample_dim;

int get_strided_mpi_rank(MPI_Comm comm);
MPI_Comm get_strided_mpi_comm(MPI_Comm comm);

/** Initialize Distconv
 */
void initialize(MPI_Comm comm);

/** Finalize Distconv
 */
void finalize();

/** Return MPI_Comm used for distconv

    Note that training only a single model is considered. This should
    be equal to MPI_COMM_WORLD.
 */
MPI_Comm get_mpi_comm();

/** Return the MPI rank
 */
int get_mpi_rank();

/** Return the number of MPI ranks
 */
int get_mpi_num_ranks();

/** Query if this rank is the root of the MPI communiator
 */
bool is_mpi_root();

/** Query rank stride
 */
int get_rank_stride();

/** Query if the execution is for performance evaluation
 */
bool evaluate_performance();

/** Query convolution forward algorithm name.
 */
std::string get_convolution_fwd_algorithm();

/** Query convolution backward data algorithm name.
 */
std::string get_convolution_bwd_data_algorithm();

/** Query convolution backward filter algorithm name.
 */
std::string get_convolution_bwd_filter_algorithm();

/** Query method for random number generation in synthetic data reader.
 */
std::string get_synthetic_data_reader_randgen();

/** Query the number of synthetic data to pre-generate.
 */
int get_number_of_pre_generated_synthetic_data();

/** Query if determinism is requested
 */
bool is_deterministic();

/** Query the number of partitions in the depth dimension.
 */
int get_number_of_io_partitions();

/** Query if Cosmoflow parallel I/O is enabled.
 */
bool is_cosmoflow_parallel_io_enabled();

#ifdef DISTCONV_HAS_P2P
/** Get p2p handle
 */
p2p::P2P& get_p2p();
#endif // DISTCONV_HAS_P2P

/** Get Aluminum host-transfer backend
 */
AlCommType& get_hosttransfer();

/** Get Distconv backend handle.
 */
Backend& get_backend();

/** Return a HaloExchangeMethod
 */
::distconv::HaloExchangeMethod get_halo_exchange_method();

template <typename TensorDataType>
TensorShuffler<TensorDataType>*
get_tensor_shuffler(const TensorDev<TensorDataType>& src,
                    const TensorDev<TensorDataType>& dst);

MPI_Comm get_input_comm(const lbann_comm& comm);

/** Return the MPI rank when reading input dataset
 */
int get_input_rank(const lbann_comm& comm);

/** Return Dist for data-parallel Hydrogen matrices
 */
Dist get_hydrogen_data_parallel_distribution(int num_dims);

template <typename Tensor>
void dump_tensor(const Tensor& t, const std::string& path)
{
  dc::MPIPrintStreamDebug() << "Dumping tensor to " << path;
  h2::gpu::sync();
  distconv::dump_tensor(t, path, true);
}

size_t get_workspace_capacity();

int get_num_dims(const Layer& layer);
int get_num_spatial_dims(const Layer& layer);

#ifndef LBANN_UTILS_DISTCONV_INSTANTIATE
#define PROTO(T)                                                               \
  extern template TensorShuffler<T>* get_tensor_shuffler<T>(                   \
    const TensorDev<T>&,                                                       \
    const TensorDev<T>&);

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF
#endif // LBANN_UTILS_DISTCONV_INSTANTIATE

} // namespace dc
} // namespace lbann

#endif // LBANN_HAS_DISTCONV
#endif // LBANN_UTILS_DISTCONV_HPP
