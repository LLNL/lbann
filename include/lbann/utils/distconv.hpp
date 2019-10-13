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

#ifndef LBANN_UTILS_DISTCONV_HPP
#define LBANN_UTILS_DISTCONV_HPP

#include "lbann_config.hpp"
#include "El.hpp"
#include "lbann/comm.hpp"
#include <vector>

#ifdef LBANN_HAS_DISTCONV

#ifdef LBANN_DEBUG
#define DISTCONV_DEBUG
#endif

#define DISTCONV_HAS_CUDNN

//#define DISTCONV_ZERO_OUT_ERROR_SIGNALS
// temporary workaround
#define DISTCONV_USE_SAME_RELU_CALL_AS_LBANN


#include "distconv/distconv.hpp"
#include "distconv/tensor/tensor_mpi_cuda.hpp"
#include "distconv/tensor/shuffle_mpi.hpp"
#include "distconv/tensor/shuffle_mpi_cuda.hpp"
#include "distconv/tensor/shuffle_mpi_cuda_p2p.hpp"
#include "distconv/tensor/shuffle_mpi_cuda_al.hpp"
#include "distconv/tensor/shuffle_mpi_cuda_hybrid.hpp"
#include "distconv/tensor/algorithms.hpp"
#include "distconv/util/util.hpp"
#include "p2p/p2p.hpp"

namespace lbann {
namespace dc {

static constexpr int num_dims = LBANN_DISTCONV_NUM_DIMS;
static constexpr int num_spatial_dims = num_dims - 2;

////////////////////////////////////////////////////////////
// Helper type aliases
////////////////////////////////////////////////////////////
template <typename DataType>
using Vector = ::distconv::Vector<DataType>;
using IntVector = ::distconv::IntVector;
using IndexVector = ::distconv::IndexVector;
using Shape = ::distconv::tensor::Shape;

template <typename DT>
using TensorHost = ::distconv::tensor::Tensor<
  DT, ::distconv::tensor::LocaleMPI,
  ::distconv::tensor::BaseAllocator>;

using TensorDev = ::distconv::tensor::Tensor<
  DataType, ::distconv::tensor::LocaleMPI,
  ::distconv::tensor::CUDAAllocator>;

template <typename DT>
using TensorHostShuffler = ::distconv::tensor::TensorMPIShuffler<
  DT, ::distconv::tensor::BaseAllocator>;

using TensorShuffler = ::distconv::tensor::TensorMPICUDAShuffler<DataType>;
using TensorShufflerP2P = ::distconv::tensor::TensorMPICUDAShufflerP2P<DataType>;
using TensorShufflerAL = ::distconv::tensor::TensorMPICUDAShufflerAL<DataType>;
using TensorShufflerHybrid = ::distconv::tensor::TensorMPICUDAShufflerHybrid<DataType>;

using Dist = ::distconv::tensor::Distribution;
static constexpr int num_dists = 4;

using LocaleMPI = ::distconv::tensor::LocaleMPI;

using MPIPrintStreamDebug = ::distconv::util::MPIPrintStreamDebug;
using MPIPrintStreamError = ::distconv::util::MPIPrintStreamError;
using MPIPrintStreamInfo = ::distconv::util::MPIPrintStreamInfo;
using MPIPrintStreamWarning = ::distconv::util::MPIPrintStreamWarning;
using MPIRootPrintStreamDebug = ::distconv::util::MPIRootPrintStreamDebug;
using MPIRootPrintStreamError = ::distconv::util::MPIRootPrintStreamError;
using MPIRootPrintStreamInfo = ::distconv::util::MPIRootPrintStreamInfo;
using MPIRootPrintStreamWaning = ::distconv::util::MPIRootPrintStreamWarning;

using Backend = ::distconv::cudnn::BackendCUDNN;
using ReLU = ::distconv::ReLU<Backend>;
using LeakyReLU = ::distconv::LeakyReLU<Backend>;
using Convolution = ::distconv::Convolution<Backend, num_dims, DataType>;
using Pooling = ::distconv::Pooling<Backend, num_dims, DataType>;
using BatchNormalization = ::distconv::BatchNormalization<Backend, num_dims, DataType>;

namespace tensor = ::distconv::tensor;
namespace util = ::distconv::util;

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

/** Return a cached Elemental communicator for the spatial domain
 */
std::shared_ptr<El::mpi::Comm> get_spatial_el_comm(const LocaleMPI &spatial_loc);

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

/** Query profiling
 */
bool is_profiling_enabled();

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

/** Get p2p handle
 */
p2p::P2P &get_p2p();

/** Get Aluminum MPI-CUDA backend
 */
Al::mpicuda_backend::comm_type &get_mpicuda();

/** Get Distconv backend handle.
 */
Backend &get_backend();
/** Get Distconv stream.
 */
cudaStream_t get_stream();
/** Return a HaloExchangeMethod
 */
::distconv::HaloExchangeMethod get_halo_exchange_method();

TensorShuffler *get_tensor_shuffler(const TensorDev &src,
                                    const TensorDev &dst);

MPI_Comm get_input_comm(const lbann_comm &comm);
/** Return the MPI rank when reading input dataset
 */
int get_input_rank(const lbann_comm &comm);

} // namespace dc
} // namespace lbann

#endif // LBANN_HAS_DISTCONV
#endif // LBANN_UTILS_DISTCONV_HPP
