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
#include "distconv/tensor/shuffle.hpp"
#include "distconv/tensor/shuffle_p2p.hpp"
#include "distconv/tensor/shuffle_al.hpp"
#include "distconv/tensor/shuffle_hybrid.hpp"
#include "distconv/tensor/algorithms.hpp"
#include "p2p/p2p.hpp"

namespace lbann {
namespace dc {

////////////////////////////////////////////////////////////
// Helper type aliases
////////////////////////////////////////////////////////////
using Array4 = ::distconv::tensor::Array<4>;

using TensorHost = ::distconv::tensor::Tensor<
  4, DataType, ::distconv::tensor::LocaleMPI,
  ::distconv::tensor::CUDAAllocator>;

using TensorDev = ::distconv::tensor::Tensor<
  4, DataType, ::distconv::tensor::LocaleMPI,
  ::distconv::tensor::CUDAAllocator>;

using TensorShuffler = ::distconv::tensor::TensorMPICUDAShuffler<
  4, DataType>;
using TensorShufflerP2P = ::distconv::tensor::TensorMPICUDAShufflerP2P<
  4, DataType>;
using TensorShufflerAL = ::distconv::tensor::TensorMPICUDAShufflerAL<
  4, DataType>;
using TensorShufflerHybrid = ::distconv::tensor::TensorMPICUDAShufflerHybrid<
  4, DataType>;

using Dist = ::distconv::tensor::Distribution<4>;

using LocaleMPI = ::distconv::tensor::LocaleMPI;

using MPIPrintStreamDebug = ::distconv::util::MPIPrintStreamDebug;
using MPIPrintStreamError = ::distconv::util::MPIPrintStreamError;
using MPIPrintStreamInfo = ::distconv::util::MPIPrintStreamInfo;
using MPIRootPrintStreamDebug = ::distconv::util::MPIRootPrintStreamDebug;
using MPIRootPrintStreamError = ::distconv::util::MPIRootPrintStreamError;
using MPIRootPrintStreamInfo = ::distconv::util::MPIRootPrintStreamInfo;

using Backend = ::distconv::cudnn::BackendCUDNN;
using ReLU = ::distconv::ReLU<Backend>;
using Convolution = ::distconv::Convolution<Backend, 4, DataType>;
using Pooling = ::distconv::Pooling<Backend, 4, DataType>;
using BatchNormalization = ::distconv::BatchNormalization<Backend, DataType>;

namespace tensor = ::distconv::tensor;
namespace util = ::distconv::util;

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

/** Query if partial aggregation of batch normalization statistics is
 * enabled
 */
bool use_partial_aggregation_in_bn();

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
} // namespace dc
} // namespace lbann

#endif // LBANN_HAS_DISTCONV
#endif // LBANN_UTILS_DISTCONV_HPP
