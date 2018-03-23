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
using Convolution = ::distconv::Convolution<Backend>;
using Pooling = ::distconv::Pooling<Backend>;
template <typename DataType>
using BatchNormalization = ::distconv::BatchNormalization<Backend, DataType>;

namespace tensor = ::distconv::tensor;
namespace util = ::distconv::util;

/** Get Distconv backend handle.
 */
Backend &get_backend();

} // namespace dc
} // namespace lbann

#endif // LBANN_HAS_DISTCONV
#endif // LBANN_UTILS_DISTCONV_HPP
