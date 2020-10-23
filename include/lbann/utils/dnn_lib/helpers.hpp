////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_UTILS_DNN_LIB_HELPERS_HPP
#define LBANN_UTILS_DNN_LIB_HELPERS_HPP

#include "lbann_config.hpp"

#if defined LBANN_HAS_CUDNN || defined LBANN_HAS_MIOPEN
#define LBANN_HAS_DNN_LIB
#endif

// Import the GPU __device__ function library
#if defined LBANN_HAS_CUDNN
#include "cudnn.hpp"
namespace lbann {
namespace dnn_lib {
using namespace cuda;
}// namespace dnn_lib
}// namespace lbann

#elif defined LBANN_HAS_MIOPEN

// For now, this is placeholder.
#include "miopen.hpp"
namespace lbann {
namespace dnn_lib = ::lbann::miopen;
}// namespace lbann

#endif // LBANN_HAS_CUDnn

#if defined LBANN_HAS_DNN_LIB
#include "dnn_lib.hpp"
#endif // LBANN_HAS_DNN_LIB

#endif // LBANN_UTILS_DNN_LIB_HELPERS_HPP
