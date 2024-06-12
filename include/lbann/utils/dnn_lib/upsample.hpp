////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_UTILS_DNN_LIB_UPSAMPLE_HPP
#define LBANN_UTILS_DNN_LIB_UPSAMPLE_HPP

#include "lbann_config.hpp"

#if defined LBANN_HAS_CUDNN
#include "lbann/utils/dnn_lib/cudnn/upsample.hpp"
#elif defined LBANN_HAS_MIOPEN
#include "lbann/utils/dnn_lib/miopen/upsample.hpp"
#else
static_assert(false,
              "This file must be included only if there is support from a "
              "valid DNN library.");
#endif // LBANN_HAS_CUDNN

#endif // LBANN_UTILS_DNN_LIB_UPSAMPLE_HPP
