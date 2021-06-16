////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_UTILS_TENSOR_HPP
#define LBANN_UTILS_TENSOR_HPP

#include "lbann/base.hpp"

namespace lbann {

/// @brief Function to efficiently select the best method for copying between
/// two distributed tensors. Enable selection between synchronous and
/// asynchronous copies based on tensor distribution and
/// pre-processing macros
template <typename TDT>
void do_tensor_copy(const BaseDistMat& src,
                    El::AbstractDistMatrix<TDT>& tgt);

/// @brief If distributed tensors have the same distribution setup the
/// target to use a view to the source tensor, otherwise copy the src
/// to target.
template <typename TDT>
void view_or_copy_tensor(const BaseDistMat& src,
                         El::AbstractDistMatrix<TDT>& tgt);
}

#endif // LBANN_UTILS_TENSOR_HPP
