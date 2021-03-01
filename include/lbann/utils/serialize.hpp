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
#pragma once
#ifndef LBANN_UTILS_SERIALIZE_HPP_
#define LBANN_UTILS_SERIALIZE_HPP_

#include <lbann_config.hpp>
#include "serialization/cereal_utils.hpp"

// Serialization code is only valid in C++ code.
#if !(defined __CUDACC__)
#include "serialization/rooted_archive_adaptor.hpp"
#ifdef LBANN_HAS_HALF
#include "serialization/serialize_half.hpp"
#endif // LBANN_HAS_HALF
#include "serialization/serialize_matrices.hpp"

#endif // !(defined __CUDACC__ || defined __HIPCC__)
#endif // LBANN_UTILS_SERIALIZE_HPP_
