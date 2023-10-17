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
#ifndef OPTIMIZERS_UNIT_TEST_OPTIMIZER_COMMON_HPP_
#define OPTIMIZERS_UNIT_TEST_OPTIMIZER_COMMON_HPP_

// Some common includes
#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>
#include <lbann/base.hpp>
#include <lbann/utils/serialize.hpp>
#include <lbann_config.hpp>

// This header should only be used in the unit testing code, so this
// is fine.
using namespace h2::meta;

// Get the description out as a string. Useful for comparing objects
// that might not expose accessor functions for all metadata.
template <typename ObjectType>
std::string desc_string(ObjectType const& opt)
{
  std::ostringstream desc;
  desc << opt.get_description();
  return desc.str();
}

// Simple groups of Output/Input archive types. The car is the output
// archive, the cadr is the input archive. Accessor metafunctions are
// defined below.

using FpTypes = TL<float
#ifdef LBANN_HAS_DOUBLE
                   ,
                   double
#endif // LBANN_HAS_DOUBLE
#ifdef LBANN_HAS_HALF
                   ,
                   lbann::cpu_fp16
#ifdef LBANN_HAS_GPU_FP16
                   ,
                   lbann::fp16
#endif // LBANN_HAS_GPU_FP16
#endif // LBANN_HAS_HALF
                   >;

#ifdef LBANN_HAS_CEREAL_BINARY_ARCHIVES
template <typename T>
using BinaryArchiveTypeBundle =
  TL<T, cereal::BinaryOutputArchive, cereal::BinaryInputArchive>;
using BinaryArchiveTypes = tlist::ExpandTL<BinaryArchiveTypeBundle, FpTypes>;
#else
using BinaryArchiveTypes = tlist::Empty;
#endif // LBANN_HAS_CEREAL_BINARY_ARCHIVES

#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
template <typename T>
using XMLArchiveTypeBundle =
  TL<T, cereal::XMLOutputArchive, cereal::XMLInputArchive>;
using XMLArchiveTypes = tlist::ExpandTL<XMLArchiveTypeBundle, FpTypes>;
#else
using XMLArchiveTypes = tlist::Empty;
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES

using AllArchiveTypes = tlist::Append<BinaryArchiveTypes, XMLArchiveTypes>;

#endif // OPTIMIZERS_UNIT_TEST_OPTIMIZER_COMMON_HPP_
