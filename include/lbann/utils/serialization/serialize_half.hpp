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
#ifndef LBANN_UTILS_SERIALIZATION_SERIALIZE_HALF_HPP_INCLUDED
#define LBANN_UTILS_SERIALIZATION_SERIALIZE_HALF_HPP_INCLUDED

/** @file
 *
 *  Serialization functions for arithmetic types. Specializations for
 *  Cereal's Binary, JSON, and XML archives are provided.
 */

#include "lbann_config.hpp"

// Half-precision support comes from here:
#include <El.hpp> // IWYU pragma: export

#include "cereal_utils.hpp"

/** @namespace cereal
 *
 *  Extensions to Cereal for extra arithmetic types used by LBANN.
 */
namespace cereal {
#ifdef LBANN_HAS_HALF
#ifdef LBANN_HAS_GPU_FP16

/** @name General templates */
///@{

/** @brief Save a GPU half-precision value. */
template <typename OutputArchiveT>
void save(OutputArchiveT& archive, __half const& value) {
  float x = value;
  archive(x);
}

/** @brief Load a GPU half-precision value. */
template <typename InputArchiveT>
void load(InputArchiveT& archive, __half& value) {
  float x = 0.f;
  archive(x);
  value = x;
}

///@}
/** @name Binary archives */
///@{

/** @brief Save this half-precision value in Binary */
void save(BinaryOutputArchive&, __half const&);

/** @brief Load this half-precision value from Binary */
void load(BinaryInputArchive&, __half&);

///@}
#endif // LBANN_HAS_GPU_FP16

/** @name XML archives */
///@{

// Remove the default definitions from Cereal
inline void save(XMLOutputArchive&, half_float::half const&) = delete;
inline void load(XMLInputArchive&, half_float::half&) = delete;

/** @brief Save this half-precision value as a float for XML */
float save_minimal(XMLOutputArchive const&,
                   half_float::half const&) noexcept;

/** @brief Load this half-precision value as a float from XML */
void load_minimal(
  XMLInputArchive const&, half_float::half&, float const&) noexcept;

///@}
#endif // LBANN_HAS_HALF
}// namespace cereal

#endif // LBANN_UTILS_SERIALIZATION_SERIALIZE_HALF_HPP_INCLUDED
