////////////////////////////////////////////////////////////////////////////////xecu
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

#ifndef LBANN_BASE_HPP_INCLUDED
#define LBANN_BASE_HPP_INCLUDED

#include "El.hpp"
#include "lbann/Elemental_extensions.hpp"
#include "lbann/utils/cyg_profile.hpp"
#include "lbann/utils/file_utils.hpp"

// Defines, among other things, DataType.
#include "lbann_config.hpp"

// Support for OpenMP macros
#include "lbann/utils/omp_pragma.hpp"

#include <functional>
#include <iostream>
#include <memory>
#include <string>

namespace lbann {

// Forward-declaration.
class lbann_comm;

// Note that this should only be used to wrap the thing coming out of
// initialize()! This will be removed when we have proper RAII around
// these things.
using world_comm_ptr =
    std::unique_ptr<lbann_comm, std::function<void(lbann_comm*)>>;

/** Create LBANN communicator.
 *
 *  Initializes Elemental, which in turn initializes MPI, Aluminum,
 *  and CUDA. The LBANN communicator is initialized with one trainer
 *  (which can be changed by calling @c lbann_comm::split_trainers
 *  afterward).
 *
 *  @param argc Command line arguments.
 *  @param argv Number of command line arguments.
 *  @param seed RNG seed.
 *  @return     LBANN communicator corresponding to MPI_COMM_WORLD.
 */
world_comm_ptr initialize(int& argc, char**& argv, int seed = -1);

/** Destroy LBANN communicator.
 *
 *  Finalizes Elemental, which in turn finalizes MPI, Aluminum, and
 *  CUDA.
 */
void finalize(lbann_comm* comm = nullptr);

// Typedefs for Elemental matrices
using AbsMat = El::AbstractMatrix<DataType>;
using CPUMat = El::Matrix<DataType, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
using GPUMat = El::Matrix<DataType, El::Device::GPU>;
#endif // LBANN_HAS_GPU
using AbsDistMat = El::AbstractDistMatrix<DataType>;

// Deprecated typedefs
/// @todo Remove
using EGrid      = El::Grid;
using Grid       = El::Grid;
template <El::Device D>
using DMat       = El::Matrix<DataType, D>;
template <El::Device D>
using AbsDistMatReadProxy = El::AbstractDistMatrixReadDeviceProxy<DataType, D>;
using ElMat      = El::ElementalMatrix<DataType>;
using BlockMat   = El::BlockMatrix<DataType>;
template <El::Device D>
using MCMRMat    = El::DistMatrix<DataType, El::MC  , El::MR  , El::ELEMENT, D>;
template <El::Device D>
using CircMat    = El::DistMatrix<DataType, El::CIRC, El::CIRC, El::ELEMENT, D>;
template <El::Device D>
using StarMat    = El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, D>;
template <El::Device D>
using StarVCMat  = El::DistMatrix<DataType, El::STAR, El::VC  , El::ELEMENT, D>;
template <El::Device D>
using VCStarMat  = El::DistMatrix<DataType, El::VC  , El::STAR, El::ELEMENT, D>; /// ColSumStarVCMat
template <El::Device D>
using MCStarMat  = El::DistMatrix<DataType, El::MC  , El::STAR, El::ELEMENT, D>; /// RowSumMat
template <El::Device D>
using MRStarMat  = El::DistMatrix<DataType, El::MR  , El::STAR, El::ELEMENT, D>; /// ColSumMat
template <El::Device D>
using StarMRMat  = El::DistMatrix<DataType, El::STAR, El::MR  , El::ELEMENT, D>;
using DistMat = MCMRMat<El::Device::CPU>;
using Mat = El::Matrix<DataType, El::Device::CPU>; // Temporarily define as CPUMat

// Datatype for model evaluation
// Examples: timing, metrics, objective functions
using EvalType = double;

/// Distributed matrix format
enum class matrix_format {MC_MR, CIRC_CIRC, STAR_STAR, STAR_VC, MC_STAR, invalid};

/// Data layout that is optimized for different modes of parallelism
enum class data_layout {MODEL_PARALLEL, DATA_PARALLEL, invalid};
matrix_format data_layout_to_matrix_format(data_layout layout);

/// Neural network execution mode
enum class execution_mode {training, validation, testing, prediction, invalid};
std::string to_string(execution_mode m);

/** @brief Convert a string to an execution_mode. */
execution_mode exe_mode_from_string(std::string const& str);
/** @brief Extract an execution_mode from a stream. */
std::istream& operator>>(std::istream& os, execution_mode& e);

/** Pooling layer mode */
enum class pool_mode {invalid, max, average, average_no_pad};

/** returns a string representation of the pool_mode */
std::string get_pool_mode_name(pool_mode m);

// NA - Not applicable, used for input layers that don't produce a second output
enum class data_reader_target_mode {CLASSIFICATION, REGRESSION, RECONSTRUCTION, NA};

/*
 * endsWith: http://thispointer.com/c-how-to-check-if-a-string-ends-with-an-another-given-string/
 * Case Sensitive Implementation of endsWith()
 * It checks if the string 'mainStr' ends with given string
 * 'toMatch'
 */
bool endsWith(const std::string mainStr, const std::string &toMatch);

/// Print the dimensions and name of a Elemental matrix
void print_matrix_dims(AbsDistMat *m, const char *name);
#define LBANN_PRINT_MATRIX_DIMS(x) print_matrix_dims(x, #x);

/// Print the dimensions and name of a Elemental matrix
void print_local_matrix_dims(AbsMat *m, const char *name);
#define LBANN_PRINT_LOCAL_MATRIX_DIMS(x) print_local_matrix_dims(x, #x);

#define LBANN_MAKE_STR_(x) #x
#define LBANN_MAKE_STR(x) LBANN_MAKE_STR_(x)

} // namespace lbann

#endif // LBANN_BASE_HPP_INCLUDED
