////////////////////////////////////////////////////////////////////////////////xecu
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

#ifndef LBANN_BASE_HPP_INCLUDED
#define LBANN_BASE_HPP_INCLUDED

#include <El.hpp>

// Defines, among other things, DataType.
#include "lbann_config.hpp"

#include "lbann/Elemental_extensions.hpp"
#include "lbann/utils/enum_iterator.hpp"
#include "lbann/utils/file_utils.hpp"

// Support for OpenMP macros
#include "lbann/utils/omp_pragma.hpp"

#include <functional>
#include <iostream>
#include <memory>
#include <string>

namespace lbann {

// Forward-declaration.
class lbann_comm;

/// Creating an observer_ptr to complement the unique_ptr and shared_ptr
template <typename T>
using observer_ptr = typename std::add_pointer<T>::type;

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
 *  @return     LBANN communicator corresponding to MPI_COMM_WORLD.
 */
world_comm_ptr initialize(int& argc, char**& argv);

/** @brief Initialize LBANN for use with external applcations
 *  @param argc Command line arguments.
 *  @param argv Number of command line arguments.
 *  @return LBANN communicator
 */
std::unique_ptr<lbann_comm> initialize_lbann(int argc, char** argv);

/** @brief Initialize LBANN for use with external applcations
 *  @param[in] c MPI communicator
 *  @return LBANN communicator using provided MPI comm
 */
std::unique_ptr<lbann_comm> initialize_lbann(MPI_Comm c);

/** @brief Initialize LBANN for use with external applcations
 *  @param[in] c Hydrogen MPI communicator
 *  @return LBANN communicator using provided Hydrogen comm
 */
std::unique_ptr<lbann_comm> initialize_lbann(El::mpi::Comm&& c);

/** @brief Destroy LBANN communicator for external application
 *  @param[in] comm LBANN communicator
 */
void finalize_lbann(lbann_comm* comm = nullptr);

/** Destroy LBANN communicator.
 *
 *  Finalizes Elemental, which in turn finalizes MPI, Aluminum, and
 *  CUDA.
 */
void finalize(lbann_comm* comm = nullptr);

#ifdef LBANN_HAS_HALF
using cpu_fp16 = El::cpu_half_type;
#endif

#ifdef LBANN_HAS_GPU_FP16
using fp16 = El::gpu_half_type;
#endif

// Typedefs for Elemental matrices
using AbsMat = El::AbstractMatrix<DataType>;
using CPUMat = El::Matrix<DataType, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
using GPUMat = El::Matrix<DataType, El::Device::GPU>;
#endif // LBANN_HAS_GPU
using AbsDistMat = El::AbstractDistMatrix<DataType>;
using BaseDistMat = El::BaseDistMatrix;

// Deprecated typedefs
/// @todo Remove
using EGrid = El::Grid;
using Grid = El::Grid;
template <El::Device D>
using DMat = El::Matrix<DataType, D>;
template <El::Device D>
using AbsDistMatReadProxy = El::AbstractDistMatrixReadDeviceProxy<DataType, D>;
using ElMat = El::ElementalMatrix<DataType>;
using BlockMat = El::BlockMatrix<DataType>;

template <typename TensorDataType>
using CPUMatDT = El::Matrix<TensorDataType, El::Device::CPU>;

template <typename TensorDataType, El::Device D>
using MCMRMatDT =
  El::DistMatrix<TensorDataType, El::MC, El::MR, El::ELEMENT, D>;
template <typename TensorDataType, El::Device D>
using CircMatDT =
  El::DistMatrix<TensorDataType, El::CIRC, El::CIRC, El::ELEMENT, D>;
template <typename TensorDataType, El::Device D>
using StarMatDT =
  El::DistMatrix<TensorDataType, El::STAR, El::STAR, El::ELEMENT, D>;
template <typename TensorDataType, El::Device D>
using StarVCMatDT =
  El::DistMatrix<TensorDataType, El::STAR, El::VC, El::ELEMENT, D>;
template <typename TensorDataType, El::Device D>
using VCStarMatDT = El::DistMatrix<TensorDataType,
                                   El::VC,
                                   El::STAR,
                                   El::ELEMENT,
                                   D>; /// ColSumStarVCMat
template <typename TensorDataType, El::Device D>
using MCStarMatDT = El::
  DistMatrix<TensorDataType, El::MC, El::STAR, El::ELEMENT, D>; /// RowSumMat
template <typename TensorDataType, El::Device D>
using MRStarMatDT = El::
  DistMatrix<TensorDataType, El::MR, El::STAR, El::ELEMENT, D>; /// ColSumMat
template <typename TensorDataType, El::Device D>
using StarMRMatDT =
  El::DistMatrix<TensorDataType, El::STAR, El::MR, El::ELEMENT, D>;
template <typename TensorDataType>
using DistMatDT = MCMRMatDT<TensorDataType, El::Device::CPU>;

template <El::Device D>
using MCMRMat = MCMRMatDT<DataType, D>;
template <El::Device D>
using CircMat = CircMatDT<DataType, D>;
template <El::Device D>
using StarMat = StarMatDT<DataType, D>;
template <El::Device D>
using StarVCMat = StarVCMatDT<DataType, D>;
template <El::Device D>
using VCStarMat = VCStarMatDT<DataType, D>; /// ColSumStarVCMat
template <El::Device D>
using MCStarMat = MCStarMatDT<DataType, D>; /// RowSumMat
template <El::Device D>
using MRStarMat = MRStarMatDT<DataType, D>; /// ColSumMat
template <El::Device D>
using StarMRMat = StarMRMatDT<DataType, D>;
using DistMat = MCMRMat<El::Device::CPU>;
using Mat =
  El::Matrix<DataType, El::Device::CPU>; // Temporarily define as CPUMat

// Datatype for model evaluation
// Examples: timing, metrics, objective functions
using EvalType = double;

/// Distributed matrix format
enum class matrix_format
{
  MC_MR,
  CIRC_CIRC,
  STAR_STAR,
  STAR_VC,
  MC_STAR,
  invalid
};

/// Backpropagation requirements from a layer or operator
enum BackpropRequirements
{
  NO_REQUIREMENTS = 0,
  ERROR_SIGNALS = 1,    // Error signals from child layers
  PREV_ACTIVATIONS = 2, // Input activations from forward pass
  ACTIVATIONS = 4,      // Output activations from forward pass
  WEIGHTS = 8,          // Weights
};

/// @todo This should move to hydrogen
std::string to_string(El::Device const& d);
El::Device device_from_string(std::string const& str);

/// Data layout that is optimized for different modes of parallelism
enum class data_layout
{
  MODEL_PARALLEL,
  DATA_PARALLEL,
  invalid
};
matrix_format data_layout_to_matrix_format(data_layout layout);
std::string to_string(data_layout const& dl);
data_layout data_layout_from_string(std::string const& str);

/// Neural network execution mode
enum class execution_mode
{
  training,
  validation,
  testing,
  prediction,
  tournament,
  inference,
  invalid
};
std::string to_string(execution_mode m);
using execution_mode_iterator = enum_iterator<execution_mode,
                                              execution_mode::training,
                                              execution_mode::invalid>;

/** @brief Convert a string to an execution_mode. */
execution_mode exec_mode_from_string(std::string const& str);

/*
 * endsWith:
 * http://thispointer.com/c-how-to-check-if-a-string-ends-with-an-another-given-string/
 * Case Sensitive Implementation of endsWith()
 * It checks if the string 'mainStr' ends with given string
 * 'toMatch'
 */
bool endsWith(const std::string mainStr, const std::string& toMatch);

/// Print the dimensions and name of a Elemental matrix
void print_matrix_dims(AbsDistMat* m, const char* name);
#define LBANN_PRINT_MATRIX_DIMS(x) print_matrix_dims(x, #x);

/// Print the dimensions and name of a Elemental matrix
void print_local_matrix_dims(AbsMat* m, const char* name);
#define LBANN_PRINT_LOCAL_MATRIX_DIMS(x) print_local_matrix_dims(x, #x);

#define LBANN_MAKE_STR_(x) #x
#define LBANN_MAKE_STR(x) LBANN_MAKE_STR_(x)

void lbann_mpi_err_handler(MPI_Comm* comm, int* err_code, ...);

} // namespace lbann

/** @brief Extract an execution_mode from a stream. */
std::istream& operator>>(std::istream& os, lbann::execution_mode& e);

#endif // LBANN_BASE_HPP_INCLUDED
