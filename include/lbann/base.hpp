////////////////////////////////////////////////////////////////////////////////xecu
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
//
// lbann_base .hpp - Basic definitions, functions
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_BASE_HPP
#define LBANN_BASE_HPP

#include "El.hpp"
#include "lbann/Elemental_extensions.hpp"
#include "lbann/utils/cyg_profile.hpp"

// Defines, among other things, lbann::DataType.
#include "lbann_config.hpp"

// Typedefs for Elemental matrices
using EGrid      = El::Grid;
using Grid       = El::Grid;
using Mat        = El::Matrix<lbann::DataType>;
using AbsDistMat = El::AbstractDistMatrix<lbann::DataType>;
using ElMat      = El::ElementalMatrix<lbann::DataType>;
using BlockMat   = El::BlockMatrix<lbann::DataType>;
using MCMRMat    = El::DistMatrix<lbann::DataType, El::MC  , El::MR  >;
using CircMat    = El::DistMatrix<lbann::DataType, El::CIRC, El::CIRC>;
using StarMat    = El::DistMatrix<lbann::DataType, El::STAR, El::STAR>;
using StarVCMat  = El::DistMatrix<lbann::DataType, El::STAR, El::VC  >;
using VCStarMat  = El::DistMatrix<lbann::DataType, El::VC  , El::STAR>;
using MCStarMat  = El::DistMatrix<lbann::DataType, El::MC  , El::STAR>;
using MRStarMat  = El::DistMatrix<lbann::DataType, El::MR  , El::STAR>;
using StarMRMat  = El::DistMatrix<lbann::DataType, El::STAR, El::MR  >;

// Deprecated typedefs for Elemental matrices
using DistMat         = MCMRMat;
using RowSumMat       = MCStarMat;
using ColSumStarVCMat = VCStarMat;
using ColSumMat       = MRStarMat;

// Datatype for model evaluation
// Examples: timing, metrics, objective functions
using EvalType = double;

/// Distributed matrix format
enum class matrix_format {MC_MR, CIRC_CIRC, STAR_STAR, STAR_VC, MC_STAR, invalid};

/// Data layout that is optimized for different modes of parallelism
enum class data_layout {MODEL_PARALLEL, DATA_PARALLEL, invalid};
static matrix_format __attribute__((used)) data_layout_to_matrix_format(data_layout layout) {
  matrix_format format;
  switch(layout) {
  case data_layout::MODEL_PARALLEL:
    format = matrix_format::MC_MR;
    break;
  case data_layout::DATA_PARALLEL:
    /// Weights are stored in STAR_STAR and data in STAR_VC
    format = matrix_format::STAR_STAR;
    break;
  default:
    throw(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " Invalid data layout selected");
  }
  return format;
}

/// Neural network execution mode
enum class execution_mode {training, validation, testing, prediction, invalid};
static const char *__attribute__((used)) _to_string(execution_mode m) {
  switch(m) {
  case execution_mode::training:
    return "training";
  case execution_mode::validation:
    return "validation";
  case execution_mode::testing:
    return "testing";
  case execution_mode::prediction:
    return "prediction";
  case execution_mode::invalid:
    return "invalid";
  default:
    throw("Invalid execution mode specified"); /// @todo this should be an lbann_exception but then the class has to move to resolve dependencies
  }
}

/** Pooling layer mode */
enum class pool_mode {max, average, average_no_pad};

/** returns a string representation of the pool_mode */
std::string get_pool_mode_name(pool_mode m);

namespace lbann {

// Forward-declaration.
class lbann_comm;

/**
 * Initialize LBANN.
 * The comm instance this returns places every process in one model. This can be
 * changed with lbann_comm::split_models afterward.
 * @param argc The program's argc.
 * @param argv The program's argv.
 * @param seed Optional seed for random number generators.
 */
lbann_comm* initialize(int& argc, char**& argv, int seed = -1);
/**
 * Perform finalization.
 */
void finalize(lbann_comm* comm = nullptr);

class CUtility {
 public:
  static void convolveMat(StarMat *Kernels, BlockMat& InputMat, BlockMat& OutputMat,
                          uint InputWidth, uint InputHeight);
};

/*
 * endsWith: http://thispointer.com/c-how-to-check-if-a-string-ends-with-an-another-given-string/
 * Case Sensitive Implementation of endsWith()
 * It checks if the string 'mainStr' ends with given string
 * 'toMatch'
 */
static bool __attribute__((used)) endsWith(const std::string &mainStr, const std::string &toMatch)
{
  if(mainStr.size() >= toMatch.size() &&
     mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0)
    return true;
  else
    return false;
}

}  // namespace lbann

/// Print the dimensions and name of a Elemental matrix
static void __attribute__((used)) _display_matrix(ElMat *m, const char *name) {
  std::cout << "DISPLAY MATRIX: " << name << " = " << m->Height() << " x " << m->Width() << std::endl;
}
#define DISPLAY_MATRIX(x) _display_matrix(x, #x);

// FIXME
#if 1
// __FILE__
#define log_msg(...) {\
  char str[256];\
  sprintf(str, __VA_ARGS__);\
  std::cout << "[" << m_comm->get_model_rank() << "." << m_comm->get_rank_in_model() << "][" << __FUNCTION__ << "][Line " << __LINE__ << "]" << str << std::endl; \
  }
#define log_simple_msg(...) {\
  char str[256];\
  sprintf(str, __VA_ARGS__);\
  std::cout << "[" << __FUNCTION__ << "][Line " << __LINE__ << "]" << str << std::endl; \
  }
#else
#define log_msg(...)
#define log_simple_msg(...)
#endif

#define LBANN_MAKE_STR(x) _LBANN_MAKE_STR(x)
#define _LBANN_MAKE_STR(x) #x

#endif // LBANN_BASE_HPP
