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

#include "datatype.hpp"
#include "El.hpp"

// typedef double DataType; // if you change this, also update DataTypeMPI
// static MPI_Datatype DataTypeMPI = MPI_DOUBLE;
typedef float DataType; // if you change this, also update DataTypeMPI
static MPI_Datatype DataTypeMPI = MPI_FLOAT;

typedef El::Grid EGrid;
typedef El::Grid Grid;
typedef El::Matrix<DataType> Mat;
typedef El::AbstractDistMatrix<DataType> AbsDistMat;
typedef El::DistMatrix<DataType> DistMat;
typedef El::DistMatrix<DataType, El::CIRC, El::CIRC> CircMat;
typedef El::DistMatrix<DataType, El::STAR, El::STAR> StarMat;
typedef El::DistMatrix<DataType, El::MR, El::STAR> ColSumMat; /* Summary matrix over columns */
typedef El::DistMatrix<DataType, El::STAR, El::VC> StarVCMat;
typedef El::BlockMatrix<DataType> BlockMat;
typedef El::ElementalMatrix<DataType> ElMat;

/// Distributed matrix format
enum class matrix_format {MC_MR, CIRC_CIRC, STAR_STAR, STAR_VC};

/// Neural network execution mode
enum class execution_mode {training, validation, testing, prediction, invalid};
static const char* __attribute__((used)) _to_string(execution_mode m) { 
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
    throw("Invalid execution mode specified");
  }
  return NULL;
}

/// Weight matrix initialization scheme
enum class weight_initialization {zero, uniform, normal, glorot_normal, glorot_uniform, he_normal, he_uniform};

/// Pooling layer mode
enum class pool_mode {max, average, average_no_pad};

namespace lbann
{
    class CUtility
    {
    public:
        static void convolveMat(StarMat* Kernels, BlockMat& InputMat, BlockMat& OutputMat,
                                uint InputWidth, uint InputHeight);
    };
    
    
}

/// Print the dimensions and name of a Elemental matrix
static const char* __attribute__((used)) _display_matrix(ElMat *m, const char *name) {
  std::cout << "DISPLAY MATRIX: " << name << " = " << m->Height() << " x " << m->Width() << std::endl;
}
#define DISPLAY_MATRIX(x) _display_matrix(x, #x);

#endif // LBANN_BASE_HPP
