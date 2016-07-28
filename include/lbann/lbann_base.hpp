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

enum class matrix_format {MC_MR, CIRC_CIRC, STAR_STAR, STAR_VC};

enum class execution_mode {training, validation, testing, prediction, invalid};

namespace lbann
{
    class CUtility
    {
    public:
        static void convolveMat(StarMat* Kernels, BlockMat& InputMat, BlockMat& OutputMat,
                                uint InputWidth, uint InputHeight);
    };
    
    
}

#endif // LBANN_BASE_HPP
