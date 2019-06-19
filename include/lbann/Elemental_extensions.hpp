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
// http://github.com/LBANN.
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

#include "El.hpp"

namespace El {

template<typename F>
void ColumnSum(const Matrix<F>& X, Matrix<F>& sums);

template<typename F>
void ColumnSum(const AbstractMatrix<F>& X, AbstractMatrix<F>& sums);

template<typename F>
void ColumnSum(const AbstractDistMatrix<F>& A, AbstractDistMatrix<F>& sums);

template<typename F>
void RowSum(const Matrix<F>& X, Matrix<F>& sums);

template<typename F>
void RowSum(const AbstractMatrix<F>& X, AbstractMatrix<F>& sums);

template <typename F>
void RowSum(const AbstractDistMatrix<F>& A, AbstractDistMatrix<F>& sums);

#define LBANN_PROTO(F)                                           \
  template void ColumnSum                                        \
  ( const Matrix<F>& X, Matrix<F>& norms );                      \
  template void RowSum                                           \
  ( const Matrix<F>& X, Matrix<F>& norms );                      \
  template void ColumnSum                                        \
  ( const AbstractMatrix<F>& X, AbstractMatrix<F>& norms );      \
  template void RowSum                                           \
  ( const AbstractMatrix<F>& X, AbstractMatrix<F>& norms );      \
  template void ColumnSum                                        \
  (const AbstractDistMatrix<F>& X, AbstractDistMatrix<F>& sums); \
  template void RowSum                                           \
  (const AbstractDistMatrix<F>& X, AbstractDistMatrix<F>& sums); \

/* #define EL_NO_INT_PROTO */
/* #define EL_ENABLE_QUAD */
/* #include "El/macros/Instantiate.h" */

#ifndef LBANN_PROTO_REAL
# define LBANN_PROTO_REAL(T) LBANN_PROTO(T)
#endif

#ifndef LBANN_PROTO_FLOAT
# define LBANN_PROTO_FLOAT LBANN_PROTO_REAL(float)
#endif
#ifndef LBANN_PROTO_DOUBLE
# define LBANN_PROTO_DOUBLE LBANN_PROTO_REAL(double)
#endif

} // namespace El
