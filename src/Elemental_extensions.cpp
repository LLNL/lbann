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
//
// This file is an extension of several Elemental functions
////////////////////////////////////////////////////////////////////////////////

#include "El.hpp"

#include "lbann/Elemental_extensions.hpp"

namespace El {

template<typename F>
void ColumnSum( const Matrix<F>& X, Matrix<F>& sums ) {
//    DEBUG_ONLY(CSE cse("ColumnSum"))

    // Input matrix parameters
    const Int m = X.Height();
    const Int n = X.Width();
    const F *XBuf = X.LockedBuffer();
    const Int XLDim = X.LDim();

    // Initialize output
    Zeros( sums, 1, n );
    F *sumsBuf = sums.Buffer();
    const Int sumsLDim = sums.LDim();

    // Compute sum over each column
    EL_PARALLEL_FOR
    for( Int j=0; j<n; ++j )
    {
        for( Int i=0; i<m; ++i )
        {
            sumsBuf[j*sumsLDim] += XBuf[i+j*XLDim];
        }
    }

}

template<typename F>
void ColumnSum( const AbstractMatrix<F>& X, AbstractMatrix<F>& sums ) {
    if (X.GetDevice() != sums.GetDevice())
        LogicError("ColumnSum requires matching device types.");

    if ((X.GetDevice() == Device::CPU)) {
      ColumnSum(static_cast<const Matrix<F,Device::CPU>&>(X),
                static_cast<Matrix<F,Device::CPU>&>(sums));
#ifdef LBANN_HAS_GPU
    }else if ((X.GetDevice() == Device::GPU)) {
      LogicError("ColumnSum: Unsupported device type.");
      // ColumnSum(static_cast<const Matrix<F,Device::GPU>&>(X),
      //           static_cast<Matrix<F,Device::GPU>&>(sums));
#endif // LBANN_HAS_GPU
    }else {
      LogicError("ColumnSum: Unsupported device type.");
    }
}

template<typename F>
void ColumnSum
( const AbstractDistMatrix<F>& A, AbstractDistMatrix<F>& sums ) {
//    DEBUG_ONLY(CSE cse("ColumnSum"))

    // Check that distributed matrix formats are valid
    if( A.DistData().rowDist != sums.DistData().rowDist
        || sums.DistData().colDist != STAR
        || A.DistData().blockHeight != sums.DistData().blockHeight
        || A.DistData().blockWidth != sums.DistData().blockWidth)
    {
        LogicError("Matrices do not have compatible data distributions");
    }

    // Compute column-wise sums
    sums.AlignWith( A );
    sums.Resize( 1, A.Width() );
    ColumnSum( A.LockedMatrix(), sums.Matrix() );
    AllReduce( sums.Matrix(), sums.RedundantComm(), mpi::SUM );

}

template<typename F>
void RowSum(const Matrix<F>& X, Matrix<F>& sums) {

    // Input matrix parameters
    const Int m = X.Height();
    const Int n = X.Width();
    const F *XBuf = X.LockedBuffer();
    const Int XLDim = X.LDim();

    // Initialize output
    Zeros( sums, m, 1 );
    F *sumsBuf = sums.Buffer();

    // Iterate through row blocks
    const Int bsize = Max( 64 / sizeof(F), 1 );
    EL_PARALLEL_FOR
    for( Int i=0; i<m; i+=bsize )
    {
        const Int mb = Min( bsize, m - i );
        for( Int j=0; j<n; ++j )
        {
            for( Int ib=0; ib<mb; ++ib )
            {
                sumsBuf[i+ib] += XBuf[(i+ib)+j*XLDim];
            }
        }
    }

}

template<typename F>
void RowSum(const AbstractMatrix<F>& X, AbstractMatrix<F>& sums) {
    if (X.GetDevice() != sums.GetDevice())
        LogicError("RowSum requires matching device types.");

    if ((X.GetDevice() == Device::CPU)) {
      RowSum(static_cast<const Matrix<F,Device::CPU>&>(X),
             static_cast<Matrix<F,Device::CPU>&>(sums));
#ifdef LBANN_HAS_GPU
    }else if ((X.GetDevice() == Device::GPU)) {
      LogicError("RowSum: Unsupported device type.");
      // RowSum(static_cast<const Matrix<F,Device::GPU>&>(X),
      //        static_cast<Matrix<F,Device::GPU>&>(sums));
#endif // LBANN_HAS_GPU
    }else {
      LogicError("RowSum: Unsupported device type.");
    }
}

template <typename F>
void RowSum(const AbstractDistMatrix<F>& A, AbstractDistMatrix<F>& sums) {

  // Check that distributed matrix formats are valid
  if( A.DistData().colDist != sums.DistData().colDist
      || sums.DistData().rowDist != STAR
      || A.DistData().blockHeight != sums.DistData().blockHeight
      || A.DistData().blockWidth != sums.DistData().blockWidth)
  {
      LogicError("Matrices do not have compatible data distributions");
  }

  // Compute row-wise sums
  sums.AlignWith( A );
  sums.Resize( A.Height(), 1 );
  RowSum( A.LockedMatrix(), sums.Matrix() );
  AllReduce( sums.Matrix(), sums.RedundantComm(), mpi::SUM );

}

LBANN_PROTO_FLOAT
LBANN_PROTO_DOUBLE

} // namespace El
