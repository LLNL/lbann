////////////////////////////////////////////////////////////////////////////////
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
// This file is an extension of several Elemental functions
////////////////////////////////////////////////////////////////////////////////

#include "El.hpp"

#include "lbann/lbann_Elemental_extensions.h"

namespace El {

template<typename F>
void ColumnSum( const Matrix<F>& X, Matrix<Base<F>>& sums )
{
  //    DEBUG_ONLY(CSE cse("ColumnSum"))
  //typedef Base<F> Real;
    const Int m = X.Height();
    const Int n = X.Width();
    const F* XBuf = X.LockedBuffer();
    const Int XLDim = X.LDim();

    sums.Resize( n, 1 );
    Base<F>* sumsBuf = sums.Buffer();
    for( Int j=0; j<n; ++j )
    {
      for( Int i=0; i<m; ++i )
        sumsBuf[j] += XBuf[i+j*XLDim];
    }
}

// template<typename F,Dist U,Dist V>
// void ColumnSum
// ( const DistMatrix<F,U,V>& A, DistMatrix<Base<F>,V,STAR>& sums )
// {
//     DEBUG_ONLY(CSE cse("ColumnSum"))
//     const Int n = A.Width();
//     const Int mLocal = A.LocalHeight();
//     const Int nLocal = A.LocalWidth();
//     const F* ABuf = A.LockedBuffer();
//     const Int ALDim = A.LDim();
//     sums.AlignWith( A );

//     // TODO: Switch to more stable parallel norm computation using scaling
//     sums.Resize( n, 1 );
//     Base<F>* sumsBuf = sums.Buffer();
//     for( Int jLoc=0; jLoc<nLocal; ++jLoc )
//     {
//         Base<F> localNorm = blas::Nrm2(mLocal,&ABuf[jLoc*ALDim],1);
//         normBuf[jLoc] = localNorm*localNorm;
//     }

//     mpi::AllReduce( normBuf, nLocal, mpi::SUM, A.ColComm() );
//     for( Int jLoc=0; jLoc<nLocal; ++jLoc )
//         normBuf[jLoc] = Sqrt(normBuf[jLoc]);
// }

template<typename F,Dist U,Dist V>
void ColumnSum
( const DistMatrix<F,U,V>& A, DistMatrix<Base<F>,V,STAR>& sums )
{
  //    DEBUG_ONLY(CSE cse("ColumnSum"))
    const Int n = A.Width();
    sums.AlignWith( A );
    sums.Resize( n, 1 );
    ColumnSum( A.LockedMatrix(), sums.Matrix() );
    AllReduce( sums.Matrix(), A.ColComm(), mpi::SUM );
}

template<typename F>
void ColumnMax( const Matrix<F>& X, Matrix<Base<F>>& norms )
{
  //    DEBUG_ONLY(CSE cse("ColumnMax"))
    typedef Base<F> Real;
    const Int m = X.Height();
    const Int n = X.Width();
    const F* XBuf = X.LockedBuffer();
    const Int XLDim = X.LDim();

    norms.Resize( n, 1 );
    Base<F>* normBuf = norms.Buffer();
    for( Int j=0; j<n; ++j )
    {
        Real colMax = 0;
        for( Int i=0; i<m; ++i )
            colMax = Max(colMax,XBuf[i+j*XLDim]);
        normBuf[j] = colMax;
    }
}

template<typename F,Dist U,Dist V>
void ColumnMax
( const DistMatrix<F,U,V>& A, DistMatrix<Base<F>,V,STAR>& norms )
{
  //    DEBUG_ONLY(CSE cse("ColumnMax"))
    const Int n = A.Width();
    norms.AlignWith( A );
    norms.Resize( n, 1 );
    ColumnMax( A.LockedMatrix(), norms.Matrix() );
    AllReduce( norms.Matrix(), A.ColComm(), mpi::MAX );
}

template<typename F>
void ColumnMax( const DistMultiVec<F>& X, Matrix<Base<F>>& norms )
{
  //    DEBUG_ONLY(CSE cse("ColumnMax"))
    ColumnMax( X.LockedMatrix(), norms );
    AllReduce( norms, X.Comm(), mpi::MAX );
}

template<typename F>
void ColumnMax( const SparseMatrix<F>& A, Matrix<Base<F>>& norms )
{
  //    DEBUG_ONLY(CSE cse("ColumnMax"))
    // Explicitly forming the transpose is overkill...
    // The following would be correct but is best avoided.
    /*
    SparseMatrix<F> ATrans;
    Transpose( A, ATrans );
    RowMaxNorms( ATrans, norms );
    */

    // Form the maxima
    // ---------------
    typedef Base<F> Real;
    Zeros( norms, A.Width(), 1 );

    const Int numEntries = A.NumEntries();
    const Int* colBuf = A.LockedTargetBuffer();
    const F* values = A.LockedValueBuffer();
    Real* normBuf = norms.Buffer(); 
    for( Int e=0; e<numEntries; ++e )
        normBuf[colBuf[e]] = Max(normBuf[colBuf[e]],values[e]);
}

template<typename F>
void ColumnMax
( const DistSparseMatrix<F>& A, DistMultiVec<Base<F>>& norms )
{
  //DEBUG_ONLY(CSE cse("ColumnMax"))
    typedef Base<F> Real;
    // Explicitly forming the transpose is overkill...
    // The following would be correct but is best avoided.
    /*
    DistSparseMatrix<F> ATrans(A.Comm());
    Transpose( A, ATrans );
    RowMaxNorms( ATrans, norms );
    */

    // Modify the communication pattern from an adjoint Multiply
    // =========================================================
    Zeros( norms, A.Width(), 1 );
    // TODO: (Moon 9/12/16) I replaced the commented code since
    // Elemental no longer has a DistSparseMultMeta class. I do not
    // know if the code is correct.
    // A.InitializeMultMeta();
    // const auto& meta = A.multMeta;
    const auto& meta = A.InitializeMultMeta();

    // Pack the send values 
    // --------------------
    vector<Real> sendVals( meta.numRecvInds, 0 );
    const Int numEntries = A.NumLocalEntries();
    const F* values = A.LockedValueBuffer();
    for( Int e=0; e<numEntries; ++e )
        sendVals[meta.colOffs[e]] = 
          Max(sendVals[meta.colOffs[e]],values[e]);

    // Inject the updates into the network
    // -----------------------------------
    const Int numRecvInds = meta.sendInds.size();
    vector<Real> recvVals( numRecvInds );
    mpi::AllToAll
    ( sendVals.data(), meta.recvSizes.data(), meta.recvOffs.data(),
      recvVals.data(), meta.sendSizes.data(), meta.sendOffs.data(),
      A.Comm() );

    // Form the maxima over all the values received
    // --------------------------------------------
    const Int firstLocalRow = norms.FirstLocalRow();
    Real* normBuf = norms.Matrix().Buffer();
    for( Int s=0; s<numRecvInds; ++s )
    {
        const Int i = meta.sendInds[s];
        const Int iLoc = i - firstLocalRow;
        normBuf[iLoc] = Max(normBuf[iLoc],recvVals[s]);
    }
}

LBANN_PROTO_FLOAT
LBANN_PROTO_DOUBLE

} // namespace El
