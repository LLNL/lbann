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
// lbann_layer_pooling .hpp .cpp - Pooling layer
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/lbann_layer_pooling.hpp"
#include "lbann/models/lbann_model_sequential.hpp"

using namespace std;
using namespace El;


lbann::PoolingLayer::PoolingLayer(const uint index,
                                  const int _PoolDim,
                                  const int _Channels,
                                  const int* _InputDims,
                                  const int* _PoolWindowDim,
                                  const int* _PoolPad,
                                  const int* _PoolStride,
                                  const int _PoolMode,
                                  const uint miniBatchSize,
                                  lbann_comm* comm,
                                  Optimizer* optimizer)
  : Layer(index, comm, optimizer, miniBatchSize),
    PoolDim(_PoolDim), Channels(_Channels), PoolMode(_PoolMode)
{
    for(int d=0; d<PoolDim; d++) {
      InputDims[d] = _InputDims[d];
      PoolWindowDim[d] = _PoolWindowDim[d];
      PoolPad[d] = _PoolPad[d];
      PoolStride[d] = _PoolStride[d];
      OutputDims[d] = (InputDims[d]+2*PoolPad[d])/PoolStride[d];
    }
    for(int d=PoolDim; d<3; d++) {
      InputDims[d] = 1;
      PoolWindowDim[d] = 1;
      PoolPad[d] = 1;
      PoolStride[d] = 1;
      OutputDims[d] = 1;
    }

    // Matrices are in Star,VC format, not default MC,MR
    delete Zs;
    delete Ds;
    delete Ds_Temp;
    delete Acts;
    Zs = new StarVCMat(comm->get_model_grid());
    Ds = new StarVCMat(comm->get_model_grid());
    Ds_Temp = new StarVCMat(comm->get_model_grid());
    Acts = new StarVCMat(comm->get_model_grid());
    

    // TODO: obtain dimensions from cudnnNet
    NumNeurons = Channels*OutputDims[0]*OutputDims[1]*OutputDims[2];

#ifdef __LIB_CUDNN
    cudnnLayer = NULL;
#endif
}

lbann::PoolingLayer::~PoolingLayer()
{
#ifdef __LIB_CUDNN
    if (cudnnLayer) delete cudnnLayer;
#endif
}

void lbann::PoolingLayer::setup(CudnnNet<DataType> *cudnnNet)
{
#ifdef __LIB_CUDNN
    // setup cudnn-based pooling layer instance
    int cudnnSrcTensorDims[] = {1, Channels,
                                InputDims[0], InputDims[1], InputDims[2]};
    int cudnnDstTensorDims[] = {1, Channels,
                                OutputDims[0], OutputDims[1], OutputDims[2]};
    if (cudnnLayer) delete cudnnLayer;
    cudnnLayer = new CudnnLayer<DataType>(cudnnNet);
    cudnnLayer->setupPoolLayer(PoolDim, cudnnSrcTensorDims, PoolMode,
                               PoolWindowDim, PoolPad, PoolStride,
                               cudnnDstTensorDims);
    //printf("pooling layer: outputsize: %d\n", cudnnLayer->dstDataSize);

    // create matrix for output and input deltas
    Ones(*Zs, cudnnLayer->dstDataSize+1, m_mini_batch_size);
    Zeros(*Ds, cudnnLayer->srcDataSize+1, m_mini_batch_size);
    Zeros(*Ds_Temp, cudnnLayer->srcDataSize+1, m_mini_batch_size);
    Ones(*Acts, cudnnLayer->dstDataSize+1, m_mini_batch_size);

#endif
}

void lbann::PoolingLayer::fp_linearity(ElMat& _WB, ElMat& _X, ElMat& _Z, ElMat& _Y) {
  // Convert matrices to desired formats
  DistMatrixReadProxy<DataType,DataType,STAR,VC> XProxy(_X);
  DistMatrixWriteProxy<DataType,DataType,STAR,VC> ZProxy(_Z);
  DistMatrixWriteProxy<DataType,DataType,STAR,VC> YProxy(_Y);
  StarVCMat& X = XProxy.Get();
  StarVCMat& Z = ZProxy.Get();
  StarVCMat& Y = YProxy.Get();

  // Get local matrices
  Mat& XLocal = X.Matrix();
  Mat& ZLocal = Z.Matrix();
  Mat& YLocal = Y.Matrix();
 
  for (int j = 0; j < XLocal.Width(); j++) {
    DataType* src = XLocal.Buffer(0, j);
    DataType* dst = ZLocal.Buffer(0, j);

#ifdef __LIB_CUDNN
    cudnnLayer->poolForwardHost(src, dst);
#endif
  }

  // Z and Y are identical after fp linearity step
  Copy(ZLocal, YLocal);

}

void lbann::PoolingLayer::bp_linearity() {
  // Convert matrices to desired formats
  DistMatrixReadProxy<DataType,DataType,STAR,VC> OutputDeltaProxy(*bp_input);
  DistMatrixReadProxy<DataType,DataType,STAR,VC> InputProxy(*fp_input); // TODO: store from fp step
  StarVCMat& OutputDelta = OutputDeltaProxy.Get();
  StarVCMat& Input = InputProxy.Get();

  // Get local matrices
  Mat& InputLocal = Input.Matrix();
  Mat& OutputLocal = Acts->Matrix();
  Mat& InputDeltaLocal = Ds_Temp->Matrix();
  Mat& OutputDeltaLocal = OutputDelta.Matrix();

  // Iterate through samples in minibatch
  for (int j = 0; j < InputLocal.Width(); j++) {
    DataType* src = InputLocal.Buffer(0, j);
    DataType* dst = OutputLocal.Buffer(0, j);
    DataType* srcD = InputDeltaLocal.Buffer(0, j);
    DataType* dstD = OutputDeltaLocal.Buffer(0, j);

#ifdef __LIB_CUDNN
    cudnnLayer->poolBackwardHost(src, dst, dstD, srcD);
#endif
  }

}

bool lbann::PoolingLayer::update()
{
    // nothing to update in pooling layer
    return true;
}
