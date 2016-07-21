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
// lbann_layer_convolutional .hpp .cpp - Convolutional Layer
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/lbann_layer_convolutional.hpp"

using namespace std;
using namespace El;

lbann::ConvolutionalLayer::ConvolutionalLayer(const uint index,
                                              const int  _ConvDim,
                                              const int  _InputChannels,
                                              const int* _InputDims,
                                              const int  _OutputChannels,
                                              const int* _FilterDims,
                                              const uint miniBatchSize,
                                              lbann_comm* comm,
                                              Optimizer* optimizer)
  : Layer(index, comm, optimizer, miniBatchSize),
    ConvDim(_ConvDim), InputChannels(_InputChannels),
    OutputChannels(_OutputChannels)
{
    // Initialize input, output, and filter dimensions
    for(int d=0; d<ConvDim; d++) {
      InputDims[d] = _InputDims[d];
      FilterDims[d] = _FilterDims[d];
      OutputDims[d] = InputDims[d] - FilterDims[d] + 1;
    }
    for(int d=ConvDim; d<3; d++) {
      InputDims[d] = 1;
      FilterDims[d] = 1;
      OutputDims[d] = 1;
    }

    // Matrices should be in Star,Star and Star,VC distributions
    delete WB;
    delete WB_D;
    delete Zs;
    delete Ds;
    delete Ds_Temp;
    delete Acts;
    WB = new StarMat(comm->get_model_grid());
    WB_D = new StarMat(comm->get_model_grid());
    Zs = new StarVCMat(comm->get_model_grid());
    Ds = new StarVCMat(comm->get_model_grid());
    Ds_Temp = new StarVCMat(comm->get_model_grid());
    Acts = new StarVCMat(comm->get_model_grid());

    // TODO: obtain dimensions from cudnnNet
    NumNeurons = OutputChannels*OutputDims[0]*OutputDims[1]*OutputDims[2];
    std::cout << "conv layer neurons = "<< NumNeurons << std::endl;

#ifdef __LIB_CUDNN
    cudnnLayer = NULL;
#endif
}

lbann::ConvolutionalLayer::~ConvolutionalLayer()
{
#ifdef __LIB_CUDNN
    if (cudnnLayer) delete cudnnLayer;
#endif
}

void lbann::ConvolutionalLayer::setup(CudnnNet<DataType> *cudnnNet)
{
#ifdef __LIB_CUDNN

    // setup cudnn-based convolutional layer instance
    int cudnnSrcTensorDims[5] = {1, InputChannels,
                                 InputDims[0], InputDims[1], InputDims[2]};
    int cudnnFilterDims[5] = {OutputChannels, InputChannels,
                              FilterDims[0], FilterDims[1], FilterDims[2]};
    int cudnnDstTensorDims[5] = {1, OutputChannels,
                                 OutputDims[0], OutputDims[1], OutputDims[2]};
    int convPads[] = {0,0,0};
    int convStrides[] = {1,1,1};
    if (cudnnLayer) delete cudnnLayer;
    cudnnLayer = new CudnnLayer<DataType>(cudnnNet);
    cudnnLayer->setupConvLayer(ConvDim,
                               cudnnSrcTensorDims,
                               cudnnFilterDims,
                               cudnnDstTensorDims,
                               convPads,
                               convStrides);
    //printf("convolutional layer: outputsize: %d\n", cudnnLayer->dstDataSize);
    //printf("convolutional layer: outputdim: %d %d %d %d\n", OutputDim[0], OutputDim[1], OutputDim[2], OutputDim[3]);

    if(optimizer != NULL) {
      optimizer->setup(1, cudnnLayer->filterSize);
    }

    // Initialize matrices
    DataType var_scale = sqrt(3.0 / (cudnnLayer->srcDataSize));
    Gaussian(*WB, cudnnLayer->filterSize, 1, (DataType)0.0, var_scale);
    Zeros(*WB_D, cudnnLayer->filterSize, 1);
    Ones(*Zs, cudnnLayer->dstDataSize+1, m_mini_batch_size);
    Zeros(*Ds, cudnnLayer->srcDataSize+1, m_mini_batch_size);
    Zeros(*Ds_Temp, cudnnLayer->srcDataSize+1, m_mini_batch_size);
    Ones(*Acts, cudnnLayer->dstDataSize+1, m_mini_batch_size);

#endif
}

void lbann::ConvolutionalLayer::fp_linearity(ElMat& _WB, ElMat& _X, ElMat& _Z, ElMat& _Y) {
  
  // Convert matrices to desired formats
  DistMatrixReadProxy<DataType,DataType,STAR,STAR> WBProxy(_WB);
  DistMatrixReadProxy<DataType,DataType,STAR,VC> XProxy(_X);
  DistMatrixWriteProxy<DataType,DataType,STAR,VC> ZProxy(_Z);
  DistMatrixWriteProxy<DataType,DataType,STAR,VC> YProxy(_Y);
  StarMat& WB = WBProxy.Get();
  StarVCMat& X = XProxy.Get();
  StarVCMat& Z = ZProxy.Get();
  StarVCMat& Y = YProxy.Get();

  // Get local matrices
  Mat& XLocal = X.Matrix();
  Mat& ZLocal = Z.Matrix();
  Mat& YLocal = Y.Matrix();

#ifdef __LIB_CUDNN
  // Apply convolution
  cudnnLayer->convForward(XLocal.Width(),
                          XLocal.LockedBuffer(),
                          XLocal.Height(),
                          WB.LockedBuffer(),
                          ZLocal.Buffer(),
                          ZLocal.Height());
#endif

  // Z and Y are identical after fp linearity step
  Copy(ZLocal, YLocal);

}

void lbann::ConvolutionalLayer::bp_linearity() {

  // Convert matrices to desired formats
  DistMatrixReadProxy<DataType,DataType,STAR,VC> OutputDeltaProxy(*bp_input);
  DistMatrixReadProxy<DataType,DataType,STAR,VC> InputProxy(*fp_input); // TODO: store from fp step
  StarVCMat& OutputDelta = OutputDeltaProxy.Get();
  StarVCMat& Input = InputProxy.Get();

  // Get local matrices
  Mat& InputLocal = Input.Matrix();
  Mat& FilterLocal = WB->Matrix();
  Mat& InputDeltaLocal = Ds_Temp->Matrix();
  Mat& FilterDeltaLocal = WB_D->Matrix();
  Mat& OutputDeltaLocal = OutputDelta.Matrix();

#ifdef __LIB_CUDNN
  // Apply back prop
  cudnnLayer->convBackward(InputLocal.Width(),
                           InputLocal.LockedBuffer(),
                           InputLocal.Height(),
                           FilterLocal.LockedBuffer(),
                           OutputDeltaLocal.LockedBuffer(),
                           OutputDeltaLocal.Height(),
                           FilterDeltaLocal.Buffer(),
                           InputDeltaLocal.Buffer(),
                           InputDeltaLocal.Height());
#endif

  // Obtain filter delta with reduction and scaling
  AllReduce(*WB_D, mpi::COMM_WORLD);  
  *WB_D *= 1.0/get_effective_minibatch_size();
  
}

///
/// @todo Convolutional layer weight/bias update
///
bool lbann::ConvolutionalLayer::update()
{
  // add a new function to optimizer
  optimizer->update_weight_bias_matrix(*WB_D, *WB);
  return true;
}
