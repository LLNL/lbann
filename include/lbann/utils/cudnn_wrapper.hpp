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
// cudnn_wrapper.hpp - CUDNN support - wrapper classes, utility functions
////////////////////////////////////////////////////////////////////////////////

#ifndef CUDNN_WRAPPER_HPP_INCLUDED
#define CUDNN_WRAPPER_HPP_INCLUDED

#ifdef __LIB_CUDNN

// General
#include <iostream>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

// CUDNN
#include "cudnn.h"

// Error utility
#define checkCUDA(status) {                                             \
    if (status != cudaSuccess) {                                        \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << "\n"; \
      cudaDeviceReset();                                                \
      exit(-1);                                                         \
    }                                                                   \
  }
#define checkCUDNN(status) {                                            \
    if (status != CUDNN_STATUS_SUCCESS) {                               \
      std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << "\n"; \
      cudaDeviceReset();                                                \
      exit(-1);                                                         \
    }                                                                   \
  }

template <class DataType>
class CudnnNet {

public:
  /// Number of GPUs
  const int numGPUs;
  /// List of cuDNN handles
  std::vector<cudnnHandle_t> cudnnHandles;
  /// cuDNN datatype
  const cudnnDataType_t cudnnDataType;

private:
  static int getNumGPUs() {
    int count;
    checkCUDA(cudaGetDeviceCount(&count));
    return count;
  }

public:
  /// Constructor
  CudnnNet()
    : numGPUs(getNumGPUs()),
      cudnnDataType((sizeof(DataType)==sizeof(float)) ?
                    CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE) {
    cudnnHandles.resize(numGPUs, NULL);
    for(int i=0; i<numGPUs; ++i) {
      checkCUDA(cudaSetDevice(i));
      checkCUDNN(cudnnCreate(&cudnnHandles[i]));
    }
  }

  /// Destructor
  ~CudnnNet() {
    for(int i=0; i<cudnnHandles.size(); ++i) {
      if(cudnnHandles[i])
        checkCUDNN(cudnnDestroy(cudnnHandles[i]));
    }
  }

  /// Print cuDNN version number
  void printVersion() const {
    int version = (int)cudnnGetVersion();
    std::cout << "cudnnGetVersion() :" << version << ", "
              << "CUDNN_VERSION from cudnn.h : " << CUDNN_VERSION << std::endl;
  }

};


template <class DataType>
class CudnnLayer {

public:
  CudnnNet<DataType>* dnnNet;

  /// Input tensor descriptor
  cudnnTensorDescriptor_t srcTensorDesc;
  /// Output tensor descriptor
  cudnnTensorDescriptor_t dstTensorDesc;
  /// Bias tensor descriptor
  cudnnTensorDescriptor_t biasTensorDesc;
  /// Filter descriptor
  cudnnFilterDescriptor_t filterDesc;
  /// Convolution descriptor
  cudnnConvolutionDescriptor_t convDesc;
  /// Pooling descriptor
  cudnnPoolingDescriptor_t poolDesc;

  /// Forward pass algorithm
  cudnnConvolutionFwdAlgo_t convForwardAlg;
  /// Forward pass algorithm workspace size (in bytes)
  size_t convForwardWSSize;

  /// Backward pass filter algorithm
  /** Compute gradient w.r.t. filter. */
  cudnnConvolutionBwdFilterAlgo_t convBackFilterAlg;
  /// Backward pass filter algorithm workspace size (in bytes)
  /** Compute gradient w.r.t. filter. */
  size_t convBackFilterWSSize;

  /// Backward pass data algorithm
  /** Compute gradient w.r.t. data, which is passed to previous layer. */
  cudnnConvolutionBwdDataAlgo_t convBackDataAlg;
  /// Backward pass data algorithm workspace size (in bytes)
  /** Compute gradient w.r.t. data, which is passed to previous layer. */
  size_t convBackDataWSSize;

  /// Input tensor size
  int srcDataSize;
  /// Output tensor size
  int dstDataSize;
  /// Filter size
  int filterSize;

public:
  
  /// Constructor
  CudnnLayer(CudnnNet<DataType>* net) : dnnNet(net) {

    srcTensorDesc = NULL;
    dstTensorDesc = NULL;
    biasTensorDesc = NULL;
    filterDesc = NULL;
    convDesc = NULL;
    poolDesc = NULL;

    convForwardAlg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    convForwardWSSize = 0;
    convBackFilterAlg = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    convBackFilterWSSize = 0;
    convBackDataAlg = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    convBackDataWSSize = 0;

    checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
  }

  /// Destructor
  ~CudnnLayer() {
    if (srcTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
    if (dstTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
    if (biasTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
    if (filterDesc) checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    if (convDesc) checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    if (poolDesc) checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
  }

  /// Setup convolutional layer
  void setupConvLayer(int nConvDims,
                      int* srcTensorDims,
                      int* filterDims,
                      int* dstTensorDims,
                      int* convPads,
                      int* convStrides) {

    // Get parameters from cudnnNet
    std::vector<cudnnHandle_t>& handles = dnnNet->cudnnHandles;
    const cudnnDataType_t cudnnDataType = dnnNet->cudnnDataType;

    // Set input tensor descriptor
    // Example: srcTensorDims[4] = {n,c,h,w}, srcTensorStrides[4] = {c*h*w, h*w, w, 1}
    std::vector<int> srcStrides(nConvDims+2);
    srcStrides[nConvDims + 1] = 1;
    for(int n=nConvDims; n>=0; --n)
      srcStrides[n] = srcStrides[n+1] * srcTensorDims[n+1];
    checkCUDNN(cudnnSetTensorNdDescriptor(srcTensorDesc,
                                          cudnnDataType,
                                          nConvDims + 2,
                                          srcTensorDims,
                                          srcStrides.data()));
  
    // Set filter descriptor
    // Example: filterDims[4] = {output channels, input channels, filter w, filter h}
    const cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW; // still return CUDNN_STATUS_SUCCESS
    checkCUDNN(cudnnSetFilterNdDescriptor(filterDesc,
                                          cudnnDataType,
                                          format,
                                          nConvDims + 2,
                                          filterDims));

    // Set convolution descriptor
    // Note: upscales are not supported as of cuDNN v5.1
    std::vector<int> convUpscales(nConvDims, 1);
    checkCUDNN(cudnnSetConvolutionNdDescriptor(convDesc,
                                               nConvDims,
                                               convPads,
                                               convStrides,
                                               convUpscales.data(),
                                               CUDNN_CROSS_CORRELATION,
                                               cudnnDataType));

    // Determine size of data
    checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                                     srcTensorDesc,
                                                     filterDesc,
                                                     nConvDims + 2,
                                                     dstTensorDims));
    srcDataSize = 1;
    dstDataSize = 1;
    filterSize = 1;
    for(int n=0; n<nConvDims+2; ++n) {
      srcDataSize *= srcTensorDims[n];
      filterSize  *= filterDims[n];
      dstDataSize *= dstTensorDims[n];
    }

    // Set output tensor descriptor
    std::vector<int> dstStrides(nConvDims+2);
    dstStrides[nConvDims + 1] = 1;
    for(int n=nConvDims; n>=0; --n)
      dstStrides[n] = dstStrides[n+1] * dstTensorDims[n+1];
    checkCUDNN(cudnnSetTensorNdDescriptor(dstTensorDesc,
                                          cudnnDataType,
                                          nConvDims + 2,
                                          dstTensorDims,
                                          dstStrides.data()));

    // Initialize forward prop algorithm
    // Note: assume all GPUs are identical to GPU 0
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(handles[0],
                                                   srcTensorDesc,
                                                   filterDesc,
                                                   convDesc,
                                                   dstTensorDesc,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   0,
                                                   &convForwardAlg));
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handles[0],
                                                       srcTensorDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       dstTensorDesc,
                                                       convForwardAlg,
                                                       &convForwardWSSize));

    // Initialize backward prop algorithm
    // Note: assume all GPUs are identical to GPU 0
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(handles[0],
                                                          srcTensorDesc,
                                                          dstTensorDesc,
                                                          convDesc,
                                                          filterDesc,
                                                          CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                          0,
                                                          &convBackFilterAlg));
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(handles[0],
                                                              srcTensorDesc,
                                                              dstTensorDesc,
                                                              convDesc,
                                                              filterDesc,
                                                              convBackFilterAlg,
                                                              &convBackFilterWSSize));
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(handles[0],
                                                        filterDesc,
                                                        dstTensorDesc,
                                                        convDesc,
                                                        srcTensorDesc,
                                                        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                        0,
                                                        &convBackDataAlg));
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handles[0],
                                                            filterDesc,
                                                            dstTensorDesc,
                                                            convDesc,
                                                            srcTensorDesc,
                                                            convBackDataAlg,
                                                            &convBackDataWSSize));

  }

  /// Setup pooling layer
  void setupPoolLayer(int nPoolDims,
                      int* srcTensorDims,
                      int poolMode,
                      int* poolWindowDims,
                      int* poolPads,
                      int* poolStrides,
                      int* dstTensorDims) {

    // Get parameters from cudnnNet
    std::vector<cudnnHandle_t>& handles = dnnNet->cudnnHandles;
    const cudnnDataType_t cudnnDataType = dnnNet->cudnnDataType;

    // Set pooling descriptor
    cudnnPoolingMode_t pmode;
    switch(poolMode) {
    case 0: pmode = CUDNN_POOLING_MAX; break;
    case 1: pmode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING; break;
    case 2: pmode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING; break;
    default:
      std::cerr << "Unknown pooling mode (" << poolMode << "). Using max pooling.\n";
      pmode = CUDNN_POOLING_MAX;
      break;
    }
    checkCUDNN(cudnnSetPoolingNdDescriptor(poolDesc,
                                           pmode,
                                           CUDNN_PROPAGATE_NAN,
                                           nPoolDims,
                                           poolWindowDims,
                                           poolPads,
                                           poolStrides));

    // Set input tensor descriptor
    // Example: srcTensorDims[4] = {n,c,h,w}, srcTensorStrides[4] = {c*h*w, h*w, w, 1}
    std::vector<int> srcStrides(nPoolDims+2);
    srcStrides[nPoolDims + 1] = 1;
    for(int n=nPoolDims; n>=0; --n)
      srcStrides[n] = srcStrides[n+1] * srcTensorDims[n+1];
    checkCUDNN(cudnnSetTensorNdDescriptor(srcTensorDesc,
                                          cudnnDataType,
                                          nPoolDims + 2,
                                          srcTensorDims,
                                          srcStrides.data()));

    // Determine size of data
    checkCUDNN(cudnnGetPoolingNdForwardOutputDim(poolDesc,
                                                 srcTensorDesc,
                                                 nPoolDims + 2,
                                                 dstTensorDims));
    srcDataSize = 1;
    dstDataSize = 1;
    for (int n=0; n<nPoolDims+2; ++n) {
      srcDataSize *= srcTensorDims[n];
      dstDataSize *= dstTensorDims[n];
    }
  
    // Set output tensor descriptor
    std::vector<int> dstStrides(nPoolDims+2);
    dstStrides[nPoolDims + 1] = 1;
    for(int n=nPoolDims; n>=0; --n)
      dstStrides[n] = dstStrides[n+1] * dstTensorDims[n+1];
    checkCUDNN(cudnnSetTensorNdDescriptor(dstTensorDesc,
                                          cudnnDataType,
                                          nPoolDims + 2,
                                          dstTensorDims,
                                          dstStrides.data()));

  }

  void convForward(const int numSamples,
                   const DataType* srcData,
                   const DataType* filterData,
                   DataType* dstData) {

    // Useful constants
    const DataType one = 1.0;
    const DataType zero = 0.0;
 
    // Get parameters from cudnnNet
    std::vector<cudnnHandle_t>& handles = dnnNet->cudnnHandles;
    const int numGPUs = dnnNet->numGPUs;
  
    // Allocate memory on GPU
    std::vector<DataType*> gpuSrcData(numGPUs, NULL);
    std::vector<DataType*> gpuFtrData(numGPUs, NULL);
    std::vector<DataType*> gpuDstData(numGPUs, NULL);
    std::vector<DataType*> gpuWorkSpaces(numGPUs, NULL);
    for(int dev=0; dev<numGPUs; ++dev) {
      checkCUDA(cudaSetDevice(dev));
      checkCUDA(cudaMalloc(&gpuSrcData[dev], sizeof(DataType)*srcDataSize));
      checkCUDA(cudaMalloc(&gpuDstData[dev], sizeof(DataType)*dstDataSize));
      checkCUDA(cudaMalloc(&gpuFtrData[dev], sizeof(DataType)*filterSize));
      checkCUDA(cudaMemcpyAsync(gpuFtrData[dev],
                                filterData,
                                sizeof(DataType)*filterSize,
                                cudaMemcpyHostToDevice));
      if(convForwardWSSize > 0)
        checkCUDA(cudaMalloc(&gpuWorkSpaces[dev], convForwardWSSize));
    }

    // Perform convolution on data samples
    for(int i=0; i<numSamples; ++i) {

      // Determine GPU
      int dev = i % numGPUs;
      checkCUDA(cudaSetDevice(dev));

      // Transfer input data to GPU
      checkCUDA(cudaMemcpyAsync(gpuSrcData[dev],
                                srcData+i*srcDataSize,
                                sizeof(DataType)*srcDataSize,
                                cudaMemcpyHostToDevice));
      
      // Perform convolution
      checkCUDNN(cudnnConvolutionForward(handles[dev],
                                         &one,
                                         srcTensorDesc,
                                         gpuSrcData[dev],
                                         filterDesc,
                                         gpuFtrData[dev],
                                         convDesc,
                                         convForwardAlg,
                                         gpuWorkSpaces[dev],
                                         convForwardWSSize,
                                         &zero,
                                         dstTensorDesc,
                                         gpuDstData[dev]));
      
      // Transfer output data from GPU
      checkCUDA(cudaMemcpyAsync(dstData+i*dstDataSize,
                                gpuDstData[dev],
                                sizeof(DataType)*dstDataSize,
                                cudaMemcpyDeviceToHost));

    }
    
    // Free memory on GPU
    // Note: cudaFree is synchronous
    for(int dev=0; dev<numGPUs; ++dev) {
      checkCUDA(cudaSetDevice(dev));
      checkCUDA(cudaFree(gpuSrcData[dev]));
      checkCUDA(cudaFree(gpuDstData[dev]));
      checkCUDA(cudaFree(gpuFtrData[dev]));
      checkCUDA(cudaFree(gpuWorkSpaces[dev]));
    }
  
  }

  // Back filter: Given src=x (g_input), diff=dJ/dy (g_outputDelta), computes grad = dJ/dw (g_filterDelta)
  // Back data: Given filter=w (g_filter), diff = dJ/dy (g_outputDelta), computes grad = dJ/dx (g_inputDelta)
  void convBackward(const int numSamples,
                    const DataType* srcData,
                    const DataType* filterData,
                    const DataType* dstDelta,
                    DataType* filterDelta,
                    DataType* srcDelta) {

    // Useful constants
    const DataType one = 1.0;
    const DataType zero = 0.0;

    // Get parameters from cudnnNet
    std::vector<cudnnHandle_t>& handles = dnnNet->cudnnHandles;
    const int numGPUs = dnnNet->numGPUs;

    // Allocate memory on GPU
    std::vector<DataType*> gpuSrcData(numGPUs, NULL);
    std::vector<DataType*> gpuFtrData(numGPUs, NULL);
    std::vector<DataType*> gpuSrcDelta(numGPUs, NULL);
    std::vector<DataType*> gpuDstDelta(numGPUs, NULL);
    std::vector<DataType*> gpuFtrDelta(numGPUs, NULL);
    std::vector<DataType*> gpuFilterWorkSpaces(numGPUs, NULL);
    std::vector<DataType*> gpuDataWorkSpaces(numGPUs, NULL);
    for(int dev=0; dev<numGPUs; ++dev) {
      checkCUDA(cudaSetDevice(dev));
      checkCUDA(cudaMalloc(&gpuSrcData[dev], sizeof(DataType)*srcDataSize));
      checkCUDA(cudaMalloc(&gpuFtrData[dev], sizeof(DataType)*filterSize));
      checkCUDA(cudaMalloc(&gpuSrcDelta[dev], sizeof(DataType)*srcDataSize));
      checkCUDA(cudaMalloc(&gpuDstDelta[dev], sizeof(DataType)*dstDataSize));
      checkCUDA(cudaMalloc(&gpuFtrDelta[dev], sizeof(DataType)*filterSize));
      checkCUDA(cudaMemcpyAsync(gpuFtrData[dev],
                                filterData,
                                sizeof(DataType)*filterSize,
                                cudaMemcpyHostToDevice));
      checkCUDA(cudaMemset(gpuFtrDelta[dev], 0, sizeof(DataType)*filterSize));
      if(convBackFilterWSSize > 0)
        checkCUDA(cudaMalloc(&gpuFilterWorkSpaces[dev], convBackFilterWSSize));
      if(convBackDataWSSize > 0)
        checkCUDA(cudaMalloc(&gpuDataWorkSpaces[dev], convBackDataWSSize));
    }

    // Perform back prop on data samples
    for(int i=0; i<numSamples; ++i) {

      // Determine GPU
      int dev = i % numGPUs;
      checkCUDA(cudaSetDevice(dev));

      // Transfer data and next layer's delta to GPU
      checkCUDA(cudaMemcpyAsync(gpuSrcData[dev],
                                srcData+i*srcDataSize,
                                sizeof(DataType)*srcDataSize,
                                cudaMemcpyHostToDevice));
      checkCUDA(cudaMemcpyAsync(gpuDstDelta[dev],
                                dstDelta+i*dstDataSize,
                                sizeof(DataType)*dstDataSize,
                                cudaMemcpyHostToDevice));
      
      // Perform back prop
      checkCUDNN(cudnnConvolutionBackwardFilter(handles[dev],
                                                &one,
                                                srcTensorDesc,
                                                gpuSrcData[dev],
                                                dstTensorDesc,
                                                gpuDstDelta[dev],
                                                convDesc,
                                                convBackFilterAlg,
                                                gpuFilterWorkSpaces[dev],
                                                convBackFilterWSSize,
                                                &one,
                                                filterDesc,
                                                gpuFtrDelta[dev]));
      checkCUDNN(cudnnConvolutionBackwardData(handles[dev],
                                              &one,
                                              filterDesc,
                                              gpuFtrData[dev],
                                              dstTensorDesc,
                                              gpuDstDelta[dev],
                                              convDesc,
                                              convBackDataAlg,
                                              gpuDataWorkSpaces[dev],
                                              convBackDataWSSize,
                                              &zero,
                                              srcTensorDesc,
                                              gpuSrcDelta[dev]));

      // Transfer data delta from GPU
      checkCUDA(cudaMemcpyAsync(srcDelta+i*srcDataSize,
                                gpuSrcDelta[dev],
                                sizeof(DataType)*srcDataSize,
                                cudaMemcpyDeviceToHost));

    }

    // Transfer filter deltas from GPU and accumulate
    DataType* filterDeltaTemp = new DataType[filterSize];
    memset(filterDelta, 0, sizeof(DataType)*filterSize);
    for(int dev=0; dev<numGPUs; ++dev) {
      checkCUDA(cudaMemcpy(filterDeltaTemp,
                           gpuFtrDelta[dev],
                           sizeof(DataType)*filterSize,
                           cudaMemcpyDeviceToHost));
      // TODO: use BLAS AXPY
      for(int i=0; i<filterSize; ++i)
        filterDelta[i] += filterDeltaTemp[i];
    }
    delete filterDeltaTemp;

    // Free memory on GPU
    // Note: cudaFree is synchronous
    for(int dev=0; dev<numGPUs; ++dev) {
      checkCUDA(cudaSetDevice(dev));
      checkCUDA(cudaFree(gpuSrcData[dev]));
      checkCUDA(cudaFree(gpuFtrData[dev]));
      checkCUDA(cudaFree(gpuSrcDelta[dev]));
      checkCUDA(cudaFree(gpuDstDelta[dev]));
      checkCUDA(cudaFree(gpuFtrDelta[dev]));
      checkCUDA(cudaFree(gpuFilterWorkSpaces[dev]));
      checkCUDA(cudaFree(gpuDataWorkSpaces[dev]));
    }
  
  }

  void poolForward(const int numSamples,
                   const DataType* srcData,
                   DataType* dstData) {

    // Useful constants
    const DataType one = 1.0;
    const DataType zero = 0.0;

    // Get parameters from cudnnNet
    std::vector<cudnnHandle_t>& handles = dnnNet->cudnnHandles;
    const int numGPUs = dnnNet->numGPUs;

    // Allocate memory on GPU
    std::vector<DataType*> gpuSrcData(numGPUs, NULL);
    std::vector<DataType*> gpuDstData(numGPUs, NULL);
    for(int dev=0; dev<numGPUs; ++dev) {
      checkCUDA(cudaSetDevice(dev));
      checkCUDA(cudaMalloc(&gpuSrcData[dev], sizeof(DataType)*srcDataSize));
      checkCUDA(cudaMalloc(&gpuDstData[dev], sizeof(DataType)*dstDataSize));
    }
    
    // Perform pooling on data samples
    for(int i=0; i<numSamples; ++i) {

      // Determine GPU
      int dev = i % numGPUs;
      checkCUDA(cudaSetDevice(dev));

      // Transfer input data to GPU
      checkCUDA(cudaMemcpyAsync(gpuSrcData[dev],
                                srcData+i*srcDataSize,
                                sizeof(DataType)*srcDataSize,
                                cudaMemcpyHostToDevice));

      // Perform pooling
      checkCUDNN(cudnnPoolingForward(handles[dev],
                                     poolDesc,
                                     &one,
                                     srcTensorDesc,
                                     gpuSrcData[dev],
                                     &zero,
                                     dstTensorDesc,
                                     gpuDstData[dev]));

      // Transfer output data from GPU
      checkCUDA(cudaMemcpyAsync(dstData+i*dstDataSize,
                                gpuDstData[dev],
                                sizeof(DataType)*dstDataSize,
                                cudaMemcpyDeviceToHost));
 
    }

    // Free memory on GPU
    for(int dev=0; dev<numGPUs; ++dev) {
      checkCUDA(cudaSetDevice(dev));
      checkCUDA(cudaFree(gpuSrcData[dev]));
      checkCUDA(cudaFree(gpuDstData[dev]));
    }
    
  }

  // Given src=y (g_output), srcdiff = dJ/dy (g_outputDelta), dest=x (g_input), destdiff = dJ/dx
  void poolBackward(const int numSamples,
                    const DataType* srcData,
                    const DataType* dstData,
                    const DataType* dstDelta,
                    DataType* srcDelta) {

    // Useful constants
    const DataType one = 1.0;
    const DataType zero = 0.0;

    // Get parameters from cudnnNet
    std::vector<cudnnHandle_t>& handles = dnnNet->cudnnHandles;
    const int numGPUs = dnnNet->numGPUs;

    // Allocate memory on GPU
    std::vector<DataType*> gpuSrcData(numGPUs, NULL);
    std::vector<DataType*> gpuSrcDelta(numGPUs, NULL);
    std::vector<DataType*> gpuDstData(numGPUs, NULL);
    std::vector<DataType*> gpuDstDelta(numGPUs, NULL);
    for(int dev=0; dev<numGPUs; ++dev) {
      checkCUDA(cudaSetDevice(dev));
      checkCUDA(cudaMalloc(&gpuSrcData[dev], sizeof(DataType)*srcDataSize));
      checkCUDA(cudaMalloc(&gpuSrcDelta[dev], sizeof(DataType)*srcDataSize));
      checkCUDA(cudaMalloc(&gpuDstData[dev], sizeof(DataType)*dstDataSize));
      checkCUDA(cudaMalloc(&gpuDstDelta[dev], sizeof(DataType)*dstDataSize));
    }

    // Perform back prop on data samples
    for(int i=0; i<numSamples; ++i) {

      // Determine GPU
      int dev = i % numGPUs;
      checkCUDA(cudaSetDevice(dev));

      // Transfer data and next layer's delta to GPU
      checkCUDA(cudaMemcpyAsync(gpuSrcData[dev],
                                srcData+i*srcDataSize,
                                sizeof(DataType)*srcDataSize,
                                cudaMemcpyHostToDevice));
      checkCUDA(cudaMemcpyAsync(gpuDstData[dev],
                                dstData+i*dstDataSize,
                                sizeof(DataType)*dstDataSize,
                                cudaMemcpyHostToDevice));
      checkCUDA(cudaMemcpyAsync(gpuDstDelta[dev],
                                dstDelta+i*dstDataSize,
                                sizeof(DataType)*dstDataSize,
                                cudaMemcpyHostToDevice));

      // Perform back prop 
      checkCUDNN(cudnnPoolingBackward(handles[dev],
                                      poolDesc,
                                      &one,
                                      dstTensorDesc,
                                      gpuDstData[dev],
                                      dstTensorDesc,
                                      gpuDstDelta[dev],
                                      srcTensorDesc,
                                      gpuSrcData[dev],
                                      &zero,
                                      srcTensorDesc,
                                      gpuSrcDelta[dev]));

      // Transfer delta from GPU
      checkCUDA(cudaMemcpyAsync(srcDelta+i*srcDataSize,
                                gpuSrcDelta[dev],
                                sizeof(DataType)*srcDataSize,
                                cudaMemcpyDeviceToHost));

    }

    // Free memory on GPU
    for(int dev=0; dev<numGPUs; ++dev) {
      checkCUDA(cudaSetDevice(dev));
      checkCUDA(cudaFree(gpuSrcData[dev]));
      checkCUDA(cudaFree(gpuSrcDelta[dev]));
      checkCUDA(cudaFree(gpuDstData[dev]));
      checkCUDA(cudaFree(gpuDstDelta[dev]));
    }

  }

#if 0
  void setupActLayer(int nTensorDims, int* TensorDims)
  {
    cudnnDataType_t datatype = dnnNet->cudnnDataType;
  
    int strides[5];
    strides[nTensorDims - 1] = 1;
    for (int n = nTensorDims - 2; n >= 0; n--)
      strides[n] = strides[n + 1] * TensorDims[n + 1];

    // set input tensor descriptor
    checkCUDNN(cudnnSetTensorNdDescriptor(srcTensorDesc,
                                          datatype, nTensorDims,
                                          TensorDims,
                                          strides));

    // set output tensor descriptor
    checkCUDNN(cudnnSetTensorNdDescriptor(dstTensorDesc,
                                          datatype, nTensorDims,
                                          TensorDims,
                                          strides));

    srcDataSize = 1;
    dstDataSize = 1;
    for (int n = 0; n < nTensorDims; n++) {
      srcDataSize *= TensorDims[n];
      dstDataSize *= TensorDims[n];
    }
  }

  void setupSoftmaxLayer(int nTensorDims, int* TensorDims)
  {
    this->setupActLayer(nTensorDims, TensorDims);
  }

  bool actForward(DataType* srcData, DataType* dstData)
  {
    cudnnHandle_t handle = dnnNet->cudnnHandles;

    DataType alpha = 1.0f;
    DataType beta = 0.0f;
    checkCUDNN(cudnnActivationForward(handle, &alpha,
                                      srcTensorDesc, srcData, &beta, dstTensorDesc, dstData));

    return true;
  }

  bool actBackward(DataType* srcData, DataType* dstData, DataType* dstDelta, DataType* srcDelta)
  {
    cudnnHandle_t handle = dnnNet->cudnnHandles;

    DataType alpha = 1.0f;
    DataType beta = 0.0f;
    checkCUDNN(cudnnActivationBackward(handle, &alpha,
                                       dstTensorDesc, dstData, dstTensorDesc, dstDelta,
                                       srcTensorDesc, srcData, &beta, dstTensorDesc, srcDelta));

    return true;
  }

  bool softmaxForward(DataType* srcData, DataType* dstData)
  {
    cudnnHandle_t handle = dnnNet->cudnnHandles;

    DataType alpha = 1.0f;
    DataType beta = 0.0f;
    checkCUDNN(cudnnSoftmaxForward(handle,
                                   CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
                                   srcTensorDesc, srcData, &beta, dstTensorDesc, dstData));

    return true;
  }
 
  bool actForwardHost(DataType* srcDataHost, DataType* dstDataHost)
  {
    DataType *srcData, *dstData;
    checkCUDA(cudaMalloc(&srcData, sizeof(DataType) * srcDataSize));
    checkCUDA(cudaMalloc(&dstData, sizeof(DataType) * dstDataSize));

    checkCUDA(cudaMemcpy(srcData, srcDataHost, sizeof(DataType) * srcDataSize, cudaMemcpyHostToDevice));
    this->actForward(srcData, dstData);
    checkCUDA(cudaMemcpy(dstDataHost, dstData, sizeof(DataType) * dstDataSize, cudaMemcpyDeviceToHost));

    checkCUDA(cudaFree(srcData));
    checkCUDA(cudaFree(dstData));
  }

  bool softmaxForwardHost(DataType* srcDataHost, DataType* dstDataHost)
  {
    DataType *srcData, *dstData;
    checkCUDA(cudaMalloc(&srcData, sizeof(DataType) * srcDataSize));
    checkCUDA(cudaMalloc(&dstData, sizeof(DataType) * dstDataSize));

    checkCUDA(cudaMemcpy(srcData, srcDataHost, sizeof(DataType) * srcDataSize, cudaMemcpyHostToDevice));
    this->softmaxForward(srcData, dstData);
    checkCUDA(cudaMemcpy(dstDataHost, dstData, sizeof(DataType) * dstDataSize, cudaMemcpyDeviceToHost));

    checkCUDA(cudaFree(srcData));
    checkCUDA(cudaFree(dstData));
    return true;
  }

  bool softmaxBackward(DataType* srcData, DataType* dstData, DataType* dstDelta, DataType* srcDelta)
  {
    cudnnHandle_t handle = dnnNet->cudnnHandle;

    DataType alpha = 1.0f;
    DataType beta = 0.0f;
    checkCUDNN(cudnnSoftmaxBackward(handle,
                                    CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
                                    dstTensorDesc, dstData, dstTensorDesc, dstDelta, &beta,
                                    srcTensorDesc, srcDelta));

    return true;
  }
#endif

};

#else  // __LIB_CUDNN

template <class T>
class CudnnNet {};

#endif  // __LIB_CUDNN

#endif // CUDNN_WRAPPER_HPP_INCLUDED
