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


// error utility
#define checkCUDA(status) {                                             \
    if (status != 0) {                                                  \
        std::cerr << "cuda error: " << cudaGetErrorString(status);      \
        cudaDeviceReset();                                              \
        exit(0);                                                        \
    }                                                                   \
}

#define checkCUDNN(status) {                                            \
    if (status != CUDNN_STATUS_SUCCESS) {                               \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status);    \
        cudaDeviceReset();                                              \
        exit(0);                                                        \
    }                                                                   \
}


template <class DataType>
class CudnnNet
{
public:
    CudnnNet()
	{
        cudnnHandle = NULL;
        checkCUDNN(cudnnCreate(&cudnnHandle));
		cudnnDataType = ((sizeof(DataType) == 4) ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE);
	}
    ~CudnnNet()
    {
        if (cudnnHandle) checkCUDNN(cudnnDestroy(cudnnHandle));
    }

    void printVersion()
	{
	    int version = (int)cudnnGetVersion();
	    std::cout << "cudnnGetVersion() :" << version << ", CUDNN_VERSION from cudnn.h : " << CUDNN_VERSION << endl;
	}

public:
    cudnnHandle_t               cudnnHandle;
    cudnnDataType_t 			cudnnDataType;

};


template <class DataType>
class CudnnLayer
{
public:
    CudnnLayer(CudnnNet<DataType>* net) : dnnNet(net)
    {
        srcTensorDesc = NULL;
        dstTensorDesc = NULL;
        biasTensorDesc = NULL;
        filterDesc = NULL;
        convDesc = NULL;
        poolDesc = NULL;

        convForwardAlg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        convForwardWSSize = 0;
        convForwardWS = NULL;
        convBackFilterAlg = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        convBackFilterWSSize = 0;
        convBackFilterWS = NULL;
        convBackDataAlg = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
		convBackDataWSSize = 0;
        convBackDataWS = NULL;

        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    }
    ~CudnnLayer()
    {
        if (srcTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
        if (dstTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
        if (biasTensorDesc) checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
        if (filterDesc) checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
        if (convDesc) checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
        if (poolDesc) checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
    }

    void setupConvLayer(int nConvDims, int* srcTensorDims, int* filterDims, int* dstTensorDims)
    {
    	cudnnHandle_t handle = dnnNet->cudnnHandle;
		cudnnDataType_t datatype = dnnNet->cudnnDataType;

		// set input tensor descriptor
		// example: srcTensorDims[4] = {n,c,h,w}, srcTensorStrides[4] = {c*h*w, h*w, w, 1}
		int srcstrides[5];
		srcstrides[nConvDims + 1] = 1;
		for (int n = nConvDims; n >= 0; n--)
			srcstrides[n] = srcstrides[n + 1] * srcTensorDims[n + 1];
		checkCUDNN(cudnnSetTensorNdDescriptor(srcTensorDesc,
											  datatype, nConvDims + 2,
											  srcTensorDims,
											  srcstrides));

		// set filter descriptor
		// example: filterDims[4] = {output channels, input channels, filter w, filter h}
		cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW; // still return CUDNN_STATUS_SUCCESS
		checkCUDNN(cudnnSetFilterNdDescriptor(filterDesc,
											  datatype,
											  format,
											  nConvDims + 2,
											  filterDims));

		// set convolutional descriptor
		int conv_pad[3] = {0, 0, 0};
		int conv_stride[3] = {1, 1, 1};
		int conv_upscale[3] = {1, 1, 1};
		checkCUDNN(cudnnSetConvolutionNdDescriptor(convDesc, nConvDims,
												   conv_pad, conv_stride, conv_upscale,
												   CUDNN_CROSS_CORRELATION, datatype));

		// set output tensor descriptor
        checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(convDesc, srcTensorDesc,
                                                filterDesc, nConvDims + 2,
												dstTensorDims));

		int dststrides[5];
		dststrides[nConvDims + 1] = 1;
		for (int n = nConvDims; n >= 0; n--)
			dststrides[n] = dststrides[n + 1] * dstTensorDims[n + 1];
		checkCUDNN(cudnnSetTensorNdDescriptor(dstTensorDesc,
											  datatype, nConvDims + 2,
											  dstTensorDims,
											  dststrides));

        // setup workspace
    	// get the fastest algorithm
    	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
    			handle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convForwardAlg));

    	// get its workspace size
    	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
    			handle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
				convForwardAlg, &convForwardWSSize));

    	//printf("Fastest forward algorithm: %d (workspace: %lu)\n", g_forward_alg, g_forward_wssize);

    	// get the fastest algorithm for filter
    	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
    			handle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
				CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &convBackFilterAlg));

    	// get its workspace size
    	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    			handle, srcTensorDesc, dstTensorDesc, convDesc, filterDesc,
				convBackFilterAlg, &convBackFilterWSSize));

    	//printf("Fastest backward filter algorithm: %d (workspace: %lu)\n", g_backward_filter_alg, g_backward_filter_wssize);

    	// get the fastest algorithm for data
    	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
            handle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
			CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &convBackDataAlg));

    	// get its workspace size
    	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, filterDesc, dstTensorDesc, convDesc, srcTensorDesc,
			convBackDataAlg, &convBackDataWSSize));

    	srcDataSize = 1;
    	dstDataSize = 1;
    	filterSize = 1;
    	for (int n = 0; n < nConvDims + 2; n++) {
    		srcDataSize *= srcTensorDims[n];
    		dstDataSize *= dstTensorDims[n];
    		filterSize  *= filterDims[n];
    	}
    	//printf("srcDataSize=%d, dstDataSize=%d, filterSize=%d\b", srcDataSize, dstDataSize, filterSize);
    }

    void setupPoolLayer(int nPoolDims, int* srcTensorDims, int poolMode,
    		int* poolWindowDims, int* poolPads,  int* poolStrides, int* dstTensorDims)
    {
		cudnnDataType_t datatype = dnnNet->cudnnDataType;
		cudnnPoolingMode_t pmode = CUDNN_POOLING_MAX;
    	if (poolMode == 1)
    		pmode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    	else if (poolMode == 2)
    		pmode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

    	// set pooling descriptor
    	checkCUDNN(cudnnSetPoolingNdDescriptor(poolDesc, pmode, CUDNN_PROPAGATE_NAN,
    			nPoolDims, poolWindowDims, poolPads, poolStrides));

    	// set input tensor descriptor
		int srcstrides[5];
		srcstrides[nPoolDims + 1] = 1;
		for (int n = nPoolDims; n >= 0; n--)
			srcstrides[n] = srcstrides[n + 1] * srcTensorDims[n + 1];
		checkCUDNN(cudnnSetTensorNdDescriptor(srcTensorDesc,
											  datatype, nPoolDims + 2,
											  srcTensorDims,
											  srcstrides));

		// set output tensor descriptor
        checkCUDNN(cudnnGetPoolingNdForwardOutputDim(
        		poolDesc, srcTensorDesc, nPoolDims + 2, dstTensorDims));

		int dststrides[5];
		dststrides[nPoolDims + 1] = 1;
		for (int n = nPoolDims; n >= 0; n--)
			dststrides[n] = dststrides[n + 1] * dstTensorDims[n + 1];
		checkCUDNN(cudnnSetTensorNdDescriptor(dstTensorDesc,
											  datatype, nPoolDims + 2,
											  dstTensorDims,
											  dststrides));

    	srcDataSize = 1;
    	dstDataSize = 1;
    	for (int n = 0; n < nPoolDims + 2; n++) {
    		srcDataSize *= srcTensorDims[n];
    		dstDataSize *= dstTensorDims[n];
    	}
    }

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

    bool convForward(DataType* srcData, DataType* filterData, DataType* dstData)
    {
    	cudnnHandle_t handle = dnnNet->cudnnHandle;

		// allocate workspace
        if (convForwardWSSize > 0) checkCUDA(cudaMalloc((void**)&convForwardWS, convForwardWSSize));

        DataType alpha = 1.0f;
        DataType beta = 0.0f;
        checkCUDNN(cudnnConvolutionForward(handle,
										   &alpha,
										   srcTensorDesc,
										   srcData,
										   filterDesc,
										   filterData,
										   convDesc,
										   convForwardAlg,
										   convForwardWS,
										   convForwardWSSize,
										   &beta,
										   dstTensorDesc,
										   dstData));
        //addBias(dstTensorDesc, conv, c, *dstTensorData);

        // free workspace
	    if (convForwardWS) { checkCUDA(cudaFree(convForwardWS)); convForwardWS = NULL; }

		return true;
    }

    // Back filter: Given src=x (g_input), diff=dJ/dy (g_outputDelta), computes grad = dJ/dw (g_filterDelta)
    // Back data: Given filter=w (g_filter), diff = dJ/dy (g_outputDelta), computes grad = dJ/dx (g_inputDelta)
	bool convBackward(DataType* srcData, DataType* filterData, DataType* dstDelta, DataType* filterDelta, DataType* srcDelta)
    {
    	cudnnHandle_t handle = dnnNet->cudnnHandle;

    	// allocate workspace
        if (convBackFilterWSSize > 0) checkCUDA(cudaMalloc(&convBackFilterWS, convBackFilterWSSize));
        if (convBackDataWSSize > 0) checkCUDA(cudaMalloc(&convBackDataWS, convBackDataWSSize));

        DataType alpha = 1.0f;
        DataType beta = 0.0f;
		checkCUDNN(cudnnConvolutionBackwardFilter(handle, &alpha,
				srcTensorDesc, srcData, dstTensorDesc, dstDelta,
				convDesc, convBackFilterAlg, convBackFilterWS, convBackFilterWSSize,
				&beta, filterDesc, filterDelta));

		checkCUDNN(cudnnConvolutionBackwardData(handle, &alpha,
				filterDesc, filterData, dstTensorDesc, dstDelta,
				convDesc, convBackDataAlg, convBackDataWS, convBackDataWSSize,
				&beta, srcTensorDesc, srcDelta));

        // free workspace
	    if (convBackFilterWS) { checkCUDA(cudaFree(convBackFilterWS)); convBackFilterWS = NULL; }
	    if (convBackDataWS) { checkCUDA(cudaFree(convBackDataWS)); convBackDataWS = NULL; }

		return true;
    }

    bool convForwardHost(DataType* srcDataHost, DataType* filterDataHost, DataType* dstDataHost)
    {
    	DataType *srcData, *ftrData, *dstData;
    	checkCUDA(cudaMalloc(&srcData, sizeof(DataType) * srcDataSize));
    	checkCUDA(cudaMalloc(&ftrData, sizeof(DataType) * filterSize));
    	checkCUDA(cudaMalloc(&dstData, sizeof(DataType) * dstDataSize));

    	checkCUDA(cudaMemcpy(srcData, srcDataHost, sizeof(DataType) * srcDataSize, cudaMemcpyHostToDevice));
    	checkCUDA(cudaMemcpy(ftrData, filterDataHost, sizeof(DataType) * filterSize, cudaMemcpyHostToDevice));
    	this->convForward(srcData, ftrData, dstData);
    	checkCUDA(cudaMemcpy(dstDataHost, dstData, sizeof(DataType) * dstDataSize, cudaMemcpyDeviceToHost));

    	checkCUDA(cudaFree(srcData));
    	checkCUDA(cudaFree(ftrData));
    	checkCUDA(cudaFree(dstData));
    	return true;
    }

    bool convBackwardHost(DataType* srcDataHost, DataType* filterDataHost, DataType* dstDeltaHost, DataType* filterDeltaHost, DataType* srcDeltaHost)
    {
    	DataType *srcData, *ftrData, *dstDelta, *ftrDelta, *srcDelta;
    	checkCUDA(cudaMalloc(&srcData, sizeof(DataType) * srcDataSize));
    	checkCUDA(cudaMalloc(&ftrData, sizeof(DataType) * filterSize));
    	checkCUDA(cudaMalloc(&dstDelta, sizeof(DataType) * dstDataSize));
    	checkCUDA(cudaMalloc(&ftrDelta, sizeof(DataType) * filterSize));
    	checkCUDA(cudaMalloc(&srcDelta, sizeof(DataType) * srcDataSize));

    	checkCUDA(cudaMemcpy(srcData, srcDataHost, sizeof(DataType) * srcDataSize, cudaMemcpyHostToDevice));
    	checkCUDA(cudaMemcpy(ftrData, filterDataHost, sizeof(DataType) * filterSize, cudaMemcpyHostToDevice));
    	checkCUDA(cudaMemcpy(dstDelta, dstDeltaHost, sizeof(DataType) * dstDataSize, cudaMemcpyHostToDevice));
    	this->convBackward(srcData, ftrData, dstDelta, ftrDelta, srcDelta);
    	checkCUDA(cudaMemcpy(filterDeltaHost, ftrDelta, sizeof(DataType) * filterSize, cudaMemcpyDeviceToHost));
    	checkCUDA(cudaMemcpy(srcDeltaHost, srcDelta, sizeof(DataType) * srcDataSize, cudaMemcpyDeviceToHost));

    	checkCUDA(cudaFree(srcData));
    	checkCUDA(cudaFree(ftrData));
    	checkCUDA(cudaFree(dstDelta));
    	checkCUDA(cudaFree(ftrDelta));
    	checkCUDA(cudaFree(srcDelta));
    	return true;
    }

	bool poolForward(DataType* srcData, DataType* dstData)
	{
    	cudnnHandle_t handle = dnnNet->cudnnHandle;

    	DataType alpha = 1.0f;
        DataType beta = 0.0f;
	    checkCUDNN(cudnnPoolingForward(handle, poolDesc, &alpha,
	    		srcTensorDesc, srcData, &beta,
				dstTensorDesc, dstData));

	    return true;
	}

	// Given src=y (g_output), srcdiff = dJ/dy (g_outputDelta), dest=x (g_input), destdiff = dJ/dx
	bool poolBackward(DataType* srcData, DataType* dstData, DataType* dstDelta, DataType* srcDelta)
	{
    	cudnnHandle_t handle = dnnNet->cudnnHandle;

    	DataType alpha = 1.0f;
        DataType beta = 0.0f;
    	checkCUDNN(cudnnPoolingBackward(handle, poolDesc, &alpha,
    			dstTensorDesc, dstData, dstTensorDesc, dstDelta,
				srcTensorDesc, srcData, &beta, srcTensorDesc, srcDelta));

		return true;
	}

    bool poolForwardHost(DataType* srcDataHost, DataType* dstDataHost)
    {
    	DataType *srcData, *dstData;
    	checkCUDA(cudaMalloc(&srcData, sizeof(DataType) * srcDataSize));
    	checkCUDA(cudaMalloc(&dstData, sizeof(DataType) * dstDataSize));

    	checkCUDA(cudaMemcpy(srcData, srcDataHost, sizeof(DataType) * srcDataSize, cudaMemcpyHostToDevice));
    	this->poolForward(srcData, dstData);
    	checkCUDA(cudaMemcpy(dstDataHost, dstData, sizeof(DataType) * dstDataSize, cudaMemcpyDeviceToHost));

    	checkCUDA(cudaFree(srcData));
    	checkCUDA(cudaFree(dstData));
    	return true;
    }

    bool poolBackwardHost(DataType* srcDataHost, DataType* dstDataHost, DataType* dstDeltaHost, DataType* srcDeltaHost)
    {
    	DataType *srcData, *dstData, *dstDelta, *srcDelta;
    	checkCUDA(cudaMalloc(&srcData, sizeof(DataType) * srcDataSize));
    	checkCUDA(cudaMalloc(&dstData, sizeof(DataType) * dstDataSize));
    	checkCUDA(cudaMalloc(&dstDelta, sizeof(DataType) * dstDataSize));
    	checkCUDA(cudaMalloc(&srcDelta, sizeof(DataType) * srcDataSize));

    	checkCUDA(cudaMemcpy(srcData, srcDataHost, sizeof(DataType) * srcDataSize, cudaMemcpyHostToDevice));
    	checkCUDA(cudaMemcpy(dstData, dstDataHost, sizeof(DataType) * dstDataSize, cudaMemcpyHostToDevice));
    	checkCUDA(cudaMemcpy(dstDelta, dstDeltaHost, sizeof(DataType) * dstDataSize, cudaMemcpyHostToDevice));
    	this->poolBackward(srcData, dstData, dstDelta, srcDelta);
    	checkCUDA(cudaMemcpy(srcDeltaHost, srcDelta, sizeof(DataType) * srcDataSize, cudaMemcpyDeviceToHost));

    	checkCUDA(cudaFree(srcData));
    	checkCUDA(cudaFree(dstData));
    	checkCUDA(cudaFree(dstDelta));
    	checkCUDA(cudaFree(srcDelta));
    	return true;
    }

    bool actForward(DataType* srcData, DataType* dstData)
	{
    	cudnnHandle_t handle = dnnNet->cudnnHandle;

    	DataType alpha = 1.0f;
        DataType beta = 0.0f;
    	checkCUDNN(cudnnActivationForward(handle, &alpha,
				srcTensorDesc, srcData, &beta, dstTensorDesc, dstData));

    	return true;
	}

	bool actBackward(DataType* srcData, DataType* dstData, DataType* dstDelta, DataType* srcDelta)
	{
		cudnnHandle_t handle = dnnNet->cudnnHandle;

		DataType alpha = 1.0f;
		DataType beta = 0.0f;
		checkCUDNN(cudnnActivationBackward(handle, &alpha,
				dstTensorDesc, dstData, dstTensorDesc, dstDelta,
				srcTensorDesc, srcData, &beta, dstTensorDesc, srcDelta));

		return true;
	}

	bool softmaxForward(DataType* srcData, DataType* dstData)
	{
    	cudnnHandle_t handle = dnnNet->cudnnHandle;

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

public:
	CudnnNet<DataType>* 			dnnNet;

    cudnnTensorDescriptor_t         srcTensorDesc;            // input tensor descriptor
    cudnnTensorDescriptor_t         dstTensorDesc;            // output tensor descriptor
    cudnnTensorDescriptor_t         biasTensorDesc;           // bias tensor descriptor
    cudnnFilterDescriptor_t         filterDesc;               // filter descriptor
    cudnnConvolutionDescriptor_t    convDesc;                 // convolutional descriptor
    cudnnPoolingDescriptor_t        poolDesc;                 // pool descriptor

    cudnnConvolutionFwdAlgo_t       convForwardAlg;           // cudnn forward pass algorithm
    size_t                          convForwardWSSize;        // cudnn forward pass workspace size
    void*                           convForwardWS;            // cudnn forward pass workspace

    cudnnConvolutionBwdFilterAlgo_t convBackFilterAlg;        // cudnn backward pass filter algorithm
    size_t                          convBackFilterWSSize;     // cudnn backward pass filter workspace size
    void*                           convBackFilterWS;         // cudnn backward pass filter workspace

    cudnnConvolutionBwdDataAlgo_t   convBackDataAlg;          // cudnn backward pass data algorithm
    size_t                          convBackDataWSSize;       // cudnn backward pass data workspace size
    void*                           convBackDataWS;           // cudnn backward pass data workspace

    int								srcDataSize;
    int								dstDataSize;
    int								filterSize;

};

#else  // __LIB_CUDNN

template <class T>
class CudnnNet {};

#endif  // __LIB_CUDNN

#endif // CUDNN_WRAPPER_HPP_INCLUDED
