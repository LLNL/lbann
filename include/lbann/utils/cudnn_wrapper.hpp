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
// cudnn_wrapper .hpp .cpp - cuDNN support - wrapper classes, utility functions
////////////////////////////////////////////////////////////////////////////////

#ifndef CUDNN_WRAPPER_HPP_INCLUDED
#define CUDNN_WRAPPER_HPP_INCLUDED

#ifdef __LIB_CUDNN

#include <vector>
#include <cuda.h>
#include <cudnn.h>
#include <cub/util_allocator.cuh>
#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"


// moved to make it visible from lbann_layer_convolution
#include "lbann/utils/lbann_exception.hpp"
#define checkCUDA(status) {                                             \
    if (status != cudaSuccess) {                                        \
      std::cerr << "CUDA error: " << cudaGetErrorString(status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  /* TODO: remove */ \
      cudaDeviceReset();                                                \
      throw lbann::lbann_exception("cudnn_wrapper: CUDA error");        \
    }                                                                   \
  }

namespace cudnn
{

  /** cuDNN manager class */
  class cudnn_manager
  {

  public:
    /** Constructor
     *  @param _comm         Pointer to LBANN communicator
     *  @param max_num_gpus  Maximum Number of available GPUs. If
     *                       negative, then use all available GPUs.
     */
    cudnn_manager(lbann::lbann_comm* _comm, int max_num_gpus = -1);

    /** Destructor */
    ~cudnn_manager();

    /** Print cuDNN version information to standard output */
    void print_version() const;

    /// Register a block of memory to pin
    void pin_ptr(void* ptr, size_t sz);
    /// Pin the memory block of a matrix
    void pin_memory_block(ElMat *mat);
    /// Unregister a block of pinnedmemory
    void unpin_ptr(void* ptr);
    /// Unregister all the memories registered to pin
    void unpin_ptrs(void);

  public:

    /** LBANN communicator */
    lbann::lbann_comm* comm;

    /** Number of GPUs for current MPI rank */
    int m_num_gpus;
    /** Number of available GPUs */
    int m_num_total_gpus;

    /** GPU memory allocator
     *  Faster than cudaMalloc/cudaFree since it uses a memory pool */
    cub::CachingDeviceAllocator* m_gpu_memory;

    /** GPUs for current MPI rank */
    std::vector<int> m_gpus;
    /** CUDA streams for current MPI rank */
    std::vector<cudaStream_t> m_streams;
    /** cuDNN handles for current MPI rank */
    std::vector<cudnnHandle_t> m_handles;
    /// pinned memory addresses
    std::map<void*, size_t> pinned_ptr;

  };


  /// cuDNN convolutional layer
  class cudnn_convolutional_layer
  {
  public:

    /// Constructor
    cudnn_convolutional_layer(int num_dims,
                              int src_channels,
                              int dst_channels,
                              const int* src_dims,
                              const int* filter_dims,
                              const int* conv_pads,
                              const int* conv_strides,
                              const uint mini_batch_size,
                              cudnn_manager* cudnn);
    
    /// Destructor
    ~cudnn_convolutional_layer();
    
    /// Setup convolutional layer
    void setup();

    /// Convolutional layer forward pass
    /** @todo Handle case where GPU can't hold entire mini-batch. */
    void forward(const Mat& src, const Mat& filter, const Mat& bias, Mat& dst);
    
    /// Convolutional layer backward pass
    /** @todo Handle case where GPU can't hold entire mini-batch. */
    void backward(const Mat& src, const Mat& filter, const Mat& grad_dst,
                  Mat& grad_filter, Mat& grad_bias, Mat& grad_src);

    /// Return the pointer to the associated cudnn_manager
    cudnn_manager* get_cudnn_manager(void) { return m_cudnn; }

  public:
      
    /// Number of dimensions
    const int m_num_dims;

    /// Input tensor size
    int m_src_size;
    /// Output tensor size
    int m_dst_size;
    /// Filter size
    int m_filter_size;

    /// Input tensor dimensions
    /** cuDNN's NCHW or NCDHW format */
    std::vector<int> m_src_dims;
    /// Output tensor dimensions
    /** cuDNN's NCHW or NCDHW format */
    std::vector<int> m_dst_dims;
    /// Filter dimensions
    /** cuDNN's KCHW or KCDHW format */
    std::vector<int> m_filter_dims;

    /// Convolution padding
    std::vector<int> m_conv_pads;
    /// Convolution strides
    std::vector<int> m_conv_strides;
  
  private:

    /// cuDNN manager
    cudnn_manager* m_cudnn;

    /// cuDNN datatype
    const cudnnDataType_t m_cudnn_data_type;

    /// Number of data samples per GPU
    int m_samples_per_gpu;

    /// Input tensor descriptor
    cudnnTensorDescriptor_t m_src_desc;
    /// Output tensor descriptor
    cudnnTensorDescriptor_t m_dst_desc;
    /// Filter descriptor
    cudnnFilterDescriptor_t m_filter_desc;
    /// Convolution descriptor
    cudnnConvolutionDescriptor_t m_conv_desc;

    /// Forward pass algorithm
    cudnnConvolutionFwdAlgo_t m_forward_algo;
    /// Forward pass algorithm work space size (in bytes)
    size_t m_forward_work_space_size;

    /// Backward pass filter algorithm
    /** Compute gradient w.r.t. filter. */
    cudnnConvolutionBwdFilterAlgo_t m_backward_filter_algo;
    /// Backward pass filter algorithm work space size (in bytes)
    /** Compute gradient w.r.t. filter. */
    size_t m_backward_filter_work_space_size;

    /// Backward pass data algorithm
    /** Compute gradient w.r.t. data, which is passed to previous layer. */
    cudnnConvolutionBwdDataAlgo_t m_backward_data_algo;
    /// Backward pass data algorithm work space size (in bytes)
    /** Compute gradient w.r.t. data, which is passed to previous layer. */
    size_t m_backward_data_work_space_size;

    /// Input tensor strides
    std::vector<int> m_src_strides;
    /// Output tensor strides
    std::vector<int> m_dst_strides;


    const uint m_mini_batch_size;

    std::vector<DataType*> d_src;
    std::vector<DataType*> d_filter;
    std::vector<DataType*> d_bias;
    std::vector<DataType*> d_dst;
    std::vector<DataType*> d_work_space;

    std::vector<DataType*> d_prev_error_signal;
    std::vector<DataType*> d_filter_gradient;
    std::vector<DataType*> d_error_signal;
    std::vector<DataType*> d_filter_work_space;
    std::vector<DataType*> d_data_work_space;

    /// Allocate memory on GPUs once and for all for the entire execution
    void device_allocate(void);
    /// Deallocate all the memory blocked on GPUs
    void device_deallocate(void);
    /// Allocate memory on GPUs for the forward path
    void device_allocate_for_forward(void);
    /// Allocate memory on GPUs for the backward path
    void device_allocate_for_backward(void);
    /// Deallocate memory on GPUs for the forward path
    void device_deallocate_for_forward(void);
    /// Deallocate memory on GPUs for the backward path
    void device_deallocate_for_backward(void);

    /// A temporary matrix used to collect data from each GPU during reduction
    Mat temp;

  };

  /// cuDNN pooling layer
  class cudnn_pooling_layer
  {

  public:

    /// Constructor
    cudnn_pooling_layer(int num_dims,
                        int channels,
                        const int* src_dims,
                        pool_mode _pool_mode,
                        const int* pool_dims,
                        const int* pool_pads,
                        const int* pool_strides,
                        const uint mini_batch_size,
                        cudnn_manager* cudnn);
    
    /// Destructor
    ~cudnn_pooling_layer();
    
    /// Setup pooling layer
    void setup();

    /// Convolutional layer forward pass
    void forward(const Mat& src, Mat& dst);
    
    /// Convolutional layer backward pass
    void backward(const Mat& src, const Mat& dst,
                  const Mat& grad_dst, Mat& grad_src);

    /// Return the pointer to the associated cudnn_manager
    cudnn_manager* get_cudnn_manager(void) { return m_cudnn; }

  public:
      
    /// Number of dimensions
    const int m_num_dims;

    /// Input tensor size
    int m_src_size;
    /// Output tensor size
    int m_dst_size;

    /// Input tensor dimensions
    /** cuDNN's NCHW or NCDHW format */
    std::vector<int> m_src_dims;
    /// Output tensor dimensions
    /** cuDNN's NCHW or NCDHW format */
    std::vector<int> m_dst_dims;

    /// Pooling mode
    const cudnnPoolingMode_t m_pool_mode;

    /// Pooling dimensions
    /** HW or DHW format */
    std::vector<int> m_pool_dims;
    /// Pooling padding
    /** HW or DHW format */
    std::vector<int> m_pool_pads;
    /// Pooling strides
    /** HW or DHW format */
    std::vector<int> m_pool_strides;
  
  private:

    /// cuDNN manager
    cudnn_manager* m_cudnn;

    /// cuDNN datatype
    const cudnnDataType_t m_cudnn_data_type;

    /// Number of data samples per GPU
    int m_samples_per_gpu;

    /// Input tensor descriptor
    cudnnTensorDescriptor_t m_src_desc;
    /// Output tensor descriptor
    cudnnTensorDescriptor_t m_dst_desc;
    /// Pooling descriptor
    cudnnPoolingDescriptor_t m_pool_desc;

    /// Input tensor strides
    std::vector<int> m_src_strides;
    /// Output tensor strides
    std::vector<int> m_dst_strides;

    const uint m_mini_batch_size;

    std::vector<DataType*> d_src;
    std::vector<DataType*> d_dst;

    std::vector<DataType*> d_prev_error_signal;
    std::vector<DataType*> d_error_signal;

    /// Allocate memory on GPUs once and for all for the entire execution
    void device_allocate(void);
    /// Deallocate all the memory blocked on GPUs
    void device_deallocate(void);
    /// Allocate memory on GPUs for the forward path
    void device_allocate_for_forward(void);
    /// Allocate memory on GPUs for the backward path
    void device_allocate_for_backward(void);
    /// Deallocate memory on GPUs for the forward path
    void device_deallocate_for_forward(void);
    /// Deallocate memory on GPUs for the backward path
    void device_deallocate_for_backward(void);

  };

}

#else  // __LIB_CUDNN

namespace cudnn
{
  class cudnn_manager {};
  class cudnn_convolutional_layer {};
  class cudnn_pooling_layer {};
}

#endif  // __LIB_CUDNN

#endif // CUDNN_WRAPPER_HPP_INCLUDED
