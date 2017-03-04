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
#include "lbann/utils/lbann_exception.hpp"

using namespace std;
using namespace El;
using namespace lbann;

#ifdef __LIB_CUDNN
/// Get cuDNN pooling mode
cudnnPoolingMode_t get_cudnn_pool_mode(const pool_mode mode)
{
  switch(mode) {
  case pool_mode::max:
    return CUDNN_POOLING_MAX;
  case pool_mode::average:
    return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  case pool_mode::average_no_pad:
    return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  default:
    throw lbann::lbann_exception("cudnn_wrapper: invalid pooling mode");
  }
}
#endif // #ifdef __LIB_CUDNN

pooling_layer::pooling_layer(uint index,
                             int num_dims,
                             int num_channels,
                             const int* input_dims,
                             const int* pool_dims,
                             const int* pool_pads,
                             const int* pool_strides,
                             pool_mode _pool_mode,
                             uint mini_batch_size,
                             lbann_comm* comm,
                             cudnn::cudnn_manager* cudnn)
  : Layer(index, comm, NULL, mini_batch_size, activation_type::ID, {}),
    m_pool_mode(_pool_mode),
    m_num_dims(num_dims), m_num_channels(num_channels),
    m_cudnn(cudnn)
{
  m_type = layer_type::pooling;

  // Initialize input dimensions and pooling parameters
  m_input_dims.resize(num_dims);
  m_pool_dims.resize(num_dims);
  m_pool_pads.resize(num_dims);
  m_pool_strides.resize(num_dims);
  for(int i=0; i<num_dims; ++i) {
    m_input_dims[i] = input_dims[i];
    m_pool_dims[i] = pool_dims[i];
    m_pool_pads[i] = pool_pads[i];
    m_pool_strides[i] = pool_strides[i];
  }

  // Calculate output dimensions
  m_output_dims.resize(num_dims);
  NumNeurons = num_channels;
  for(int i=0; i<num_dims; ++i) {
    m_output_dims[i] = input_dims[i]+2*pool_pads[i]-pool_dims[i]+1;
    m_output_dims[i] = (m_output_dims[i]+pool_strides[i]-1)/pool_strides[i];
    NumNeurons *= m_output_dims[i];
  }

  // Matrices should be in Star,VC distribution
  delete m_weighted_sum;
  delete m_prev_activations;
  delete m_activations;
  delete m_prev_error_signal;
  delete m_error_signal;
  m_weighted_sum        = new StarVCMat(comm->get_model_grid());
  m_prev_activations    = new StarVCMat(comm->get_model_grid());
  m_activations         = new StarVCMat(comm->get_model_grid());
  m_prev_error_signal   = new StarVCMat(comm->get_model_grid());
  m_error_signal        = new StarVCMat(comm->get_model_grid());

  // Matrix views should be in Star,VC distributions
  delete m_weighted_sum_v;
  delete m_prev_activations_v;
  delete m_activations_v;
  delete m_prev_error_signal_v;
  delete m_error_signal_v;
  m_weighted_sum_v      = new StarVCMat(comm->get_model_grid());
  m_prev_activations_v  = new StarVCMat(comm->get_model_grid());
  m_activations_v       = new StarVCMat(comm->get_model_grid());
  m_prev_error_signal_v = new StarVCMat(comm->get_model_grid());
  m_error_signal_v      = new StarVCMat(comm->get_model_grid());

#ifdef __LIB_CUDNN

  // Initialize cuDNN objects
  m_input_desc = NULL;
  m_output_desc = NULL;
  m_pooling_desc = NULL;

  // Initialize GPU memory if using GPU
  if(cudnn) {
    m_using_gpu = true;

    // Get number of GPUs
    const int num_gpus = m_cudnn->get_num_gpus();

    // Get number of columns per GPU
    const int num_processes = m_cudnn->comm->get_procs_per_model();
    const int local_mini_batch_size = (mini_batch_size + num_processes - 1) / num_processes;
    m_mini_batch_size_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;

    // Initialize GPU memory pointers
    m_prev_activations_d.assign(num_gpus, NULL);
    m_activations_d.assign(num_gpus, NULL);
    m_prev_error_signal_d.assign(num_gpus, NULL);
    m_error_signal_d.assign(num_gpus, NULL);

  }
  is_pinned_fwd = false;
  is_pinned_bwd = false;
#endif // __LIB_CUDNN

}

pooling_layer::~pooling_layer()
{
#ifdef __LIB_CUDNN
  if(m_using_gpu) {
    if(m_input_desc)
      checkCUDNN(cudnnDestroyTensorDescriptor(m_input_desc));
    if(m_output_desc)
      checkCUDNN(cudnnDestroyTensorDescriptor(m_output_desc));
    if(m_pooling_desc)
      checkCUDNN(cudnnDestroyPoolingDescriptor(m_pooling_desc));
    for(int i=0; i<m_cudnn->get_num_gpus(); ++i) {
      const int gpu = m_cudnn->m_gpus[i];
      cub::CachingDeviceAllocator& gpu_memory = *(m_cudnn->m_gpu_memory);
      if(m_prev_activations_d.size() > i && m_prev_activations_d[i] != NULL )
        checkCUDA(gpu_memory.DeviceFree(gpu, m_prev_activations_d[i]));
      if(m_activations_d.size() > i && m_activations_d[i] != NULL )
        checkCUDA(gpu_memory.DeviceFree(gpu, m_activations_d[i]));
      if(m_prev_error_signal_d.size() > i && m_prev_error_signal_d[i] != NULL )
        checkCUDA(gpu_memory.DeviceFree(gpu, m_prev_error_signal_d[i]));
      if(m_error_signal_d.size() > i && m_error_signal_d[i] != NULL )
        checkCUDA(gpu_memory.DeviceFree(gpu, m_error_signal_d[i]));
    }
  }
#endif // __LIB_CUDNN
}

void pooling_layer::setup(const int num_prev_neurons)
{
  Layer::setup(num_prev_neurons);

#ifdef __LIB_CUDNN
  // Setup cuDNN objects
  if(m_using_gpu) {
    setup_gpu();
  }
#endif // __LIB_CUDNN

  // Check if input dimensions are valid
  int num_inputs = m_num_channels;
  for(int i=0; i<m_num_dims; ++i)
    num_inputs *= m_input_dims[i];
  if(num_inputs != num_prev_neurons) {
    throw lbann_exception("lbann_layer_pooling: unexpected number of input neurons");
  }

  // Initialize matrices
  Zeros(*m_prev_activations, num_prev_neurons, m_mini_batch_size);
  Zeros(*m_weighted_sum, NumNeurons, m_mini_batch_size);
  Zeros(*m_prev_error_signal, NumNeurons, m_mini_batch_size);
  Zeros(*m_error_signal, num_prev_neurons, m_mini_batch_size);
  Zeros(*m_activations, NumNeurons, m_mini_batch_size);

}

void lbann::pooling_layer::setup_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_pooling: cuDNN not detected");
#else

  // Initialize descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&m_input_desc));
  checkCUDNN(cudnnCreateTensorDescriptor(&m_output_desc));
  checkCUDNN(cudnnCreatePoolingDescriptor(&m_pooling_desc));

  // Set input tensor descriptor
  std::vector<int> input_dims(m_num_dims+2);
  input_dims[0] = m_mini_batch_size_per_gpu;
  input_dims[1] = m_num_channels;
  for(Int i=0; i<m_num_dims; ++i)
    input_dims[i+2] = m_input_dims[i];
  std::vector<int> input_strides(m_num_dims+2);
  input_strides[m_num_dims + 1]  = 1;
  for(Int i=m_num_dims; i>=0; --i)
    input_strides[i] = input_strides[i+1] * input_dims[i+1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_input_desc,
                                        m_cudnn->get_cudnn_data_type(),
                                        m_num_dims+2,
                                        input_dims.data(),
                                        input_strides.data()));

  // Set pooling descriptor
  std::vector<int> pool_dims(m_num_dims);
  std::vector<int> pool_pads(m_num_dims);
  std::vector<int> pool_strides(m_num_dims);
  for(Int i=0; i<m_num_dims; ++i) {
    pool_dims[i] = m_pool_dims[i];
    pool_pads[i] = m_pool_pads[i];  
    pool_strides[i] = m_pool_strides[i];
  }
  checkCUDNN(cudnnSetPoolingNdDescriptor(m_pooling_desc,
                                         get_cudnn_pool_mode(m_pool_mode),
                                         CUDNN_PROPAGATE_NAN,
                                         m_num_dims,
                                         pool_dims.data(),
                                         pool_pads.data(),
                                         pool_strides.data()));

  // Set output tensor descriptor
  std::vector<int> output_dims(m_num_dims+2);
#ifdef LBANN_DEBUG
  checkCUDNN(cudnnGetPoolingNdForwardOutputDim(m_pooling_desc,
                                               m_input_desc,
                                               m_num_dims+2,
                                               output_dims.data()));
  if(output_dims[0] != m_mini_batch_size_per_gpu)
    throw lbann_exception("lbann_layer_pooling: invalid output dimensions");
  if(output_dims[1] != m_num_channels)
    throw lbann_exception("lbann_layer_pooling: invalid output dimensions");
  for(Int i=0; i<m_num_dims; ++i) {
    if(output_dims[i+2] != m_output_dims[i]) {
      throw lbann_exception("lbann_layer_pooling: invalid output dimensions");
    }
  }
#else
  output_dims[0] = m_mini_batch_size_per_gpu;
  output_dims[1] = m_num_channels;
  for(Int i=0; i<m_num_dims; ++i)
    output_dims[i+2] = m_output_dims[i];
#endif // #ifdef LBANN_DEBUG
  std::vector<int> output_strides(m_num_dims+2);
  output_strides[m_num_dims + 1]  = 1;
  for(Int i=m_num_dims; i>=0; --i)
    output_strides[i] = output_strides[i+1] * output_dims[i+1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_output_desc,
                                        m_cudnn->get_cudnn_data_type(),
                                        m_num_dims+2,
                                        output_dims.data(),
                                        output_strides.data()));

  // Allocate GPU memory
  const int num_gpus = m_cudnn->get_num_gpus();
#pragma omp parallel for
  for(Int i=0; i<num_gpus; ++i) {
    const Int gpu = m_cudnn->m_gpus[i];
    cudaStream_t& stream = m_cudnn->m_streams[i];
    cub::CachingDeviceAllocator& gpu_memory = *(m_cudnn->m_gpu_memory);
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &m_prev_activations_d[i],
                                        m_num_prev_neurons*m_mini_batch_size_per_gpu*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &m_activations_d[i],
                                        NumNeurons*m_mini_batch_size_per_gpu*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &m_prev_error_signal_d[i],
                                        NumNeurons*m_mini_batch_size_per_gpu*sizeof(DataType),
                                        stream));
    checkCUDA(gpu_memory.DeviceAllocate(gpu,
                                        (void**) &m_error_signal_d[i],
                                        m_num_prev_neurons*m_mini_batch_size_per_gpu*sizeof(DataType),
                                        stream));
  }

#endif // #ifndef __LIB_CUDNN
}

void lbann::pooling_layer::pin_memory_blocks_fwd(void)
{
  if (!m_using_gpu) {
    std::cout << "no offloading with pooling_layer " << get_index() << std::endl;
    return;
  }

#ifdef __LIB_CUDNN
  m_cudnn->pin_memory_block(m_prev_activations);
  m_cudnn->pin_memory_block(m_weighted_sum);
  m_cudnn->pin_memory_block(m_activations);

  is_pinned_fwd = true;
#endif
}

void lbann::pooling_layer::pin_memory_blocks_bwd(void)
{
  if (!m_using_gpu) {
    std::cout << "no offloading with pooling_layer " << get_index() << std::endl;
    return;
  }

#ifdef __LIB_CUDNN
  m_cudnn->pin_memory_block(m_prev_error_signal);
  //cudnn_mgr->pin_memory_block(m_error_signal);

  is_pinned_bwd = true;
#endif
}

void lbann::pooling_layer::fp_linearity() {
  if(m_using_gpu) {
    fp_linearity_gpu();
  }
  else {
    fp_linearity_cpu();
  }
}

void lbann::pooling_layer::bp_linearity() {
  if(m_using_gpu) {
    bp_linearity_gpu();
  }
  else {
    bp_linearity_cpu();
  }
}

void lbann::pooling_layer::fp_linearity_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_pooling: cuDNN not detected");
#else

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Get number of GPUs
  const Int num_gpus = m_cudnn->get_num_gpus();

  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  const Mat& weights_local = m_weights->LockedMatrix();
  Mat& weighted_sum_local = m_weighted_sum_v->Matrix();
  Mat& activations_local = m_activations_v->Matrix();

  // Perform pooling with each GPU
#pragma omp parallel for
  for(Int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    cudaStream_t& stream = m_cudnn->m_streams[i];
    cudnnHandle_t& handle = m_cudnn->m_handles[i];

    // Data samples assigned to GPU
    const Int local_mini_batch = m_prev_activations_v->LocalWidth();
    const Int first_pos = Min(i * m_mini_batch_size_per_gpu, local_mini_batch);
    const Int last_pos = Min((i+1) * m_mini_batch_size_per_gpu, local_mini_batch);
    if(first_pos >= last_pos) {
      continue;
    }

    // Transfer inputs to GPU
    checkCUDA(cudaMemcpy2DAsync(m_prev_activations_d[i],
                                m_num_prev_neurons*sizeof(DataType),
                                prev_activations_local.LockedBuffer(0,first_pos),
                                prev_activations_local.LDim()*sizeof(DataType),
                                m_num_prev_neurons*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyHostToDevice,
                                stream));

    // Perform pooling
    checkCUDNN(cudnnPoolingForward(handle,
                                   m_pooling_desc,
                                   &one,
                                   m_input_desc,
                                   m_prev_activations_d[i],
                                   &zero,
                                   m_output_desc,
                                   m_activations_d[i]));

    // Transfer outputs from GPU
    checkCUDA(cudaMemcpy2DAsync(weighted_sum_local.Buffer(0,first_pos),
                                weighted_sum_local.LDim()*sizeof(DataType),
                                m_activations_d[i],
                                NumNeurons*sizeof(DataType),
                                NumNeurons*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyDeviceToHost,
                                stream));
    checkCUDA(cudaMemcpy2DAsync(activations_local.Buffer(0,first_pos),
                                activations_local.LDim()*sizeof(DataType),
                                m_activations_d[i],
                                NumNeurons*sizeof(DataType),
                                NumNeurons*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyDeviceToHost,
                                stream));
    
  }

  // Synchronize GPUs
#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    checkCUDA(cudaStreamSynchronize(m_cudnn->m_streams[i]));
  }

#endif // #ifndef __LIB_CUDNN
}

void lbann::pooling_layer::fp_linearity_cpu() {
  
  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  Mat& weighted_sum_local = m_weighted_sum_v->Matrix();
  Mat& activations_local = m_activations_v->Matrix();

  // Throw exception if pooling mode is not max pooling
  if(m_pool_mode != pool_mode::max) {
    throw lbann_exception("lbann_layer_pooling: CPU pooling layer only implements max pooling");
  }

  // Iterate through data samples in mini-batch
  for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {
    const Mat input_sample = prev_activations_local(ALL, IR(sample));
    Mat output_sample = weighted_sum_local(ALL, IR(sample));

    // Iterate through channels
    for(int channel = 0; channel < m_num_channels; ++channel) {
      const int input_channel_size
        = input_sample.Height() / m_num_channels;
      const int output_channel_size = NumNeurons / m_num_channels;
      const Mat input_channel
        = input_sample(IR(channel*input_channel_size,
                          (channel+1)*input_channel_size),
                       ALL);
      Mat output_channel
        = output_sample(IR(channel*output_channel_size,
                           (channel+1)*output_channel_size),
                        ALL);

      // Iterate through pool offsets
      // Note: each offset corresponds to an output entry
      int output_pos = 0;
      std::vector<int> pool_offset(m_num_dims);
      for(int d = 0; d < m_num_dims; ++d) {
        pool_offset[d] = -m_pool_pads[d];
      }
      while(pool_offset[0] + m_pool_dims[0] <= m_input_dims[0] + m_pool_pads[0]) {

        // Iterate through pool entries and find maximum
        std::vector<int> pool_pos(m_num_dims, 0);
        DataType max_value = -INFINITY;
        while(pool_pos[0] < m_pool_dims[0]) {

          // Get position of pool entry
          int input_pos = 0;
          bool valid_pos = true;
          for(int d = 0; d < m_num_dims; ++d) {
            if(pool_offset[d] + pool_pos[d] < 0
               || pool_offset[d] + pool_pos[d] >= m_input_dims[d]) {
              valid_pos = false;
              break;
            }
            input_pos *= m_input_dims[d];
            input_pos += pool_offset[d] + pool_pos[d];
          }

          // Check if pool entry is larger than previous
          DataType value = valid_pos ? input_channel.Get(input_pos, 0) : DataType(0);
          max_value = value > max_value ? value : max_value;

          // Move to next pool entry
          ++pool_pos[m_num_dims-1];
          for(int d = m_num_dims - 1; d > 0; --d) {
            if(pool_pos[d] >= m_pool_dims[d]) {
              pool_pos[d] = 0;
              ++pool_pos[d-1];
            }
          }

        }

        // Set output entry
        output_channel.Set(output_pos, 0, max_value);

        // Move to next output entry
        ++output_pos;

        // Move to next pool offset
        pool_offset[m_num_dims-1] += m_pool_strides[m_num_dims-1];
        for(int d = m_num_dims - 1; d > 0; --d) {
          if(pool_offset[d] + m_pool_dims[d] > m_input_dims[d] + m_pool_pads[d]) {
            pool_offset[d] = -m_pool_pads[d];
            pool_offset[d-1] += m_pool_strides[d-1];
          }
            
        }

      }

    }

  }

  // weighted_sum and output are identical after fp linearity step
  Copy(weighted_sum_local, activations_local);

}

void lbann::pooling_layer::bp_linearity_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_pooling: cuDNN not detected");
#else
  
  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Get number of GPUs
  const Int num_gpus = m_cudnn->get_num_gpus();

  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  const Mat& activations_local = m_activations_v->LockedMatrix();
  const Mat& prev_error_signal_local = m_prev_error_signal_v->LockedMatrix();
  Mat& weights_gradient_local = m_weights_gradient->Matrix();
  Mat& error_signal_local = m_error_signal_v->Matrix();

  // Perform back propagation on each GPU
#pragma omp parallel for
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    cudaStream_t& stream = m_cudnn->m_streams[i];
    cudnnHandle_t& handle = m_cudnn->m_handles[i];

    // Data samples assigned to GPU
    const Int local_mini_batch = prev_activations_local.Width();
    const Int first_pos = Min(i * m_mini_batch_size_per_gpu, local_mini_batch);
    const Int last_pos = Min((i+1) * m_mini_batch_size_per_gpu, local_mini_batch);
    if(first_pos >= last_pos) {
      continue;
    }

    // Transfer inputs and error signal to GPU
    checkCUDA(cudaMemcpy2DAsync(m_prev_activations_d[i],
                                m_num_prev_neurons*sizeof(DataType),
                                prev_activations_local.LockedBuffer(0,first_pos),
                                prev_activations_local.LDim()*sizeof(DataType),
                                m_num_prev_neurons*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyHostToDevice,
                                stream));
    checkCUDA(cudaMemcpy2DAsync(m_activations_d[i],
                                NumNeurons*sizeof(DataType),
                                activations_local.LockedBuffer(0,first_pos),
                                activations_local.LDim()*sizeof(DataType),
                                NumNeurons*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyHostToDevice,
                                stream));
    checkCUDA(cudaMemcpy2DAsync(m_prev_error_signal_d[i],
                                NumNeurons*sizeof(DataType),
                                prev_error_signal_local.LockedBuffer(0,first_pos),
                                prev_error_signal_local.LDim()*sizeof(DataType),
                                NumNeurons*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyHostToDevice,
                                stream));
    
    // Compute error signal to "next" layer
    checkCUDNN(cudnnPoolingBackward(handle,
                                    m_pooling_desc,
                                    &one,
                                    m_output_desc,
                                    m_activations_d[i],
                                    m_output_desc,
                                    m_prev_error_signal_d[i],
                                    m_input_desc,
                                    m_prev_activations_d[i],
                                    &zero,
                                    m_input_desc,
                                    m_error_signal_d[i]));

    // Transfer data from GPU
    checkCUDA(cudaMemcpy2DAsync(error_signal_local.Buffer(0,first_pos),
                                error_signal_local.LDim()*sizeof(DataType),
                                m_error_signal_d[i],
                                m_num_prev_neurons*sizeof(DataType),
                                m_num_prev_neurons*sizeof(DataType),
                                last_pos - first_pos,
                                cudaMemcpyDeviceToHost,
                                stream));

  }

  // Synchronize CUDA streams
  for(Int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->m_gpus[i]));
    checkCUDA(cudaStreamSynchronize(m_cudnn->m_streams[i]));
  }

#endif // #ifndef __LIB_CUDNN
}

void lbann::pooling_layer::bp_linearity_cpu() {

  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  const Mat& activations_local = m_activations_v->LockedMatrix();
  const Mat& prev_error_signal_local = m_prev_error_signal_v->LockedMatrix();
  Mat& error_signal_local = m_error_signal_v->Matrix();

  // Throw exception if pooling mode is not max pooling
  if(m_pool_mode != pool_mode::max) {
    throw lbann_exception("lbann_layer_pooling: CPU pooling layer only implements max pooling");
  }

  // Iterate through data samples in mini-batch
  for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {
    const Mat input_sample = prev_activations_local(ALL, IR(sample));
    const Mat prev_error_signal_sample = prev_error_signal_local(ALL, IR(sample));
    Mat error_signal_sample = error_signal_local(ALL, IR(sample));

    // Iterate through channels
    for(int channel = 0; channel < m_num_channels; ++channel) {
      const int input_channel_size = input_sample.Height() / m_num_channels;
      const int output_channel_size = NumNeurons / m_num_channels;
      const Mat input_channel
        = input_sample(IR(channel*input_channel_size,
                          (channel+1)*input_channel_size),
                       ALL);
      const Mat prev_error_signal_channel
        = prev_error_signal_sample(IR(channel*output_channel_size,
                                      (channel+1)*output_channel_size),
                                   ALL);
      Mat error_signal_channel
        = error_signal_sample(IR(channel*input_channel_size,
                                 (channel+1)*input_channel_size),
                              ALL);

      // Iterate through pool offsets
      // Note: each offset corresponds to an output entry
      int output_pos = 0;
      std::vector<int> pool_offset(m_num_dims);
      for(int d = 0; d < m_num_dims; ++d) {
        pool_offset[d] = -m_pool_pads[d];
      }
      while(pool_offset[0] + m_pool_dims[0] <= m_input_dims[0] + m_pool_pads[0]) {

        // Iterate through pool entries and find maximum
        std::vector<int> pool_pos(m_num_dims, 0);
        int max_input_pos = -1;
        DataType max_value = -INFINITY;
        while(pool_pos[0] < m_pool_dims[0]) {

          // Get position of pool entry
          int input_pos = 0;
          bool valid_pos = true;
          for(int d = 0; d < m_num_dims; ++d) {
            if(pool_offset[d] + pool_pos[d] < 0
               || pool_offset[d] + pool_pos[d] >= m_input_dims[d]) {
              valid_pos = false;
              break;
            }
            input_pos *= m_input_dims[d];
            input_pos += pool_offset[d] + pool_pos[d];
          }

          // Check if pool entry is larger than previous
          DataType value = valid_pos ? input_channel.Get(input_pos, 0) : 0.0;
          if(value > max_value) {
            if(valid_pos) {
              max_value = value;
              max_input_pos = input_pos;
            }
            else {
              max_input_pos = -1;
            }
          }

          // Move to next pool entry
          ++pool_pos[m_num_dims-1];
          for(int d = m_num_dims - 1; d > 0; --d) {
            if(pool_pos[d] >= m_pool_dims[d]) {
              pool_pos[d] = 0;
              ++pool_pos[d-1];
            }
          }

        }

        // Propagate error signal
        if(max_input_pos >= 0) {
          error_signal_channel.Set(max_input_pos, 0,
                                   prev_error_signal_channel.Get(output_pos,
                                                                 0));
        }

        // Move to next output entry
        ++output_pos;

        // Move to next pool offset
        pool_offset[m_num_dims-1] += m_pool_strides[m_num_dims-1];
        for(int d = m_num_dims - 1; d > 0; --d) {
          if(pool_offset[d] + m_pool_dims[d] > m_input_dims[d] + m_pool_pads[d]) {
            pool_offset[d] = -m_pool_pads[d];
            pool_offset[d-1] += m_pool_strides[d-1];
          }
            
        }

      }

    }

  }

}

bool pooling_layer::update()
{
  return true;
}
