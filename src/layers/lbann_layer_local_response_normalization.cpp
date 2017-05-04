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
// lbann_layer_local_response_normalization .hpp .cpp - LRN layer
////////////////////////////////////////////////////////////////////////////////

#include "lbann/layers/lbann_layer_local_response_normalization.hpp"
#include "lbann/utils/lbann_exception.hpp"

using namespace std;
using namespace El;
using namespace lbann;

local_response_normalization_layer::local_response_normalization_layer
(uint index,
 int num_dims,
 int num_channels,
 const int* dims,
 Int window_width,
 DataType lrn_alpha,
 DataType lrn_beta,
 DataType lrn_k,
 uint mini_batch_size,
 lbann_comm* comm,
 cudnn::cudnn_manager* cudnn)
  : Layer(data_layout::DATA_PARALLEL, index, comm, NULL, mini_batch_size, activation_type::ID, {}),
    m_window_width(window_width), m_lrn_alpha(lrn_alpha), m_lrn_beta(lrn_beta), m_lrn_k(lrn_k),
    m_num_dims(num_dims), m_num_channels(num_channels)
{
  m_type = layer_type::local_response_normalization;

  // Initialize data dimensions
  m_dims.resize(num_dims);
  NumNeurons = num_channels;
  for(int i=0; i<num_dims; ++i) {
    m_dims[i] = dims[i];
    NumNeurons *= dims[i];
  }

#ifdef __LIB_CUDNN

  // Initialize cuDNN objects
  m_tensor_desc = NULL;
  m_lrn_desc = NULL;

  // Initialize GPU memory if using GPU
  if(cudnn) {
    m_using_gpus = true;
    m_cudnn = cudnn;

    // Get number of GPUs
    const int num_gpus = m_cudnn->get_num_gpus();

    // Get number of columns per GPU
    const int num_processes = comm->get_procs_per_model();
    const int local_mini_batch_size = (mini_batch_size + num_processes - 1) / num_processes;
    m_mini_batch_size_per_gpu = (local_mini_batch_size + num_gpus - 1) / num_gpus;

  }
#endif // __LIB_CUDNN

}

local_response_normalization_layer::~local_response_normalization_layer()
{
#ifdef __LIB_CUDNN
  if(m_using_gpus) {

    // Destroy cuDNN objects
    if(m_tensor_desc)
      checkCUDNN(cudnnDestroyTensorDescriptor(m_tensor_desc));
    if(m_lrn_desc)
      checkCUDNN(cudnnDestroyLRNDescriptor(m_lrn_desc));

    // Deallocate GPU memory
    m_cudnn->deallocate_on_gpus(m_weighted_sum_d);
    m_cudnn->deallocate_on_gpus(m_activations_d);
    m_cudnn->deallocate_on_gpus(m_error_signal_d);
    if(!m_prev_layer_using_gpus)
      m_cudnn->deallocate_on_gpus(m_prev_activations_d);
    if(!m_next_layer_using_gpus)
      m_cudnn->deallocate_on_gpus(m_prev_error_signal_d);

  }
#endif // __LIB_CUDNN
}

void local_response_normalization_layer::setup(const int num_prev_neurons)
{
  Layer::setup(num_prev_neurons);

#ifdef __LIB_CUDNN
  // Setup cuDNN objects
  if(m_using_gpus) {
    setup_gpu();
  }
#endif // __LIB_CUDNN

#ifdef LBANN_DEBUG
  // Check if input dimensions are valid
  int num_inputs = m_num_channels;
  for(int i=0; i<m_num_dims; ++i)
    num_inputs *= m_dims[i];
  if(num_inputs != num_prev_neurons) {
    throw lbann_exception("lbann_layer_local_response_normalization: unexpected number of input neurons");
  }
#endif

  // Initialize matrices
  if(!m_using_gpus || !m_prev_layer_using_gpus) {
    Zeros(*m_prev_activations, m_num_prev_neurons, m_mini_batch_size);
    Zeros(*m_error_signal, m_num_prev_neurons, m_mini_batch_size);
  }
  if(!m_using_gpus || !m_next_layer_using_gpus) {
    Zeros(*m_activations, NumNeurons, m_mini_batch_size);
    Zeros(*m_prev_error_signal, NumNeurons, m_mini_batch_size);
  }
  if(!m_using_gpus) {
    Zeros(*m_weighted_sum, NumNeurons, m_mini_batch_size);
  }

}

void lbann::local_response_normalization_layer::setup_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_local_response_normalization: cuDNN not detected");
#else

  // Initialize descriptors
  checkCUDNN(cudnnCreateTensorDescriptor(&m_tensor_desc));
  checkCUDNN(cudnnCreateLRNDescriptor(&m_lrn_desc));

  // Set input tensor descriptor
  std::vector<int> dims(m_num_dims+2);
  dims[0] = m_mini_batch_size_per_gpu;
  dims[1] = m_num_channels;
  for(Int i=0; i<m_num_dims; ++i)
    dims[i+2] = m_dims[i];
  std::vector<int> strides(m_num_dims+2);
  strides[m_num_dims + 1] = 1;
  for(Int i=m_num_dims; i>=0; --i)
    strides[i] = strides[i+1] * dims[i+1];
  checkCUDNN(cudnnSetTensorNdDescriptor(m_tensor_desc,
                                        m_cudnn->get_cudnn_data_type(),
                                        m_num_dims+2,
                                        dims.data(),
                                        strides.data()));

  // Set local response normalization descriptor
  checkCUDNN(cudnnSetLRNDescriptor(m_lrn_desc,
                                   (unsigned int) m_window_width,
                                   (double) m_lrn_alpha,
                                   (double) m_lrn_beta,
                                   (double) m_lrn_k));

  // Allocate GPU memory
  m_cudnn->allocate_on_gpus(m_weighted_sum_d,
                            NumNeurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_activations_d,
                            NumNeurons,
                            m_mini_batch_size_per_gpu);
  m_cudnn->allocate_on_gpus(m_error_signal_d,
                            m_num_prev_neurons,
                            m_mini_batch_size_per_gpu);
  if(!m_prev_layer_using_gpus) {
    m_cudnn->allocate_on_gpus(m_prev_activations_d,
                              m_num_prev_neurons,
                              m_mini_batch_size_per_gpu);
  }
  if(!m_next_layer_using_gpus) {
    m_cudnn->allocate_on_gpus(m_prev_error_signal_d,
                              NumNeurons,
                              m_mini_batch_size_per_gpu);
  }

#endif // #ifndef __LIB_CUDNN
}

void lbann::local_response_normalization_layer::fp_linearity() {
  if(m_using_gpus) {
    fp_linearity_gpu();
  }
  else {
    fp_linearity_cpu();
  }
}

void lbann::local_response_normalization_layer::bp_linearity() {
  if(m_using_gpus) {
    bp_linearity_gpu();
  }
  else {
    bp_linearity_cpu();
  }
}

void lbann::local_response_normalization_layer::fp_linearity_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_local_response_normalization: cuDNN not detected");
#else

  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Perform local response normalization with each GPU
  const Int num_gpus = m_cudnn->get_num_gpus();
  for(Int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
    checkCUDNN(cudnnLRNCrossChannelForward(m_cudnn->get_handle(i),
                                           m_lrn_desc,
                                           CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                           &one,
                                           m_tensor_desc,
                                           m_prev_activations_d[i],
                                           &zero,
                                           m_tensor_desc,
                                           m_weighted_sum_d[i]));
  }

  // Copy result to output matrix
  m_cudnn->copy_on_gpus(m_activations_d,
                        m_weighted_sum_d,
                        NumNeurons,
                        m_mini_batch_size_per_gpu);

#endif // #ifndef __LIB_CUDNN
}

void lbann::local_response_normalization_layer::fp_linearity_cpu() {

  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  Mat& weighted_sum_local = m_weighted_sum_v->Matrix();
  Mat& activations_local = m_activations_v->Matrix();

  // Input and output entries are divided amongst channels
  const Int num_per_channel = NumNeurons / m_num_channels;

  ////////////////////////////////////////////////////////////////
  // activations(i) = prev_activations(i) / scale_factor(i) ^ beta
  // scale_factor(i)
  //   = k + alpha / window_width * sum( prev_activations(j) ^ 2 )
  // Note: The sum is over entries in the normalization window.
  ////////////////////////////////////////////////////////////////
  
  // Iterate through data samples in mini-batch
#pragma omp parallel for collapse(2)
  for(Int sample = 0; sample < prev_activations_local.Width(); ++sample) {
    // Iterate through positions in sample
    for(Int pos = 0; pos < num_per_channel; ++pos) {

      // Initialize normalization window
      Int window_start = - m_window_width / 2;
      Int window_end = m_window_width / 2;
      DataType window_sum = 0;
      for(Int c = Max(window_start, 0);
          c <= Min(window_end, m_num_channels-1);
          ++c) {
        const DataType x
          = prev_activations_local.Get(pos + num_per_channel*c, sample);
        window_sum += x * x;
      }

      // Iterate through channels at current position
      for(Int channel = 0; channel < m_num_channels; ++channel) {
        const Int index = pos + num_per_channel * channel;

        // Apply local response normalization to current entry
        const DataType input_entry = prev_activations_local.Get(index, sample);
        const DataType scale_factor = m_lrn_k + m_lrn_alpha / m_window_width * window_sum;
        const DataType output_entry = input_entry * Pow(scale_factor, -m_lrn_beta);
        weighted_sum_local.Set(index, sample, output_entry);
        
        // Shift normalization window by one entry
        if(window_start >= 0) {
          const Int i = pos + num_per_channel*window_start;
          const DataType x = prev_activations_local.Get(i, sample);
          window_sum -= x * x;
        }
        ++window_start;
        ++window_end;
        if(window_end < m_num_channels) {
          const Int i = pos + num_per_channel*window_end;
          const DataType x = prev_activations_local.Get(i, sample);
          window_sum += x * x;
        }

      }

    }
    
  }

  // weighted_sum and output are identical after fp linearity step
  Copy(weighted_sum_local, activations_local);

}

void lbann::local_response_normalization_layer::bp_linearity_gpu() {
#ifndef __LIB_CUDNN
  throw lbann_exception("lbann_layer_local_response_normalization: cuDNN not detected");
#else
  
  // Useful constants
  const DataType one = 1;
  const DataType zero = 0;

  // Get number of GPUs
  const Int num_gpus = m_cudnn->get_num_gpus();

  // Perform back propagation on each GPU
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
    checkCUDNN(cudnnLRNCrossChannelBackward(m_cudnn->get_handle(i),
                                            m_lrn_desc,
                                            CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                            &one,
                                            m_tensor_desc,
                                            m_weighted_sum_d[i],
                                            m_tensor_desc,
                                            m_prev_error_signal_d[i],
                                            m_tensor_desc,
                                            m_prev_activations_d[i],
                                            &zero,
                                            m_tensor_desc,
                                            m_error_signal_d[i]));
  }

#endif // #ifndef __LIB_CUDNN
}

void lbann::local_response_normalization_layer::bp_linearity_cpu() {

  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  const Mat& activations_local = m_activations_v->LockedMatrix();
  const Mat& prev_error_signal_local = m_prev_error_signal_v->LockedMatrix();
  Mat& error_signal_local = m_error_signal_v->Matrix();

  // Initialize error signal to zero
  Zero(error_signal_local);

  // Input and output entries are divided amongst channels
  const Int num_per_channel = NumNeurons / m_num_channels;

  ////////////////////////////////////////////////////////////////
  // error_signal(i)
  //   = prev_error_signal(i) / scale_factor(i) ^ beta 
  //     - 2 * alpha * beta / window_width * prev_activations(i) 
  //       * sum( prev_error_signal(j) * activations(j)
  //              / scale_factor(j) )
  // Note: See comments in fp_linearity_cpu for a definition of
  //   scale_factor. The sum is over entries in the normalization
  //   window.
  ////////////////////////////////////////////////////////////////
  
  // Iterate through data samples in mini-batch
#pragma omp parallel for collapse(2)
  for(Int sample = 0; sample < prev_activations_local.Width(); ++sample) {
    // Iterate through positions in sample
    for(Int pos = 0; pos < num_per_channel; ++pos) {

      // Initialize normalization window
      Int window_start = - m_window_width / 2;
      Int window_end = m_window_width / 2;
      DataType window_sum = 0;
      for(Int c = Max(window_start, 0);
          c <= Min(window_end, m_num_channels-1);
          ++c) {
        const DataType x
          = prev_activations_local.Get(pos + num_per_channel*c, sample);
        window_sum += x * x;
      }

      // Iterate through channels at current position
      DataType error_signal_update;
      for(Int channel = 0; channel < m_num_channels; ++channel) {
        const Int index = pos + num_per_channel * channel;

        // Get data for current entry
        const DataType activations_entry = activations_local.Get(index, sample);
        const DataType prev_error_signal_entry = prev_error_signal_local.Get(index, sample);
        const DataType scale_factor = m_lrn_k + m_lrn_alpha / m_window_width * window_sum;

        // Update current error signal entry
        error_signal_update = prev_error_signal_entry * Pow(scale_factor, -m_lrn_beta);
        error_signal_local.Update(index, sample, error_signal_update);

        // Update error signal entries in normalization window
        for(Int c = Max(window_start, 0);
            c <= Min(window_end, m_num_channels-1);
            ++c) {
          const Int i = pos + num_per_channel * c;
          const DataType prev_activations_entry = prev_activations_local.Get(i, sample);
          error_signal_update
            = (-2 * m_lrn_alpha * m_lrn_beta / m_window_width * prev_activations_entry
               * prev_error_signal_entry * activations_entry / scale_factor);
          error_signal_local.Update(i, sample, error_signal_update);
        }
        
        // Shift normalization window by one entry
        if(window_start >= 0) {
          const Int i = pos + num_per_channel*window_start;
          const DataType x = prev_activations_local.Get(i, sample);
          window_sum -= x * x;
        }
        ++window_start;
        ++window_end;
        if(window_end < m_num_channels) {
          const Int i = pos + num_per_channel*window_end;
          const DataType x = prev_activations_local.Get(i, sample);
          window_sum += x * x;
        }

      }

    }
    
  }

}

bool local_response_normalization_layer::update()
{
  double start = get_time();
  Layer::update();
  update_time += get_time() - start;
  return true;
}
