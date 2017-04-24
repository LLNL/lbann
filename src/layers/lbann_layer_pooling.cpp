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
  : Layer(data_layout::DATA_PARALLEL, index, comm, NULL, mini_batch_size, activation_type::ID, {}),
    m_pool_mode(_pool_mode),
    m_num_dims(num_dims), m_num_channels(num_channels)
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

#ifdef __LIB_CUDNN

  // Initialize cuDNN objects
  m_input_desc = NULL;
  m_output_desc = NULL;
  m_pooling_desc = NULL;

  to_pin_fwd = false;
  to_pin_bwd = false;
  is_pinned_fwd = false;
  is_pinned_bwd = false;

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

    // Default behavior set to pin memory blocks used by cuDNN
    pin_mem();

  }
#endif // __LIB_CUDNN

}

pooling_layer::~pooling_layer()
{
#ifdef __LIB_CUDNN
  if(m_using_gpus) {

    // Destroy cuDNN objects
    if(m_input_desc)
      checkCUDNN(cudnnDestroyTensorDescriptor(m_input_desc));
    if(m_output_desc)
      checkCUDNN(cudnnDestroyTensorDescriptor(m_output_desc));
    if(m_pooling_desc)
      checkCUDNN(cudnnDestroyPoolingDescriptor(m_pooling_desc));

    // Unpin pinned memory
    unpin_mem();

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

void pooling_layer::setup(const int num_prev_neurons)
{
  Layer::setup(num_prev_neurons);

#ifdef __LIB_CUDNN
  // Setup cuDNN objects
  if(m_using_gpus) {
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

/**
 * \brief Set to pin the memory blocks used by cudnn.
 * \details The actual pinning occurs at the beginning of next fp_linearity() call.
 *          No effect when cudnn is not employed.
 */
// TODO: JY - Eventually, this needs to move up to the parent class
// in case that there are more gpu wrapper classes coming to existences
void lbann::pooling_layer::pin_mem(void) {
#ifdef __LIB_CUDNN
  to_pin_fwd = true;
  to_pin_bwd = true;
#endif
}

/**
 * \brief unpin the memory blocks pinned for cudnn
 * \details The effect is immediate.
 */
// TODO: JY - Eventually, this needs to move up to the parent class
void lbann::pooling_layer::unpin_mem(void) {
#ifdef __LIB_CUDNN
  to_pin_fwd = false;
  to_pin_bwd = false;
  unpin_memory_blocks_fwd();
  unpin_memory_blocks_bwd();
#endif
}

// TODO: JY - Eventually, this needs to a virtual member function of the parent class
void lbann::pooling_layer::pin_memory_blocks_fwd(void)
{
#ifdef __LIB_CUDNN
  size_t total_size = 0u;
  if(!m_prev_layer_using_gpus) {
    total_size += m_cudnn->pin_memory_block(m_prev_activations);
  }
  if(!m_next_layer_using_gpus)
    total_size += m_cudnn->pin_memory_block(m_activations);
  //std::cout << total_size << " bytes pinned by pooling layer " 
  //          << get_index() << " forward " << std::endl;

  is_pinned_fwd = true;
#endif
}

void lbann::pooling_layer::pin_memory_blocks_bwd(void)
{
#ifdef __LIB_CUDNN
  size_t total_size = 0u;
  if(!m_next_layer_using_gpus) {
    total_size += m_cudnn->pin_memory_block(m_prev_error_signal);
  }
  if(!m_prev_layer_using_gpus)
    total_size += m_cudnn->pin_memory_block(m_error_signal);
  
  //m_cudnn->pin_memory_block(m_error_signal);
  //std::cout << total_size << " bytes pinned by pooling layer " 
  //          << get_index() << " backward " << std::endl;

  is_pinned_bwd = true;
#endif
}

void lbann::pooling_layer::unpin_memory_blocks_fwd(void)
{
#ifdef __LIB_CUDNN
  if(!m_prev_layer_using_gpus)
    m_cudnn->unpin_memory_block(m_prev_activations);
  if(!m_next_layer_using_gpus)
    m_cudnn->unpin_memory_block(m_activations);

  is_pinned_fwd = false;
#endif
}

void lbann::pooling_layer::unpin_memory_blocks_bwd(void)
{
#ifdef __LIB_CUDNN
  if(!m_next_layer_using_gpus)
    m_cudnn->unpin_memory_block(m_prev_error_signal);
  if(!m_prev_layer_using_gpus)
    m_cudnn->unpin_memory_block(m_error_signal);

  is_pinned_bwd = false;
#endif
}

void lbann::pooling_layer::forwardProp() {

  // Perform forward propagation
  Layer::forwardProp();

#ifdef __LIB_CUDNN
  // Pin memory blocks at the first step
  if(to_pin_fwd && !is_pinned_fwd)
    pin_memory_blocks_fwd();
#endif // #ifdef __LIB_CUDNN

}

void lbann::pooling_layer::backProp() {

  // Perform backward propagation
  Layer::backProp();

#ifdef __LIB_CUDNN
  // Pin memory blocks at the first step
  if(to_pin_bwd && !is_pinned_bwd)
    pin_memory_blocks_bwd();
#endif // #ifdef __LIB_CUDNN

}

void lbann::pooling_layer::fp_linearity() {
  if(m_using_gpus) {
    fp_linearity_gpu();
  }
  else {
    fp_linearity_cpu();
  }
}

void lbann::pooling_layer::bp_linearity() {
  if(m_using_gpus) {
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

  // Perform pooling with each GPU
  const Int num_gpus = m_cudnn->get_num_gpus();
  for(Int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
    checkCUDNN(cudnnPoolingForward(m_cudnn->get_handle(i),
                                   m_pooling_desc,
                                   &one,
                                   m_input_desc,
                                   m_prev_activations_d[i],
                                   &zero,
                                   m_output_desc,
                                   m_weighted_sum_d[i]));
  }

  // Copy result to output matrix
  m_cudnn->copy_on_gpus(m_activations_d,
                        m_weighted_sum_d,
                        NumNeurons,
                        m_mini_batch_size_per_gpu);

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

  // Input, output, and filter entries are divided amongst channels
  const Int num_per_output_channel = NumNeurons / m_num_channels;
  const Int num_per_input_channel = m_num_prev_neurons / m_num_channels;

  // Iterate through data samples in mini-batch
#pragma omp parallel for
  for(Int sample = 0; sample < prev_activations_local.Width(); ++sample) {

    // Iterate through channels
    for(Int channel = 0; channel < m_num_channels; ++channel) {

      // Iterate through output entries in current channel
      // Note: each output entry corresponds to a different offset of
      //   the pool kernel
      std::vector<Int> pool_offsets(m_num_dims);
      for(int d = 0; d < m_num_dims; ++d) {
        pool_offsets[d] = -m_pool_pads[d];
      }
      const Int start_output_index = channel*num_per_output_channel;
      const Int end_output_index = (channel+1)*num_per_output_channel;
      for(Int output_index = start_output_index;
          output_index < end_output_index;
          ++output_index) {

        // Iterate through pool entries and find maximum
        std::vector<Int> pool_pos(m_num_dims, 0);
        DataType max_value = -INFINITY;
        while(pool_pos[0] < m_pool_dims[0]) {

          // Get input entry corresponding to pool entry
          Int input_index = 0;
          bool valid_input_entry = true;
          for(Int d = 0; d < m_num_dims; ++d) {
            if(pool_offsets[d] + pool_pos[d] < 0
               || pool_offsets[d] + pool_pos[d] >= m_input_dims[d]) {
              valid_input_entry = false;
              break;
            }
            input_index *= m_input_dims[d];
            input_index += pool_offsets[d] + pool_pos[d];
          }
          input_index += channel * num_per_input_channel;

          // Check if pool entry is largest encountered
          if(valid_input_entry) {
            const DataType value = prev_activations_local.Get(input_index, sample);
            max_value = Max(value, max_value);
          }

          // Move to next pool entry
          ++pool_pos[m_num_dims-1];
          for(Int d = m_num_dims - 1; d > 0; --d) {
            if(pool_pos[d] >= m_pool_dims[d]) {
              pool_pos[d] = 0;
              ++pool_pos[d-1];
            }
          }

        }

        // Set output entry
        weighted_sum_local.Set(output_index, sample, max_value);

        // Move to next pool offset
        pool_offsets[m_num_dims-1] += m_pool_strides[m_num_dims-1];
        for(Int d = m_num_dims - 1; d > 0; --d) {
          if(pool_offsets[d] + m_pool_dims[d] > m_input_dims[d] + m_pool_pads[d]) {
            pool_offsets[d] = -m_pool_pads[d];
            pool_offsets[d-1] += m_pool_strides[d-1];
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

  // Perform back propagation on each GPU
  for(int i=0; i<num_gpus; ++i) {
    checkCUDA(cudaSetDevice(m_cudnn->get_gpu(i)));
    checkCUDNN(cudnnPoolingBackward(m_cudnn->get_handle(i),
                                    m_pooling_desc,
                                    &one,
                                    m_output_desc,
                                    m_weighted_sum_d[i],
                                    m_output_desc,
                                    m_prev_error_signal_d[i],
                                    m_input_desc,
                                    m_prev_activations_d[i],
                                    &zero,
                                    m_input_desc,
                                    m_error_signal_d[i]));
  }

#endif // #ifndef __LIB_CUDNN
}

void lbann::pooling_layer::bp_linearity_cpu() {

  // Get local matrices
  const Mat& prev_activations_local = m_prev_activations_v->LockedMatrix();
  const Mat& activations_local = m_activations_v->LockedMatrix();
  const Mat& prev_error_signal_local = m_prev_error_signal_v->LockedMatrix();
  Mat& error_signal_local = m_error_signal_v->Matrix();

  // Initialize error signal to zero
  Zero(error_signal_local);

  // Throw exception if pooling mode is not max pooling
  if(m_pool_mode != pool_mode::max) {
    throw lbann_exception("lbann_layer_pooling: CPU pooling layer only implements max pooling");
  }

  // Input and output entries are divided amongst channels
  const Int num_per_output_channel = NumNeurons / m_num_channels;
  const Int num_per_input_channel = m_num_prev_neurons / m_num_channels;

  // Iterate through data samples in mini-batch
#pragma omp parallel for
  for(Int sample = 0; sample < prev_activations_local.Width(); ++sample) {

    // Iterate through channels
    for(Int channel = 0; channel < m_num_channels; ++channel) {

      // Iterate through output entries in current channel
      // Note: each output entry corresponds to a different offset of
      //   the pool kernel
      std::vector<Int> pool_offsets(m_num_dims);
      for(int d = 0; d < m_num_dims; ++d) {
        pool_offsets[d] = -m_pool_pads[d];
      }
      const Int start_output_index = channel*num_per_output_channel;
      const Int end_output_index = (channel+1)*num_per_output_channel;
      for(Int output_index = start_output_index;
          output_index < end_output_index;
          ++output_index) {

        // Iterate through pool entries and find maximum
        std::vector<Int> pool_pos(m_num_dims, 0);
        std::vector<Int> max_input_indices;
        DataType max_value = -INFINITY;
        while(pool_pos[0] < m_pool_dims[0]) {

          // Get input entry corresponding to pool entry
          Int input_index = 0;
          bool valid_input_entry = true;
          for(Int d = 0; d < m_num_dims; ++d) {
            if(pool_offsets[d] + pool_pos[d] < 0
               || pool_offsets[d] + pool_pos[d] >= m_input_dims[d]) {
              valid_input_entry = false;
              break;
            }
            input_index *= m_input_dims[d];
            input_index += pool_offsets[d] + pool_pos[d];
          }
          input_index += channel * num_per_input_channel;

          // Check if pool entry is largest encountered
          if(valid_input_entry) {
            const DataType value = prev_activations_local.Get(input_index, sample);
            if(value >= max_value) {
              if(value > max_value) {
                max_value = value;
                max_input_indices.clear();
              }
              max_input_indices.push_back(input_index);
            }
          }

          // Move to next pool entry
          ++pool_pos[m_num_dims-1];
          for(Int d = m_num_dims - 1; d > 0; --d) {
            if(pool_pos[d] >= m_pool_dims[d]) {
              pool_pos[d] = 0;
              ++pool_pos[d-1];
            }
          }

        }

        // Propagate error signal
        // Note: error signal is normalized if multiple entries are the largest
        if(max_input_indices.size() > 0) {
          DataType error_signal_entry = prev_error_signal_local.Get(output_index, sample);
          if(max_input_indices.size() > 1) {
            error_signal_entry /= max_input_indices.size();
          }
          for(Int i=0; i<max_input_indices.size(); ++i) {
            error_signal_local.Update(max_input_indices[i],
                                      sample,
                                      error_signal_entry);
          }
        }

        // Move to next pool offset
        pool_offsets[m_num_dims-1] += m_pool_strides[m_num_dims-1];
        for(Int d = m_num_dims - 1; d > 0; --d) {
          if(pool_offsets[d] + m_pool_dims[d] > m_input_dims[d] + m_pool_pads[d]) {
            pool_offsets[d] = -m_pool_pads[d];
            pool_offsets[d-1] += m_pool_strides[d-1];
          }
        }

      }

    }

  }

}

bool pooling_layer::update()
{
  double start = get_time();
  Layer::update();
  update_time += get_time() - start;
  return true;
}
