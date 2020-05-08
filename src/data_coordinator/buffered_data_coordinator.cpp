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
////////////////////////////////////////////////////////////////////////////////

#include <lbann/data_coordinator/buffered_data_coordinator.hpp>
#include <lbann/trainers/trainer.hpp>

namespace lbann {

template <typename TensorDataType>
void buffered_data_coordinator<TensorDataType>::setup(thread_pool& io_thread_pool, int max_mini_batch_size) {
  data_coordinator::setup(io_thread_pool, max_mini_batch_size);

  auto data_dims = get_data_dims();
  for (auto& io_buffer : m_io_buffers) {
    io_buffer->setup_data(get_linearized_data_size(),
                          get_linearized_response_size(),
                          max_mini_batch_size);
  }

  for (auto& io_buffer : m_io_buffers) {
    // How much data does the buffer fetch, just the sample or the
    // response as well?
    for (int i = 0; i < 2; ++i) {
      io_buffer->fp_setup_data(max_mini_batch_size, i);
    }
  }

  for (auto io_buffer : m_io_buffers) {
    io_buffer->fetch_data_fn = new fetch_data_functor<IODataType>(data_reader_target_mode::CLASSIFICATION/*target_mode*/);
    io_buffer->update_data_reader_fn = new update_data_reader_functor();
  }
}

#define PROTO(T)                     \
  template class buffered_data_coordinator<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
