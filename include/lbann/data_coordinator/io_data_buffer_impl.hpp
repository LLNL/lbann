////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_IO_BUFFER_HPP_IMPL_INCLUDED
#define LBANN_IO_BUFFER_HPP_IMPL_INCLUDED

#include "lbann/data_coordinator/io_data_buffer.hpp"

namespace lbann {

template <typename TensorDataType>
template <class Archive>
void data_buffer<TensorDataType>::serialize(Archive& ar)
{
  ar(/*CEREAL_NVP(m_input_buffers)*//*,
                                  CEREAL_NVP(m_fetch_data_in_background),
     CEREAL_NVP(m_data_fetch_future),
     CEREAL_NVP(m_indices_fetched_per_mb)*/);
}

template <typename TensorDataType>
void data_buffer<TensorDataType>::initialize_buffer_for_data_field(
  data_field_type const data_field,
  lbann_comm* comm)
{
  // Allocate a buffer if the data field doesn't exist
  if (m_input_buffers.find(data_field) == m_input_buffers.end()) {
    m_input_buffers[data_field] =
      std::make_unique<StarVCMatDT<TensorDataType, El::Device::CPU>>(
        comm->get_trainer_grid());
#if defined(LBANN_HAS_GPU)
    // Pin the memory so that we get efficient GPU data transfer
    m_input_buffers[data_field]->Matrix().SetMemoryMode(1);
#endif // LBANN_HAS_GPU
    m_num_samples_per_field_distributed[data_field] = 0;
  }
}

} // namespace lbann

#endif // LBANN_IO_BUFFER_HPP_IMPL_INCLUDED
