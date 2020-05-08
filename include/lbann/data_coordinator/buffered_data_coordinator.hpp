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

#ifndef LBANN_BUFFERED_DATA_COORDINATOR_HPP
#define LBANN_BUFFERED_DATA_COORDINATOR_HPP

#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/io/data_buffers/generic_io_buffer.hpp"
#include "lbann/io/data_buffers/partitioned_io_buffer.hpp"


namespace lbann {

template <typename TensorDataType>
class buffered_data_coordinator : public data_coordinator {
 public:
  /** @name Public Types */
  ///@{

  /** @brief The local tensor type expected for IO in this object. */
  using IODataType = DataType;

  ///@}
 public:
  buffered_data_coordinator(trainer& trainer, lbann_comm *comm, std::map<execution_mode, generic_data_reader *> data_readers) :
    data_coordinator(trainer, comm, data_readers),
    m_io_buffers() {

    // Initialize two buffers
    initialize_io_buffer(comm/*, std::min(num_parallel_readers, data_type_layer<TensorDataType>::m_comm->get_procs_per_trainer())*/);
    initialize_io_buffer(comm/*, std::min(num_parallel_readers, data_type_layer<TensorDataType>::m_comm->get_procs_per_trainer())*/);

    this->m_active_buffer[execution_mode::training].store(-1);
    this->m_active_buffer[execution_mode::validation].store(-1);
    this->m_active_buffer[execution_mode::testing].store(-1);
  }

  ~buffered_data_coordinator() {
    for (auto& io_buffer : m_io_buffers) {
      delete io_buffer;
    }
  }

  // Data Coordinators copy their data readers.
  buffered_data_coordinator(const buffered_data_coordinator& other)
    : data_coordinator(other),
      m_io_buffers(other.m_io_buffers) {
    for (auto& io_buffer : m_io_buffers) {
      io_buffer = io_buffer->copy();
    }
  }

  buffered_data_coordinator& operator=(const data_coordinator& other) {
    data_coordinator::operator=(other);

    for (auto& io_buffer : m_io_buffers) {
      io_buffer = io_buffer->copy();
    }
    return *this;
  }

  /** Archive for checkpoint and restart */
  template <class Archive> void serialize( Archive & ar ) {
    //    ar(/*CEREAL_NVP(m_io_buffer),*/);
  }

  void setup(int max_mini_batch_size);

  inline void initialize_io_buffer(lbann_comm *comm) {
    /// @todo BVE FIXME we need to know how much data buffer based on
    /// the needs of the data reader, not the input layer
    m_io_buffers.push_back(new partitioned_io_buffer<TensorDataType>(comm, 2 /*this->m_expected_num_child_layers*/));
  }

  //************************************************************************
  //
  //************************************************************************

  // save state of IO to a checkpoint
  bool save_to_checkpoint_shared(persist& p) const {
    return true;
  }

  // reload state of IO from a checkpoint
  bool load_from_checkpoint_shared(persist& p) {
    return true;
  }

  bool save_to_checkpoint_distributed(persist& p) const {
    return true;
  }

  bool load_from_checkpoint_distributed(persist& p) {
    return true;
  }

 protected:

public:
  std::vector<generic_io_buffer<TensorDataType>*> m_io_buffers;
  io_buffer_map_t m_active_buffer;

};

} // namespace lbann

#endif // LBANN_BUFFERED_DATA_COORDINATOR_HPP
