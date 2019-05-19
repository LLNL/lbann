////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/io/data_buffers/generic_io_buffer.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {
generic_io_buffer::generic_io_buffer(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers)
  : m_comm(comm), fetch_data_fn(nullptr),  update_data_reader_fn(nullptr) {}

generic_io_buffer::generic_io_buffer(const generic_io_buffer& rhs)
: m_comm(rhs.m_comm)
{
  if (rhs.fetch_data_fn)
    fetch_data_fn = new fetch_data_functor(*(rhs.fetch_data_fn));
  if (rhs.update_data_reader_fn)
    update_data_reader_fn = new update_data_reader_functor(*(rhs.update_data_reader_fn));
}

generic_io_buffer& generic_io_buffer::operator=(const generic_io_buffer& rhs) {
  m_comm = rhs.m_comm;
  if (fetch_data_fn) {
    delete fetch_data_fn;
    fetch_data_fn = nullptr;
  }
  if (update_data_reader_fn) {
    delete update_data_reader_fn;
    update_data_reader_fn = nullptr;
  }
  if (rhs.fetch_data_fn)
    fetch_data_fn = new fetch_data_functor(*(rhs.fetch_data_fn));
  if (rhs.update_data_reader_fn)
    update_data_reader_fn = new update_data_reader_functor(*(rhs.update_data_reader_fn));

  return (*this);
}

} // namespace lbann
