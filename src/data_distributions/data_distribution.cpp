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

#include "lbann/data_distributions/data_distribution.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {
generic_data_distribution::generic_data_distribution(lbann_comm *comm, int num_parallel_readers, std::map<execution_mode, generic_data_reader *> data_readers)
  : m_comm(comm), m_requested_max_num_parallel_readers(num_parallel_readers), 
    fetch_data_fn(nullptr),  update_data_reader_fn(nullptr)
{
  //, m_data_readers(data_readers) {
  m_root = 0;
  m_num_samples_in_batch = 0;
}

generic_data_distribution::generic_data_distribution(const generic_data_distribution& rhs) 
: m_comm(rhs.m_comm),  m_root(rhs.m_root),
  m_requested_max_num_parallel_readers(rhs. m_requested_max_num_parallel_readers),
  m_local_reader_done(rhs.m_local_reader_done),
  m_num_samples_in_batch(rhs. m_num_samples_in_batch),
  m_local_data_valid(rhs.m_local_data_valid)
{
  if (rhs.fetch_data_fn)
    fetch_data_fn = new fetch_data_functor(*(rhs.fetch_data_fn));
  if (rhs.update_data_reader_fn)
    update_data_reader_fn = new update_data_reader_functor(*(rhs.update_data_reader_fn));
}

generic_data_distribution& generic_data_distribution::operator=(const generic_data_distribution& rhs) {
  m_comm = rhs.m_comm;
  m_root = rhs.m_root;
  m_requested_max_num_parallel_readers = rhs. m_requested_max_num_parallel_readers;
  m_local_reader_done = rhs.m_local_reader_done;
  m_num_samples_in_batch = rhs. m_num_samples_in_batch;
  m_local_data_valid = rhs.m_local_data_valid;

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
