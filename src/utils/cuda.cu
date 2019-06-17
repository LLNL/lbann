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

#include "lbann/utils/cuda.hpp"

#ifdef LBANN_HAS_GPU

namespace lbann {
namespace cuda {

////////////////////////////////////////////////////////////
// CUDA event wrapper
////////////////////////////////////////////////////////////

event_wrapper::event_wrapper() : m_event(nullptr), m_stream(0) {
  CHECK_CUDA(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
}

event_wrapper::event_wrapper(const event_wrapper& other)
  : m_event(nullptr), m_stream(other.m_stream) {
  CHECK_CUDA(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
  if (!other.query()) { record(m_stream); }
}

event_wrapper& event_wrapper::operator=(const event_wrapper& other) {
  m_stream = other.m_stream;
  if (!other.query()) { record(m_stream); }
  return *this;
}

event_wrapper::~event_wrapper() {
  cudaEventDestroy(m_event);
}

void event_wrapper::record(cudaStream_t stream) {
  m_stream = stream;
  CHECK_CUDA(cudaEventRecord(m_event, m_stream));
}

bool event_wrapper::query() const {
  const auto& status = cudaEventQuery(m_event);
  switch (status) {
  case cudaSuccess:       return true;
  case cudaErrorNotReady: return false;
  default:
    CHECK_CUDA(status);
    return false;
  }
}

void event_wrapper::synchronize() {
  CHECK_CUDA(cudaEventSynchronize(m_event));
}

cudaEvent_t& event_wrapper::get_event() { return m_event; }

} // namespace cuda
} // namespace lbann

#endif // LBANN_HAS_GPU
