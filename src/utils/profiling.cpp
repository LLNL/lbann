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
//
// profiling .hpp .cpp - Various routines for interfacing with profilers
///////////////////////////////////////////////////////////////////////////////

#include "lbann/base.hpp"
#include "lbann/utils/profiling.hpp"
#if defined(LBANN_SCOREP)
#include <scorep/SCOREP_User.h>
#elif defined(LBANN_NVPROF)
#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"
#include "nvToolsExtCudaRt.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "lbann/utils/gpu/helpers.hpp"
#endif

namespace {
bool profiling_started = false;
}

namespace lbann {

#if defined(LBANN_SCOREP)
void prof_start() {
  profiling_started = true;
  return;
}
void prof_stop() {
  return;
}
void prof_region_begin(const char *s, int, bool) {
  SCOREP_USER_REGION_BY_NAME_BEGIN(s, SCOREP_USER_REGION_TYPE_COMMON);
  return;
}
void prof_region_end(const char *s, bool) {
  SCOREP_USER_REGION_BY_NAME_END(s);
  return;
}
#elif defined(LBANN_NVPROF)
void prof_start() {
  CHECK_CUDA(cudaProfilerStart());
  profiling_started = true;
}
void prof_stop() {
  CHECK_CUDA(cudaProfilerStop());
  profiling_started = false;
}
void prof_region_begin(const char *s, int c, bool sync) {
  if (!profiling_started) return;
  if (sync) {
    hydrogen::gpu::SynchronizeDevice();
  }
  // Doesn't work with gcc 4.9
  // nvtxEventAttributes_t ev = {0};
  nvtxEventAttributes_t ev;
  memset(&ev, 0, sizeof(nvtxEventAttributes_t));
  ev.version = NVTX_VERSION;
  ev.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  ev.colorType = NVTX_COLOR_ARGB;
  ev.color = c;
  ev.messageType = NVTX_MESSAGE_TYPE_ASCII;
  ev.message.ascii = s;
  nvtxRangePushEx(&ev);
}
void prof_region_end(const char *, bool sync) {
  if (!profiling_started) return;
  if (sync) {
    hydrogen::gpu::SynchronizeDevice();
  }
  nvtxRangePop();
}
#else
void prof_start() {
  profiling_started = true;
  return;
}
void prof_stop() {
  return;
}
void prof_region_begin(const char *, int, bool) {
  return;
}
void prof_region_end(const char *, bool) {
  return;
}
#endif

}  // namespace lbann
