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
// lbann_callback_timer .hpp .cpp - Callback hooks to time training
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include "lbann/callbacks/profiler.hpp"

#if defined(LBANN_SCOREP)
#include <scorep/SCOREP_User.h>
#elif defined(LBANN_NVPROF)
#include "nvToolsExt.h"
#include "cuda_runtime.h"
#endif

namespace {
#if defined(LBANN_SCOREP)
static void prof_region_begin(const char *s, int c) {
  SCOREP_USER_REGION_BY_NAME_BEGIN(s, SCOREP_USER_REGION_TYPE_COMMON);
  return;
}
static void prof_region_end(const char *s) {
  SCOREP_USER_REGION_BY_NAME_END(s);
  return;
}
#elif defined(LBANN_NVPROF)
static void synchronize_all_devices() {
  int count;
  cudaGetDeviceCount(&count);
  for (int i = 0; i < count; ++i) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
}

static void prof_region_begin(const char *s, int c) {
  synchronize_all_devices();
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
static void prof_region_end(const char *s) {
  synchronize_all_devices();    
  nvtxRangePop();
}
#else
static void prof_region_begin(const char *s, int c) {
  return;
}
static void prof_region_end(const char *s) {
  return;
}
#endif
}

namespace lbann {

void lbann_callback_profiler::on_epoch_begin(model *m) {
  prof_region_begin("epoch", colors[0]);
}

void lbann_callback_profiler::on_epoch_end(model *m) {
  prof_region_end("epoch");
}

void lbann_callback_profiler::on_batch_begin(model *m) {
  prof_region_begin("batch", colors[1]);
}

void lbann_callback_profiler::on_batch_end(model *m) {
  prof_region_end("batch");  
}

void lbann_callback_profiler::on_forward_prop_begin(model *m) {
  prof_region_begin("forward", colors[2]);
}

void lbann_callback_profiler::on_forward_prop_end(model *m) {
  prof_region_end("forward");
}

void lbann_callback_profiler::on_backward_prop_begin(model *m) {
  prof_region_begin("backward", colors[3]);
}

void lbann_callback_profiler::on_backward_prop_end(model *m) {
  prof_region_end("backward");
}

int lbann_callback_profiler::get_color(Layer *l) {
  const std::string &lname = l->get_type();
  int idx = 4;
  if (lname == "fully_connected") {
    idx = 5;
  } else if (lname == "convolution") {
    idx = 6;
  } else if (lname == "pooling_layer") {
    idx = 7;
  } else if (lname == "input_layer_distributed_minibatch_parallel_io") {
    idx = 8;
  }
  return colors[idx % num_colors];
}

void lbann_callback_profiler::on_forward_prop_begin(model *m, Layer *l) {
  prof_region_begin(("fw " + l->get_type()).c_str(), get_color(l));  
}

void lbann_callback_profiler::on_forward_prop_end(model *m, Layer *l) {
  prof_region_end(("fw " + l->get_type()).c_str());    
}

void lbann_callback_profiler::on_backward_prop_begin(model *m, Layer *l) {
  prof_region_begin(("bw " + l->get_type()).c_str(), get_color(l));    
}

void lbann_callback_profiler::on_backward_prop_end(model *m, Layer *l) {
  prof_region_end(("bw " + l->get_type()).c_str());
}

}  // namespace lbann
