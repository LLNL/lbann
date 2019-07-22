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
// callback_sync_selected.cpp - Callback to synchronize selected layers
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_sync_selected.hpp"
#include "lbann/utils/timer.hpp"
#ifdef LBANN_NVPROF
#include <cuda_profiler_api.h>
#include "lbann/utils/file_utils.hpp"
#include <sstream>
#endif // LBANN_NVPROF

namespace lbann {

bool lbann_callback_sync_selected::m_cuda_profiler_initialized = false;
const std::map<lbann_callback_sync_selected::prop_t, std::string>
  lbann_callback_sync_selected::m_prop_str
    = {std::make_pair(lbann_callback_sync_selected::prop_t::Both, "Both"),
       std::make_pair(lbann_callback_sync_selected::prop_t::Forward, "Forward"),
       std::make_pair(lbann_callback_sync_selected::prop_t::Backward, "Backward")};

lbann_callback_sync_selected::lbann_callback_sync_selected(
  const lbann_callback_sync_selected::layers_t& layers, bool async_gpus, bool async_mpi)
  : lbann_callback_sync_layers(!async_gpus, !async_mpi, false),
    m_layers(layers), m_all_set(false) {
  #ifdef LBANN_NVPROF
  cudaProfilerStop(); // make sure to flush out profile data
  #endif

  size_t cnt_fwd = 0u;
  size_t cnt_bwd = 0u;
  for(const auto& l: m_layers) {
    switch (l.second) {
      case Forward: cnt_fwd ++; break;
      case Backward: cnt_bwd ++; break;
      case Both: cnt_fwd ++; cnt_bwd ++; break;
    }
  }
  m_fwd_ptrs.reserve(cnt_fwd);
  m_bwd_ptrs.reserve(cnt_bwd);
}

lbann_callback_sync_selected::~lbann_callback_sync_selected() {
  #ifdef LBANN_NVPROF
  cudaProfilerStop(); // make sure to flush out profile data
  #endif
}

std::string lbann_callback_sync_selected::get_description() const {
  std::string selection;
  for (const auto& l: m_layers) {
    std::map<prop_t, std::string>::const_iterator it = m_prop_str.find(l.second);
    selection += l.first + '.' + it->second + ' ';
  }
  return "sync_selected : { " + selection + '}';
}

void lbann_callback_sync_selected::turn_off_init_cuda_profiler() {
  m_cuda_profiler_initialized = true;
}

bool lbann_callback_sync_selected::check_if_cuda_profiler_initialized() {
  return m_cuda_profiler_initialized;
}

/**
 * Allow users to pass parameters to cudaProfilerInitialize() via prototext.
 * @param cfg_file configuration file for cuda profiler.
 *        (cuda_profiler_setup::config_file in the prototext)
 * @param out_dir output mode for cuda profiler.
 *        (cuda_profiler_setup::output_dir in the prototext)
 * @param out_mode output mode for cuda profiler.
 *        (cuda_profiler_setup::output_mode in the prototext)
 * @param comm global world communicator.
 * The profile output will be wrttien to out_dir/layer_name.prop.rank.prof
 */
void lbann_callback_sync_selected::init_cuda_profiler(
  const std::string cfg_file, const std::string out_dir, int out_mode, lbann_comm* comm) const {
#ifdef LBANN_NVPROF
  if (check_if_cuda_profiler_initialized()) {
    return;
  }
  turn_off_init_cuda_profiler();

  std::string o_dir = out_dir;
  if (comm->am_world_master()) {
    if (!lbann::create_dir(o_dir)) {
      throw lbann_exception("sync_selected failed to create output directory: " + out_dir);
    }
  }
  o_dir = add_delimiter(o_dir);

  El::GPUManager::SynchronizeDevice();
  comm->global_barrier();

  std::string selection;
  for (const auto& l: m_layers) {
    std::map<prop_t, std::string>::const_iterator it = m_prop_str.find(l.second);
    selection += l.first + '.' + it->second + '.';
  }
  const std::string o_prefix = o_dir + selection;
  const int my_rank = comm->get_rank_in_world();
  const std::string o_file = o_prefix + std::to_string(my_rank) + ".prof";
  const cudaOutputMode_t o_mode = (out_mode == 0)? cudaKeyValuePair : cudaCSV;

  const auto ret = cudaProfilerInitialize(cfg_file.c_str(), o_file.c_str(), o_mode);

  if (ret == cudaErrorInvalidValue) {
    throw lbann_exception("sync_selected is unabled to initialze cuda profiler: invalid inputs.");
  } else if (ret == cudaErrorProfilerDisabled) {
    std::stringstream err;
    err << "sync_selected is unable to initialize cuda profiler: " << std::endl
        << "  An external profiling tool (nvprof/nvvp) may already be running." << std::endl
        << "  To use this callback with such a tool, set 'cuda_profiler::no_init'." << std::endl;
    throw lbann_exception(err.str());
  } else {
    cudaProfilerStop(); // suppress profiling until reaching the region of interest

    if (comm->am_world_master()) {
      std::string msg = "Preparing callback sync_selected";
      if (!o_prefix.empty()) {
        msg += " with cudaProfiler writing to " + o_prefix + ".rank.prof";
      }
      std::cout << msg << std::endl;
    }
  }
#endif
}

void lbann_callback_sync_selected::setup(model *m) {
  const std::vector<Layer *>& layers = m->get_layers();
  for (auto l: layers) {
    populate_layer_ptrs(l, Forward);
    populate_layer_ptrs(l, Backward);
  }
  if (!m_all_set) {
    throw lbann_exception("sync_selected cannot recognize all the layer names");
  }
}


void lbann_callback_sync_selected::on_forward_prop_begin(model *m, Layer *l) {
  const layer_ptrs_t::const_iterator it = m_fwd_ptrs.find(l);

  if (it == m_fwd_ptrs.cend()) {
    return;
  }
  // We do not measure the time to synchronize here and thus not contribute it
  // back to the cost of the preceding layer as we are only interested in the
  // selected layer.
  do_pre_sync(l);
}

void lbann_callback_sync_selected::on_forward_prop_end(model *m, Layer *l) {
  const layer_ptrs_t::const_iterator it = m_fwd_ptrs.find(l);
  if (it == m_fwd_ptrs.cend()) {
    return;
  }
  const double start = get_time();
  do_sync(l);
  l->m_fp_time += get_time() - start;
}

void lbann_callback_sync_selected::on_backward_prop_begin(model *m, Layer *l) {
  const layer_ptrs_t::const_iterator it = m_bwd_ptrs.find(l);

  if (it == m_bwd_ptrs.cend()) {
    return;
  }
  do_pre_sync(l);
}

void lbann_callback_sync_selected::on_backward_prop_end(model *m, Layer *l) {
  const layer_ptrs_t::const_iterator it = m_bwd_ptrs.find(l);
  if (it == m_bwd_ptrs.cend()) {
    return;
  }
  const double start = get_time();
  do_sync(l);
  l->m_bp_time += get_time() - start;
}

bool lbann_callback_sync_selected::check_if_all_accounted_for() const {
  return (m_fwd_ptrs.size() + m_bwd_ptrs.size()
         == m_layers.size() + m_both_ptrs.size());
}

/**
 * When the pointer of a selected layer is not known, rely on the layer name
 * to match. When the first time the match is found, save the pointer of the
 * selected layer and use it for the subsequent matching instead of name.
 */
lbann_callback_sync_selected::layer_ptrs_t::iterator
lbann_callback_sync_selected::populate_layer_ptrs(
  Layer* l, const lbann_callback_sync_selected::prop_t current_prop) {

  std::pair<layer_ptrs_t::iterator, bool> ret
    = std::make_pair(((current_prop == Forward)? m_fwd_ptrs.end() : m_bwd_ptrs.end()), false);

  const layers_t::const_iterator it = m_layers.find(l->get_name());

  if (it != m_layers.cend()) { // A matching layer is found
    const prop_t selected_prop = it->second;

    if ((selected_prop != Both) && (selected_prop != current_prop)) {
      return ret.first; // Prop direction does not match
    }

    if (selected_prop == Forward) {
      ret = m_fwd_ptrs.emplace(l);
    } else if (selected_prop == Backward) {
      ret = m_bwd_ptrs.emplace(l);
    } else { // Both
      m_both_ptrs.emplace(l);

      if (current_prop == Forward) {
        ret = m_fwd_ptrs.emplace(l);
        m_bwd_ptrs.emplace(l);
      } else {
        m_fwd_ptrs.emplace(l);
        ret = m_bwd_ptrs.emplace(l);
      }
    }
    if (check_if_all_accounted_for()) {
      m_all_set = true;
    }
  }
  return ret.first;
}


void lbann_callback_sync_selected::do_pre_sync(Layer *l) {
  lbann_callback_sync_layers::do_sync(l);
  #ifdef LBANN_NVPROF
  cudaProfilerStart();
  #endif
}

void lbann_callback_sync_selected::do_sync(Layer *l) {
#ifdef LBANN_NVPROF //(also deinfed LBANN_HAS_GPU)
  if (m_sync_gpus) {
    El::GPUManager::SynchronizeDevice();
    cudaProfilerStop();
  }
  if (m_sync_mpi) {
    l->get_comm()->global_barrier();
  }
  if (!m_sync_gpus) {
    cudaProfilerStop();
  }
#else
  lbann_callback_sync_layers::do_sync(l);
#endif
}

std::unique_ptr<lbann_callback>
build_callback_sync_selected_from_pbuf(
  const google::protobuf::Message& proto_msg, lbann_summary*) {
  const auto& params =
    dynamic_cast<const lbann_data::Callback::CallbackSyncSelected&>(proto_msg);
  const int num_layers = params.layer_to_sync_size();
  if (num_layers == 0) {
    throw lbann_exception("sync_selected requires at least a layer "
                          "to synchronize.");
  }

  using layers_t = lbann_callback_sync_selected::layers_t;
  using prop_t = lbann_callback_sync_selected::prop_t;

  layers_t selected_layers;
  selected_layers.reserve(num_layers);

  for (int i = 0; i < num_layers; ++i) {
    const auto& layer_to_sync = params.layer_to_sync(i);
    selected_layers.emplace(layer_to_sync.name(),
                            static_cast<prop_t>(layer_to_sync.prop()));
  }

  auto cb_ptr
    = make_unique<lbann_callback_sync_selected>(selected_layers,
                                                params.async_gpus(),
                                                params.async_mpi());

#ifdef LBANN_NVPROF
  const auto& cp_setup = params.cuda_profiler_setup();
  if (cp_setup.no_init()) {
    lbann_callback_sync_selected::turn_off_init_cuda_profiler();
  } else {
    cb_ptr->init_cuda_profiler(cp_setup.config_file(),
                               cp_setup.output_dir(),
                               cp_setup.output_mode(),
                               comm);
  }
#endif // LBANN_NVPROF
  return cb_ptr;
}

}  // namespace lbann
