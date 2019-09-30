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
// callback_sync_selected.hpp - Callback to synchronize selected layers
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_SYNC_SELECTED_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_SYNC_SELECTED_HPP_INCLUDED

#include "lbann/callbacks/callback_sync_layers.hpp"
#include <unordered_map>
#include <unordered_set>

namespace lbann {

/**
 * Synchronize at the beginning and the end of the propagation operation(s) of
 * a selected layer, which can be both/either of the forward prop and/or the
 * backward prop of the layer. Additionally updates layer timing information to
 * account for the synchronization at the end of propagation(s).
 * When nvprof is enabled, cudaProfilerStart() follows the synchronization
 * inserted at the beginning of the selected prop step(s), and cudaProfilerEnd()
 * comes after the local GPU sychronization and before the global MPI barrier
 * inserted at the end of the selected prop step(s).
 * Note that this callback should come before the summarizer callback
 * as the base callback lbann_callback_sync_layers requires.
 */
class lbann_callback_sync_selected : public lbann_callback_sync_layers {
 public:
  ///type of propagation toch synchronize
  enum prop_t {Both = 0, Forward = 1, Backward = 2};
  static const std::map<prop_t, std::string> m_prop_str;

  using layers_t = std::unordered_map<std::string, prop_t>;
  using layer_ptrs_t = std::unordered_set<Layer*>;

  /**
   * @param layers specifies the layers to synchronize
   * @param async_gpus sets not to synchronize gpus. The default is false.
   * @param async_mpi sets not to synchronize mpi. The default is false.
   */
  lbann_callback_sync_selected(const layers_t& layers,
                               bool async_gpus = false, bool async_mpi = false);

  lbann_callback_sync_selected(const lbann_callback_sync_selected&) = default;

  lbann_callback_sync_selected& operator=(
    const lbann_callback_sync_selected&) = default;

  lbann_callback_sync_selected* copy() const override {
    return new lbann_callback_sync_selected(*this);
  }

  ~lbann_callback_sync_selected() override;

  std::string name() const override { return "sync_selected"; }
  std::string get_description() const;

  /// To protect in case that cudaProfilerInitialized() has already been called
  static void turn_off_init_cuda_profiler();

  /// Tells if cuda_profiler has been initialized
  static bool check_if_cuda_profiler_initialized();

  void init_cuda_profiler(const std::string cfg_file, const std::string out_dir,
                          int out_mode, lbann_comm* comm) const;

  /** Called once to set up the callback (after all layers are set up).
   * Then, populate the layer pointers */
  void setup(model *m) override;

  using lbann_callback::on_forward_prop_begin;
  using lbann_callback::on_backward_prop_begin;
  using lbann_callback_sync_layers::on_forward_prop_end;
  using lbann_callback_sync_layers::on_backward_prop_end;

  /// Synchronize at the beginning of the forward prop of layer l
  void on_forward_prop_begin(model* m, Layer* l) override;
  /// Synchronize at the end of the forward prop of layer l
  void on_forward_prop_end(model* m, Layer* l) override;
  /// Synchronize at the beginning of the backward prop of layer l
  void on_backward_prop_begin(model* m, Layer* l) override;
  /// Synchronize at the end of the backward prop of layer l
  void on_backward_prop_end(model* m, Layer* l) override;

 protected:
  bool check_if_all_accounted_for() const;

  layer_ptrs_t::iterator populate_layer_ptrs(Layer* l, const prop_t current_prop);

  /// Synchronize and enable cuda profiler
  void do_pre_sync(Layer* l);
  /// Synchronize and disble cuda profiler
  void do_sync(Layer* l) override;

  /// The layers to synchronize.
  layers_t m_layers;

  /** The pointers of layers to synchronize for forward prop.
   *  This set includes those of layers to synchronize for both props. */
  layer_ptrs_t m_fwd_ptrs;
  /** The pointers of layers to synchronize for backward prop.
   *  This set includes those of layers to synchronize for both props. */
  layer_ptrs_t m_bwd_ptrs;
  /// The pointers of layers to synchronize for both props.
  layer_ptrs_t m_both_ptrs;

  bool m_all_set; ///< whether all the layer pointers are collected

  /// Tells if cudaProfilerInitialized() has already been called.
  static bool m_cuda_profiler_initialized;
};

// Builder function
std::unique_ptr<lbann_callback>
build_callback_sync_selected_from_pbuf(
  const google::protobuf::Message&, lbann_summary*);

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_SYNC_SELECTED_HPP_INCLUDED
