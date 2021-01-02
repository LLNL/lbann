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
// callback_kfac .hpp .cpp - Callbacks for the K-FAC 2nd-order opt. method
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_KFAC_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_KFAC_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"
#include "lbann/callbacks/kfac/kfac_block.hpp"

namespace lbann {
namespace callback {

enum class kfac_inverse_strategy {
  ALL,  // Apply round-robin assingment to all of the layers. may cause load imbalance.
  EACH, // Apply round-robin assingment to every type of layers. may
  // not work well for small networks.
  ROOT, // Use only the root GPU. This is only for testing.
};

enum class kfac_reduce_scatter_mode {
  ALLREDUCE, // Use lbann_comm::allreduce
  REDUCE_SCATTER, // Use El::ReduceScatter
  REDUCE, // Use El::Reduce for each block
};

enum class kfac_allgather_mode {
  ALLREDUCE, // Use lbann_comm::allreduce
  ALLGATHER, // Use El::ReduceScatter
  BROADCAST // Use El::Broadcast for each block
};

// Forward declarations
// TODO: Remove if kfac_block no longer refers kfac
template <El::Device Device>
class kfac_block;

/** Callback hooks for the K-FAC method.
 *
 * Martens, James and Roger Grosse. "Optimizing neural networks with
 * kronecker-factored approximate curvature." International conference
 * on machine learning. 2015.
 *
 * Grosse, Roger, and James Martens. "A kronecker-factored approximate
 * fisher matrix for convolution layers." International Conference on
 * Machine Learning. 2016.
 *
 * Osawa, Kazuki, et al. "Large-scale distributed second-order
 * optimization using kronecker-factored approximate curvature for
 * deep convolutional neural networks." Proceedings of the IEEE
 * Conference on Computer Vision and Pattern Recognition. 2019.
 */
template <El::Device Device>
class kfac : public callback_base {
 public:

  /** Constructor.
   */
  kfac(const std::vector<double>& damping_act_params,
       const std::vector<double>& damping_err_params,
       const std::vector<double>& damping_bn_act_params,
       const std::vector<double>& damping_bn_err_params,
       size_t damping_warmup_steps,
       double kronecker_decay,
       bool print_time,  bool print_matrix, bool print_matrix_summary,
       bool use_pi,
       const std::vector<size_t>& update_intervals,
       size_t update_interval_steps,
       kfac_inverse_strategy inverse_strategy,
       const std::vector<std::string>& disable_layers,
       double learning_rate_factor)
  : callback_base(),
    m_damping_act_params(damping_act_params),
    m_damping_err_params(damping_err_params),
    m_damping_bn_act_params(damping_bn_act_params), m_damping_bn_err_params(damping_bn_err_params),
    m_damping_warmup_steps(damping_warmup_steps),
    m_kronecker_decay(kronecker_decay),
    m_print_time(print_time), m_print_matrix(print_matrix),
    m_print_matrix_summary(print_matrix_summary),
    m_use_pi(use_pi),
    m_update_intervals(update_intervals),
    m_update_interval_steps(update_interval_steps),
    m_inverse_strategy(inverse_strategy),
    m_disable_layers(disable_layers),
    m_learning_rate_factor(learning_rate_factor) {
    m_has_kronecker_inverse = false;
    m_damping_act = m_damping_act_params[0];
    m_damping_err = m_damping_err_params[0];
    m_damping_bn_act = m_damping_bn_act_params[0];
    m_damping_bn_err = m_damping_bn_err_params[0];
  }
  kfac(const kfac&) = default;
  kfac& operator=(const kfac&) = default;
  kfac* copy() const override { return new kfac(*this); }
  std::string name() const override { return "K-FAC"; }

#ifdef LBANN_HAS_GPU
  void setup(model *m) override;
  void setup(trainer *t) override {}
  void on_backward_prop_end(model *m) override;
  void on_epoch_end(model *m) override;
  void on_backward_prop_end(model *m, Layer *l) override {}

  /** @brief Gets the Kronecker factor matrix of a FC layer.
   *  The same key is tied with the same matrix instance. */
  El::Matrix<DataType, Device>& get_workspace_matrix(
      const std::string& key,
      size_t height, size_t width);

#else
  void setup(model *m) override {
    LBANN_ERROR("The K-FAC callback is available only on GPUs.");
  }
#endif // LBANN_HAS_GPU

  /** @brief The default parameters of a Tikhonov damping technique. */
  constexpr static const double damping_0_default = 3e-2;
  constexpr static const size_t damping_warmup_steps_default = 100;

  /** @brief The default parameters of the decay factor. */
  constexpr static const double kronecker_decay_default = 0.99;

 private:

  /** @brief Pairs of the initial and the target damping value.
   *  If only one value is specified, it will be used throughout training.
   */
  const std::vector<double> m_damping_act_params, m_damping_err_params,
    m_damping_bn_act_params, m_damping_bn_err_params;

  /** @brief The number of warmup steps of the Tikhnov damping technique. */
  const size_t m_damping_warmup_steps;

  /** @brief The decay factor of kronecker factors. */
  const double m_kronecker_decay;

  /** @brief Knobs to print information for debugging. */
  const bool m_print_time, m_print_matrix, m_print_matrix_summary;

  /** @brief Weather to use the pi constant to adjust the damping
      constant. */
  const bool m_use_pi;

  /** @brief Space-separated pairs of the initial and the target update intervals.
   *If only one value is specified, it will be used throughout
   *training.
   */
  const std::vector<size_t> m_update_intervals;

  /** @brief The number of steps for changing the update interval. */
  const size_t m_update_interval_steps;

  /** @brief Assignment strategy for the model-parallel part. */
  const kfac_inverse_strategy m_inverse_strategy;

  /** @brief List of layers to be ignored by the callback. */
  const std::vector<std::string> m_disable_layers;

  /** @brief Factor to be multiplied to the learning rate */
  const double m_learning_rate_factor;

  /** @brief Whether inverse of Kronecker factors are available. */
  bool m_has_kronecker_inverse;

  /** @brief The current damping values. */
  double m_damping_act, m_damping_err,
    m_damping_bn_act, m_damping_bn_err;

  /** @brief The current update interval. */
  size_t m_update_interval;

  /** @brief K-FAC per-layer blocks. */
  std::vector<std::shared_ptr<kfac_block<Device>>> m_blocks;

#ifdef LBANN_HAS_GPU
  /** @brief Workspace matrices that are used by m_blocks. */
  std::unordered_map<std::string,
                     El::Matrix<DataType, Device>> m_workspace;
#endif // LBANN_HAS_GPU

  /** @brief Copy of lbann_comm::get_rank_in_trainer() for internal
      functions. TODO: Get the rank from lbann_comm directly */
  int m_rank;

  /** @brief Parameters for prof_region_*. */
  constexpr static const bool prof_sync = true;
  constexpr static const int prof_color = 0;

};

// Builder function
std::unique_ptr<callback_base>
build_kfac_callback_from_pbuf(
    const google::protobuf::Message&,std::shared_ptr<lbann_summary> const&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_KFAC_HPP_INCLUDED
