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
#ifndef LBANN_EXECUTION_ALGORITHMS_KFAC_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_KFAC_HPP_INCLUDED

#include "lbann/data_ingestion/coordinator/data_coordinator.hpp"
#include "lbann/execution_algorithms/factory.hpp"
#include "lbann/execution_algorithms/kfac/execution_context.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include "lbann/execution_algorithms/training_algorithm.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/make_abstract.hpp"

#include <google/protobuf/message.h>
#include <memory>

namespace lbann {

/** @class KFAC
 *  @brief An implementation of the KFAC second-order optimization algorithm
 *
 *  Martens, James and Roger Grosse. "Optimizing neural networks with
 *  kronecker-factored approximate curvature." International
 *  conference on machine learning. 2015.
 *
 *  Grosse, Roger, and James Martens. "A kronecker-factored
 *  approximate fisher matrix for convolution layers." International
 *  Conference on Machine Learning. 2016.
 *
 *  Osawa, Kazuki, et al. "Large-scale distributed second-order
 *  optimization using kronecker-factored approximate curvature for
 *  deep convolutional neural networks." Proceedings of the IEEE
 *  Conference on Computer Vision and Pattern Recognition. 2019.
 */
class KFAC final : public TrainingAlgorithm
{

public:
  using TermCriteriaType = SGDTerminationCriteria;
  using ExeContextType = kfac::KFACExecutionContext;

public:
  /** @name Life-cycle management */
  ///@{
  /** @brief Construct KFAC from its component pieces.
   */
  KFAC(std::string name,
       std::unique_ptr<TermCriteriaType> stop,
       std::vector<double> damping_act_params,
       std::vector<double> damping_err_params,
       std::vector<double> damping_bn_act_params,
       std::vector<double> damping_bn_err_params,
       std::vector<bool> kfac_use_interval,
       size_t damping_warmup_steps,
       double kronecker_decay,
       bool print_time,
       bool print_matrix,
       bool print_matrix_summary,
       bool use_pi,
       std::vector<size_t> update_intervals,
       size_t update_interval_steps,
       kfac::kfac_inverse_strategy inverse_strategy,
       std::vector<std::string> disable_layers,
       double learning_rate_factor,
       double learning_rate_factor_gru,
       size_t compute_interval,
       bool distribute_precondition_compute,
       bool use_eigen_decomposition,
       bool enable_copy_errors,
       bool enable_copy_activations);

  ~KFAC() noexcept = default;
  KFAC(KFAC const& other) = delete;
  KFAC& operator=(const KFAC& other) = delete;
  KFAC(KFAC&& other) = default;
  KFAC& operator=(KFAC&& other) = default;
  ///@}
  /** @brief Queries */
  ///@{
  std::string get_type() const final;
  ///@}
  /** @name Apply interface */
  ///@{
  /** @brief Apply the training algorithm to refine model weights.
   *  @param[in,out] context The persistent execution context for this
   *                 algorithm.
   *  @param[in,out] m The model to be trained.
   *  @param[in,out] dc The data source for training.
   *  @param[in] mode Completely superfluous.
   */
  void apply(ExecutionContext& context,
             model& m,
             data_coordinator& dc,
             execution_mode mode) final;
  /** @brief Train a model using KFAC. */
  void train(ExeContextType& c,
             model& model,
             data_coordinator& dc,
             TermCriteriaType const& term);
  ///@}

#ifdef LBANN_HAS_GPU
  constexpr static const El::Device Device = El::Device::GPU;
#else
  constexpr static const El::Device Device = El::Device::CPU;
#endif // LBANN_HAS_GPU

  /** @brief The default parameters of a Tikhonov damping technique. */
  constexpr static const double damping_0_default = 3e-2;
  constexpr static const size_t damping_warmup_steps_default = 100;

  /** @brief The default parameters of the decay factor. */
  constexpr static const double kronecker_decay_default = 0.99;

  /** @brief Parameters for prof_region_*. */
  constexpr static const bool prof_sync = true;
  constexpr static const int prof_color = 0;

protected:
  /** @brief Train model on one step / mini-batch of an SGD forward pass */
  bool train_mini_batch(ExeContextType& c, model& model, data_coordinator& dc);

  /** @name Callback hooks */
  ///@{
  /** Execute callbacks at start of training. */
  void do_train_begin_cbs(model& model);
  /** Execute callbacks at end of training. */
  void do_train_end_cbs(model& model);
  /** Execute callbacks at start of epoch. */
  void do_epoch_begin_cbs(model& model);
  /** Execute callbacks at end of epoch. */
  void do_epoch_end_cbs(model& model);
  /** Execute callbacks at start of mini-batch. */
  void do_batch_begin_cbs(model& model);
  /** Execute callbacks at end of mini-batch. */
  void do_batch_end_cbs(model& model);
  ///@}

  /** @brief Covariant return-friendly implementation of
   *         `get_new_exection_context()`.
   */
  kfac::KFACExecutionContext* do_get_new_execution_context() const final;

  void start_send_recv_inverse_matrices(ExeContextType& context,
                                        lbann_comm* comm);
  void end_send_recv_inverse_matrices(ExeContextType& context,
                                      lbann_comm* comm);

private:
#if 1
  /** @todo Break up into more manageable pieces */
  void on_forward_prop_end(ExeContextType& context, model& model);
  void on_backward_prop_end(ExeContextType& context, model& model);

#else
  /** @brief Compute Kronecker factors */
  void compute_kronecker_factors(ExeContextType& context, model& model);

  /** @brief Compute Cholesky factorization of Kronecker factors */
  void invert_kronecker_factors(ExeContextType& context, model& model);

  /** @brief Precondition gradients with Kronecker factors */
  void precondition_gradients(ExeContextType& context, model& model);
#endif // 0

  /** @brief Data exchange functions to synchronize model and weights */
  void sync_weights_model(model& model, lbann_comm* comm);
  void start_sync_weights_async(model& model, lbann_comm* comm);
  void end_sync_weights_async(model& model, lbann_comm* comm);

  void start_old_async_weights_model(model& model,
                                     lbann_comm* comm,
                                     ExeContextType& context);
  void end_old_async_weights_model(model& model,
                                   lbann_comm* comm,
                                   ExeContextType& context);
  void allgather_precondition_gradient(lbann_comm& comm,
                                       ExeContextType& context);

  /** @brief The KFAC stopping criteria. */
  std::unique_ptr<TermCriteriaType> m_stopping_criteria;

  /** @brief Pairs of the initial and the target damping value.
   *  If only one value is specified, it will be used throughout training.
   */
  std::vector<double> m_damping_act_params, m_damping_err_params,
    m_damping_bn_act_params, m_damping_bn_err_params;

  /** @brief The number of warmup steps of the Tikhnov damping technique. */
  size_t m_damping_warmup_steps;

  /** @brief The decay factor of kronecker factors. */
  double m_kronecker_decay;

  /** @brief Knobs to print information for debugging. */
  bool m_print_time, m_print_matrix, m_print_matrix_summary;

  /** @brief Weather to use the pi constant to adjust the damping
      constant. */
  bool m_use_pi;

  /** @brief Space-separated pairs of the initial and the target update
   *intervals. If only one value is specified, it will be used throughout
   *training.
   */
  std::vector<size_t> m_update_intervals;

  /** @brief The number of steps for changing the update interval. */
  size_t m_update_interval_steps;

  /** @brief Assignment strategy for the model-parallel part. */
  kfac::kfac_inverse_strategy m_inverse_strategy;

  /** @brief List of layers to be ignored by the callback. */
  std::vector<std::string> m_disable_layers;

  /** @brief Factors to be multiplied to the learning rate */
  double m_learning_rate_factor, m_learning_rate_factor_gru;

  /** @brief Whether inverse of Kronecker factors are available. */
  bool m_has_kronecker_inverse = false;

  /** @brief KFAC Compute interval. */
  size_t m_compute_interval;

  /** @brief distribute precondition gradient compute. */
  bool m_distribute_precondition_compute;

  /** @brief copy errors to a temporary matrix to increase overlap of compute
   * and communication. */
  bool m_enable_copy_errors;

  /** @brief copy activations to a temporary matrix to increase overlap of
   * compute and communication. */
  bool m_enable_copy_activations;

  /** @brief use eigen value decomposition for inversing the matrix. */
  bool m_use_eigen_decomposition;

  El::Matrix<double, El::Device::CPU> m_inverse_matrices_size;

  int m_global_inverse_buffer_size = 0, m_weight_matrices_buffer_size = 0;

  /** @brief vector for async communication reqs. */
  std::vector<kfac::ReqT> m_inverse_matrix_communication_reqs,
    m_weights_communication_reqs;

  /** @brief Profiling variables. */
  int m_time_span_inverse_comm = 0, m_time_span_inverse_send_recv = 0,
      m_time_span_forward_comm = 0, m_time_span_forward_comm_end = 0,
      m_time_span_backward_comm = 0, m_time_span_backward_comm_end = 0,
      m_time_span_precond_comm = 0, m_time_forward_pass = 0,
      m_time_backward_pass = 0, m_time_kfac = 0;

  std::vector<bool> m_use_KFAC_epoch;

}; // class KFAC

} // namespace lbann

/** @brief Build the KFAC training algorithm from a protobuf
 *         message.
 */
template <>
std::unique_ptr<lbann::KFAC>
lbann::make<lbann::KFAC>(google::protobuf::Message const& msg);

#endif // LBANN_EXECUTION_ALGORITHMS_KFAC_HPP_INCLUDED
