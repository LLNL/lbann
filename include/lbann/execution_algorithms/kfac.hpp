////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/execution_algorithms/factory.hpp"
#include "lbann/execution_algorithms/training_algorithm.hpp"
#include "lbann/execution_algorithms/kfac/execution_context.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"
#include "lbann/models/directed_acyclic_graph.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/make_abstract.hpp"

#include <google/protobuf/message.h>
#include <memory>

namespace lbann {

/** @class KFAC
 *  @brief An implementation of the KFAC second-order optimization algorithm
 */
class KFAC final : public Cloneable<KFAC, training_algorithm>
{
  using BaseType = Cloneable<KFAC, training_algorithm>;

public:
  using TermCriteriaType = sgd_termination_criteria;
  using ExeContextType = kfac::ExecutionContext;

public:
  /** @name Life-cycle management */
  ///@{
  /** @brief Construct KFAC from its component pieces.
   */
  KFAC(std::string name,
       std::unique_ptr<TermCriteriaType> stop);

  KFAC(KFAC const& other);
  KFAC& operator=(const KFAC& other);
  ~KFAC() noexcept = default;
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
  void apply(execution_context& context,
             model& m,
             data_coordinator& dc,
             execution_mode mode) final;
  /** @brief Train a model using KFAC. */
  void train(ExeContextType& c,
             model& model,
             data_coordinator& dc,
             TermCriteriaType const& term);
  ///@}

protected:

  /** @brief Train model on one step / mini-batch of an SGD forward pass */
  bool train_mini_batch(
    ExeContextType& c,
    model& model,
    data_coordinator& dc);

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
  kfac::ExecutionContext* do_get_new_execution_context() const final;

private:

  /** @brief The KFAC stopping criteria. */
  std::unique_ptr<TermCriteriaType> m_stopping_criteria;

}; // class KFAC

} // namespace lbann

/** @brief Build the KFAC training algorithm from a protobuf
 *         message.
 */
template <>
std::unique_ptr<lbann::KFAC>
lbann::make<lbann::KFAC>(google::protobuf::Message const& msg);

#endif // LBANN_EXECUTION_ALGORITHMS_KFAC_HPP_INCLUDED
