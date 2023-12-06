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
#ifndef LBANN_EXECUTION_ALGORITHMS_LTFB_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_LTFB_HPP_INCLUDED

#include "lbann/data_ingestion/data_coordinator.hpp"
#include "lbann/execution_algorithms/factory.hpp"
#include "lbann/execution_algorithms/training_algorithm.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/make_abstract.hpp"

#include "ltfb/execution_context.hpp"
#include "ltfb/meta_learning_strategy.hpp"
#include "ltfb/termination_criteria.hpp"

#include <google/protobuf/message.h>
#include <memory>

namespace lbann {

/** @class LTFB
 *  @brief An implementation of the LTFB training algorithm.
 *
 *  This is an example of a "meta-learning" training algorithm in
 *  which multiple models are trained in parallel -- one model per
 *  trainer participating in the `lbann_comm` object. Following local
 *  training, some postprocessing strategy is applied to further
 *  optimize the solution. In the case of "classical LTFB", the
 *  trainers in the communicator are paired off randomly and "compete"
 *  in tournaments. The winner of each tournament is returned from the
 *  postprocessing to either undergo further local training or to be
 *  returned from the training algorithm.
 *
 *  The salient thing to realize is that every local training will be
 *  followed by this postprocessing. Therefore, it is expected that
 *  the output of the postprocessing be "at least as good" (by some
 *  relevant metric) as the one that went in. If, say, you want to
 *  "randomize" your model in some way, and then do some training, and
 *  then do some other stuff, this class can certainly serve as a
 *  useful guide, but is not likely to be the out-of-the-box solution.
 */
class LTFB final : public TrainingAlgorithm
{
public:
  using TermCriteriaType = ltfb::LTFBTerminationCriteria;
  using ExeContextType = ltfb::LTFBExecutionContext;

public:
  /** @name Life-cycle management */
  ///@{
  /** @brief Construct LTFB from its component pieces.
   *  @param[in] name A string identifying this instance of LTFB.
   *  @param[in] local_training_algorithm The training algorithm to
   *             be used for (trainer-)local training.
   *  @param[in] meta_learning_strategy The postprocessing algorithm.
   *  @param[in] stopping_criteria When to stop the training
   *             algorithm.
   */
  LTFB(std::string name,
       std::unique_ptr<TrainingAlgorithm> local_training_algorithm,
       std::unique_ptr<ltfb::MetaLearningStrategy> meta_learning_strategy,
       ltfb::LTFBTerminationCriteria stopping_criteria,
       bool suppress_timer)
    : TrainingAlgorithm{std::move(name)},
      m_local_algo{std::move(local_training_algorithm)},
      m_meta_learning_strategy{std::move(meta_learning_strategy)},
      m_termination_criteria{std::move(stopping_criteria)},
      m_suppress_timer{suppress_timer}
  {}

  ~LTFB() noexcept = default;
  LTFB(LTFB const& other) = delete;
  LTFB& operator=(LTFB const&) = delete;
  LTFB(LTFB&&) = default;
  LTFB& operator=(LTFB&&) = default;
  ///@}
  /** @brief Queries */
  ///@{
  std::string get_type() const final { return "LTFB"; }
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
  ///@}
protected:
  /** @brief Covariant return-friendly implementation of
   *         `get_new_exection_context()`.
   */
  ltfb::LTFBExecutionContext* do_get_new_execution_context() const final
  {
    return new ltfb::LTFBExecutionContext();
  }

private:
  /** @brief The training algorithm for trainer-local training. */
  std::unique_ptr<TrainingAlgorithm> m_local_algo;

  /** @brief The strategy for postprocessing local training outputs. */
  std::unique_ptr<ltfb::MetaLearningStrategy> m_meta_learning_strategy;

  /** @brief The LTFB stopping criteria. */
  ltfb::LTFBTerminationCriteria m_termination_criteria;

  /** @brief Suppress timer output.
   *  @deprecated This is a temporary way to disable timer
   *              output. This will be more configurable in the
   *              future.
   */
  bool m_suppress_timer = false;
}; // class LTFB

} // namespace lbann

/** @brief Build the LTFB training algorithm from a protobuf
 *         message.
 */
template <>
std::unique_ptr<lbann::LTFB>
lbann::make<lbann::LTFB>(google::protobuf::Message const& msg);

#endif // LBANN_EXECUTION_ALGORITHMS_LTFB_HPP_INCLUDED
