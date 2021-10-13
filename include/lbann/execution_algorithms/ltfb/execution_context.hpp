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
#ifndef LBANN_EXECUTION_ALGORITHMS_LTFB_EXECUTION_CONTEXT_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_LTFB_EXECUTION_CONTEXT_HPP_INCLUDED

#include "lbann/execution_contexts/execution_context.hpp"
#include <memory>
#include <string>

namespace lbann {
namespace ltfb {

/** @class ExecutionContext
 *  @brief The execution context for an LTFB algorithm.
 *
 *  This class needs to keep track of the "local algorithm" state as
 *  well. I'm not sure if the stopping criteria for the local
 *  algorithm belongs here or as part of the TerminationCriteria
 *  object. My inclination is that it really doesn't matter and either
 *  would work.
 */
class LTFBExecutionContext final : public lbann::ExecutionContext
{
public:
  LTFBExecutionContext() = default;
  ~LTFBExecutionContext() = default;

  /** Get a "clean" execution_context of the same type. */
  std::unique_ptr<lbann::ExecutionContext> get_new() const override
  {
    return std::make_unique<LTFBExecutionContext>();
  }

  /** @brief Get a string identifying the type of execution context.
   *  @details Should match the training algorithm.
   *  @todo Absorb completely into `get_state_string()`.
   */
  std::string get_type() const override { return "ltfb"; }

  /** @brief Return the state of the execution context as a string */
  std::string get_state_string() const noexcept override
  {
    return build_string(this->get_type(), ".step.", this->get_step());
  }

  /** @name Checkpointing and Serialization */
  ///@{

  /** Archive for checkpoint and restart */
  template <class Archive> void serialize(Archive& ar);

  /** @brief Checkpoint exection_context to a shared checkpoint. */
  void save_to_checkpoint_shared(persist& p) override
  {
    LBANN_ERROR("TODO: Not yet implemented.");
  }
  /** @brief Restore execution_context from a shared checkpoint. */
  void load_from_checkpoint_shared(persist& p) override
  {
    LBANN_ERROR("TODO: Not yet implemented.");
  }
  /** @brief Checkpoint exection_context to a distributed checkpoint. */
  void save_to_checkpoint_distributed(persist& p) override
  {
    LBANN_ERROR("TODO: Not yet implemented.");
  }
  /** @brief Restore execution_context from a distributed checkpoint. */
  void load_from_checkpoint_distributed(persist& p) override
  {
    LBANN_ERROR("TODO: Not yet implemented.");
  }
  ///@}

}; // class ExecutionContext

} // namespace ltfb
} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_LTFB_EXECUTION_CONTEXT_HPP_INCLUDED
