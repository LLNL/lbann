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
#ifndef LBANN_EXECUTION_ALGORITHMS_KFAC_EXECUTION_CONTEXT_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_KFAC_EXECUTION_CONTEXT_HPP_INCLUDED

#include "lbann/execution_contexts/execution_context.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"
#include <memory>
#include <string>

namespace lbann {
namespace kfac {

/** @class ExecutionContext
 *  @brief The execution context for an KFAC algorithm.
 */
class ExecutionContext final : public lbann::execution_context
{
public:
  /** Constructor. */
  ExecutionContext(size_t mini_batch_size);
  /** Destructor. */
  ~ExecutionContext() = default;

  /** Copy constructor -- deleted. */
  ExecutionContext(const ExecutionContext& other) = delete;
  /** Copy assignment operator -- deleted. */
  ExecutionContext& operator=(const ExecutionContext& other) = delete;

  /** Get a "clean" execution_context of the same type. */
  std::unique_ptr<lbann::execution_context> get_new() const override;

  /** @brief Get a string identifying the type of execution context.
   *  @details Should match the training algorithm.
   *  @todo Absorb completely into `get_state_string()`.
   */
  std::string get_type() const override;

  /** @brief Return the state of the execution context as a string */
  std::string get_state_string() const noexcept override;

  /** @brief Return execution context for SGD-family training algorithm. */
  inline sgd_execution_context& get_sgd_execution_context() noexcept
  {
    return m_sgd_execution_context;
  }

  /** @name Checkpointing and Serialization */
  ///@{

  /** Archive for checkpoint and restart */
  template <class Archive> void serialize(Archive& ar);

  /** @brief Checkpoint exection_context to a shared checkpoint. */
  void save_to_checkpoint_shared(persist& p) override;
  /** @brief Restore execution_context from a shared checkpoint. */
  void load_from_checkpoint_shared(persist& p) override;
  /** @brief Checkpoint exection_context to a distributed checkpoint. */
  void save_to_checkpoint_distributed(persist& p) override;
  /** @brief Restore execution_context from a distributed checkpoint. */
  void load_from_checkpoint_distributed(persist& p) override;
  ///@}

private:

  sgd_execution_context m_sgd_execution_context;

}; // class ExecutionContext

} // namespace kfac
} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_KFAC_EXECUTION_CONTEXT_HPP_INCLUDED
