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
#ifndef LBANN_EXECUTION_ALGORITHMS_KFAC_EXECUTION_CONTEXT_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_KFAC_EXECUTION_CONTEXT_HPP_INCLUDED

#include "lbann/execution_algorithms/execution_context.hpp"
#include "lbann/execution_algorithms/kfac/kfac_block.hpp"
#include "lbann/execution_algorithms/kfac/kfac_util.hpp"
#include "lbann/execution_algorithms/sgd_execution_context.hpp"
#include <memory>
#include <string>

// Forward declarations
namespace lbann {
class KFAC;
template <El::Device Device>
class kfac_block;
class model;
} // namespace lbann

namespace lbann {
namespace kfac {

// Typedefs
#ifdef LBANN_HAS_GPU
constexpr El::Device Device = El::Device::GPU;
#else
constexpr El::Device Device = El::Device::CPU;
#endif // LBANN_HAS_GPU

/** @class ExecutionContext
 *  @brief The execution context for an KFAC algorithm.
 */
class KFACExecutionContext final : public lbann::ExecutionContext
{
public:
  friend class ::lbann::KFAC;

  /** Constructor. */
  KFACExecutionContext(double damping_act,
                       double damping_err,
                       double damping_bn_act,
                       double damping_bn_err);
  /** Destructor. */
  ~KFACExecutionContext() = default;

  /** Copy constructor -- deleted. */
  KFACExecutionContext(const KFACExecutionContext& other) = delete;
  /** Copy assignment operator -- deleted. */
  KFACExecutionContext& operator=(const KFACExecutionContext& other) = delete;

  /** Get a "clean" execution_context of the same type. */
  std::unique_ptr<lbann::ExecutionContext> get_new() const override;

  /** @brief Get a string identifying the type of execution context.
   *  @details Should match the training algorithm.
   *  @todo Absorb completely into `get_state_string()`.
   */
  std::string get_type() const override;

  /** @brief Return the state of the execution context as a string */
  std::string get_state_string() const noexcept override;

  /** @brief Return execution context for SGD-family training algorithm. */
  inline SGDExecutionContext& get_sgd_execution_context() noexcept
  {
    return m_sgd_execution_context;
  }

  /** @brief Gets the Kronecker factor matrix of a FC layer.
   *  The same key is tied with the same matrix instance. */
  El::Matrix<DataType, Device>& get_workspace_matrix(const std::string& key,
                                                     const size_t height,
                                                     const size_t width);

  /** @name Checkpointing and Serialization */
  ///@{

  /** Archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  /** @brief Checkpoint exection_context to a shared checkpoint. */
  void save_to_checkpoint_shared(persist& p) override;
  /** @brief Restore execution_context from a shared checkpoint. */
  void load_from_checkpoint_shared(persist& p) override;
  /** @brief Checkpoint exection_context to a distributed checkpoint. */
  void save_to_checkpoint_distributed(persist& p) override;
  /** @brief Restore execution_context from a distributed checkpoint. */
  void load_from_checkpoint_distributed(persist& p) override;
  ///@}

  void print_workspace_size(model& model);

private:
  SGDExecutionContext m_sgd_execution_context;

  /** @brief The current damping values. */
  double m_damping_act, m_damping_err, m_damping_bn_act, m_damping_bn_err;

  /** @brief The current update interval. */
  size_t m_update_interval;

  /** @brief K-FAC per-layer blocks. */
  std::vector<std::shared_ptr<kfac_block<Device>>> m_blocks;

  /** @brief Workspace matrices that are used by m_blocks. */
  std::unordered_map<std::string, El::Matrix<DataType, Device>> m_workspace;

}; // class ExecutionContext

} // namespace kfac
} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_KFAC_EXECUTION_CONTEXT_HPP_INCLUDED
