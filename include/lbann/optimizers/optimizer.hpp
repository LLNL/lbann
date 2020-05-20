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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZERS_OPTIMIZER_HPP_INCLUDED
#define LBANN_OPTIMIZERS_OPTIMIZER_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/compiler_control.hpp"
#ifdef LBANN_HAS_GPU
#include "lbann/utils/cuda.hpp"
#endif // LBANN_HAS_GPU
#include "lbann/utils/description.hpp"
#include "lbann/utils/exception.hpp"

#include <cereal/types/utility.hpp>

#include <memory>
#include <string>
#include <unordered_set>

namespace lbann {

/** @brief Status of values in objective function gradient. */
enum class optimizer_gradient_status {
  /** @brief Values can be accessed immediately. */
  ready,
  /** @brief Values have been cleared.
   *  @details Buffer must be zeroed out before accessing.
   */
  cleared,
  /** @brief Allreduce is needed before accessing values. */
  allreduce_needed,
  /** @brief Allreduce on values is in progress.
   *  @details Non-blocking allreduce must be synchronized before
   *  accessing.
   */
  allreduce_started,
};

/** @brief Human-readable string for status of gradient in optimizer. */
std::string to_string(optimizer_gradient_status status);

// Forward declarations
class persist;

/** @brief Abstract base class for gradient-based optimization algorithms.
 *
 *  Uses a variant of stochastic gradient descent to optimize the
 *  values in a @c weights instance. The weights values are
 *  iteratively adjusted to minimize an objective function. Each
 *  optimization step requires the objective function gradient
 *  w.r.t. the weights.
 */
class optimizer : public Cloneable<HasAbstractFunction<optimizer>> {
public:

  /** @name Constructors and Destructor */
  ///@{

  optimizer();
  virtual ~optimizer() = default;

  ///@}

  /** @brief Human-readable type name. */
  virtual std::string get_type() const = 0;
  /** @brief Human-readable description. */
  virtual description get_description() const;

  /** @name Gradient update management */
  ///@{

  /** @brief Zero out the objective function gradient w.r.t. the weights. */
  virtual void clear_gradient() = 0;

  /** @brief Objects that are expected to contribute to the gradient. */
  El::Int get_num_gradient_sources() const;
  /** @brief Register a gradient source.
   *
   *  Any object that uses the weights and influences the objective
   *  function is expected to contribute to the objective function
   *  gradient. These objects should register themselves during
   *  forward prop.
   */
  void add_gradient_source(const void* source);

  /** @brief Unregister a gradient source.
   *
   *  When an object adds its contribution to the objective function
   *  gradient during back prop, it should unregister itself. If there
   *  are no more gradient sources remaining, a non-blocking allreduce
   *  will be launched on the gradient, if needed.
   */
  void remove_gradient_source(const void* source);

  /** @brief Perform optimization step. */
  virtual void step() = 0;

  ///@}
  /** @brief Communicator access */
  ///@{

  /** @brief Access LBANN communicator. */
  lbann_comm& get_comm() { return *m_comm; }
  /** @brief Access LBANN communicator. */
  const lbann_comm& get_comm() const { return *m_comm; }

  ///@}
  /** @brief Statistics access and management */
  ///@{

  /** @brief Time spent in optimization step. */
  EvalType get_step_time() const { return m_step_time; }

  /** @brief Reset stats counters. */
  virtual void reset_counters() { m_step_time = 0; }

  ///@}
  /** @name Checkpointing */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar) {
    // Do not save the optimizer's step time
  }

  virtual bool save_to_checkpoint_shared(persist& p, std::string m_name) = 0;
  virtual bool load_from_checkpoint_shared(persist& p, std::string m_name) = 0;
  virtual bool save_to_checkpoint_distributed(persist& p, std::string m_name) = 0;
  virtual bool load_from_checkpoint_distributed(persist& p, std::string m_name) = 0;
  ///@}

protected:
  /** @brief Copy construct/copy assign */
  optimizer(const optimizer& other);
  optimizer& operator=(const optimizer& other);

  /** @brief Return the current gradient status */
  optimizer_gradient_status get_gradient_status() const { return m_gradient_status; }

  void set_gradient_status(const optimizer_gradient_status status) { m_gradient_status = status; }

  std::unordered_set<const void*>& get_gradient_sources() { return m_gradient_sources; }

  void set_comm(lbann_comm& comm) { m_comm = &comm; }

  void set_step_time(EvalType time) { m_step_time = time; }

  void inc_step_time(EvalType time) { m_step_time += time; }

private:

  /** @brief Begin the allreduce on the gradient values. */
  virtual void start_gradient_allreduce() = 0;

private:

  /** @brief LBANN communicator. */
  lbann_comm* m_comm;

  /** @brief Sources of gradient contributions.
   *
   *  This set contains pointers to objects (e.g. layers and objective
   *  function terms) that contribute to the objective function
   *  gradient. Objects should register themselves as they use the
   *  weights during forward prop and unregister themselves as they
   *  add their gradient contributions. Once this set is empty, it is
   *  safe to launch a non-blocking allreduce on the gradient, if
   *  needed.
   */
  std::unordered_set<const void*> m_gradient_sources;

  /** @brief Status of values in objective function gradient. */
  optimizer_gradient_status m_gradient_status = optimizer_gradient_status::cleared;

  /** @brief Time spent in optimization step. */
  EvalType m_step_time = 0;

};

} // namespace lbann

#endif // LBANN_OPTIMIZERS_OPTIMIZER_HPP_INCLUDED
