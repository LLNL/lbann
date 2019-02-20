////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#include <string>
#include <memory>
#include <unordered_set>
#include "lbann/utils/compiler_control.hpp"
#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/weights/weights.hpp"
#ifdef LBANN_HAS_GPU
#include "lbann/utils/cuda.hpp"
#endif // LBANN_HAS_GPU

namespace lbann {

/** Status of values in objective function gradient. */
enum class optimizer_gradient_status {
  /** Values can be accessed immediately. */
  ready,
  /** @brief Values have been cleared.
   *  @detailed Buffer must be zeroed out before accessing.
   */
  cleared,
  /** Allreduce is needed before accessing values. */
  allreduce_needed,
  /** @brief Allreduce on values is in progress.
   *  @detailed Non-blocking allreduce must be synchronized before
   *  accessing.
   */
  allreduce_started
};

/** Human-readable string for status of gradient in optimizer. */
std::string to_string(optimizer_gradient_status status);

// Forward declarations
class weights;
class persist;

/** Abstract base class for gradient-based optimization algorithms.
 *
 *  Uses a variant of stochastic gradient descent to optimize the
 *  values in a @c weights instance. The weights values are
 *  iteratively adjusted to minimize an objective function. Each
 *  optimization step requires the objective function gradient
 *  w.r.t. the weights.
 */
class optimizer {
public:

  optimizer(lbann_comm* comm, DataType learning_rate = 0);
  optimizer(const optimizer& other);
  optimizer& operator=(const optimizer& other);
  virtual ~optimizer() = default;

  /** Create a copy of the class instance.
   *
   *  The caller is responsible for deallocating the returned object.
   */
  virtual optimizer* copy() const = 0;

  /** Human-readable type name. */
  virtual std::string get_type() const = 0;
  /** Human-readable description. */
  virtual description get_description() const;

  /** Weights being optimized. */
  weights& get_weights();
  /** Weights being optimized. */
  const weights& get_weights() const;
  /** Weights being optimized. */
  void set_weights(weights* w) { m_weights = w; }

  /** Objective function gradient w.r.t. the weights.
   *
   *  An allreduce may be launched and/or synchronized if needed.
   */
  AbsDistMat& get_gradient();

  /** Add to the objective function gradient w.r.t. the weights. */
  void add_to_gradient(const AbsDistMat& gradient,
                       DataType scale = DataType(1),
                       bool allreduce_needed = false);
  /** Zero out the objective function gradient w.r.t. the weights. */
  void clear_gradient();

  /** Objects that are expected to contribute to the gradient. */
  El::Int get_num_gradient_sources() const;
  /** Register a gradient source.
   *
   *  Any object that uses the weights and influences the objective
   *  function is expected to contribute to the objective function
   *  gradient. These objects should register themselves during
   *  forward prop.
   */
  void add_gradient_source(const void* source);
  /** Unregister a gradient source.
   *
   *  When an object adds its contribution to the objective function
   *  gradient during back prop, it should unregister itself. If there
   *  are no more gradient sources remaining, a non-blocking allreduce
   *  will be launched on the gradient, if needed.
   */
  void remove_gradient_source(const void* source);

  /** Must be called before training.
   *
   *  @param w Weights being optimized. If null, no change is made to
   *  the weights.
   */
  virtual void setup(weights* w = nullptr);

  /** Optimization step. */
  void step();

  /** LBANN communicator. */
  lbann_comm& get_comm() { return *m_comm; }
  /** LBANN communicator. */
  const lbann_comm& get_comm() const { return *m_comm; }

  /** Scaling factor for optimization step sizes. */
  DataType get_learning_rate() const;
  /** Scaling factor for optimization step sizes. */
  void set_learning_rate(DataType learning_rate);

  /** Time spent in optimization step. */
  EvalType get_step_time() const { return m_step_time; }
  /** Reset stats counters. */
  virtual void reset_counters() { m_step_time = 0; }

protected:

  /** Computation for an optimization step on CPU.
   *
   *  @c values and @gradient can be assumed to have the same
   *  distribution.
   */
  virtual void step_compute(AbsDistMat& values,
                            const AbsDistMat& gradient) = 0;
#ifdef LBANN_HAS_CUDA
  /** Computation for an optimization step on GPU.
   *
   *  The default implementation is to throw an exception. @c values
   *  and @gradient can be assumed to have the same distribution.
   */
  virtual void step_compute_gpu(AbsDistMat& values,
                                const AbsDistMat& gradient);
#endif // LBANN_HAS_CUDA

private:

  /** LBANN communicator. */
  lbann_comm* m_comm;

  /** Weights being optimized. */
  weights* m_weights = nullptr;

  /** Objective function gradient w.r.t. weights. */
  std::unique_ptr<AbsDistMat> m_gradient;

  /** Workspace matrix.
   *
   *  Helps ensure gradient contributions are in the right
   *  distribution. Most of the time, this should just be a matrix
   *  view.
   */
  std::unique_ptr<AbsDistMat> m_gradient_v;

  /** Sources of gradient contributions.
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

  /** Status of values in objective function gradient. */
  optimizer_gradient_status m_gradient_status = optimizer_gradient_status::cleared;

  /** Communication request object for gradient allreduce.
   *
   *  Used to synchronize non-blocking allreduce.
   */
  Al::request m_gradient_allreduce_req;

  /** Scaling factor for optimization step sizes.
   *
   *  This is not used by the base optimizer class, but is currently
   *  used by all derived optimizer classes. There are several cases
   *  where it is convenient to expose this in the base class,
   *  e.g. for variable learning rate schedules.
   *  @todo Consider moving this to the derived classes.
   */
  DataType m_learning_rate;

  /** Time spent in optimization step. */
  EvalType m_step_time = 0;

  /** Launch non-blocking allreduce on the gradient, if needed.
   *
   *  Does nothing if an allreduce is not needed or has already been
   *  started.
   */
  void start_gradient_allreduce();

  /** Synchronize non-blocking allreduce on the gradient, if needed.
   *
   *  Does nothing if an allreduce isn't needed. Throws an exception
   *  if an allreduce is needed but hasn't been started.
   */
  void finish_gradient_allreduce();

public:

  // ===========================================
  // Checkpointing
  // ===========================================
  virtual bool save_to_checkpoint_shared(persist& p, std::string m_name);
  virtual bool load_from_checkpoint_shared(persist& p, std::string m_name);
  virtual bool save_to_checkpoint_distributed(persist& p, std::string m_name);
  virtual bool load_from_checkpoint_distributed(persist& p, std::string m_name);

};

} // namespace lbann

#endif // LBANN_OPTIMIZERS_OPTIMIZER_HPP_INCLUDED
