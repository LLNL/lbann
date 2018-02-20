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

#ifndef LBANN_OPTIMIZER_HPP
#define LBANN_OPTIMIZER_HPP

#include "lbann/utils/compiler_control.hpp"
#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/weights/weights.hpp"
#include <string>
#include <unordered_set>

namespace lbann {

/** Abstract optimizer. */
class optimizer {
 public:

  optimizer(lbann_comm* comm, DataType learning_rate = DataType(0));
  optimizer(const optimizer& other);
  optimizer& operator=(const optimizer& other);
  virtual ~optimizer();
  virtual optimizer* copy() const = 0;

  /** Get the optimizer name. */
  virtual std::string get_type() const = 0;
  /** Get a human-readable description of the optimizer. */
  virtual std::string get_description() const;

  /** Whether the optimizer has been set up. */
  inline bool is_initialized() const { return m_weights != nullptr; }

  /** Get weights being optimized. */
  weights& get_weights();
  /** Set weights being optimized. */
  void set_weights(weights& w) { m_weights = &w; }
  /** Get learning rate. */
  DataType get_learning_rate() const { return m_learning_rate; }
  /** Set learning rate. */
  void set_learning_rate(DataType learning_rate) {
    m_learning_rate = learning_rate;
  };

  /** Get gradient matrix. */
  const AbsDistMat& get_gradient();
#ifdef LBANN_HAS_CUDNN
  /** Get gradient matrix on GPU. */
  const cudnn::matrix& get_gradient_gpu();
#endif // LBANN_HAS_CUDNN

  /** Clear gradient matrix. */
  void clear_gradient();
  /** Add to the gradient matrix.
   *  If the optimizer has a cuDNN manager, the data is copied to GPUs
   *  and added to the GPU gradient matrix.
   */
  void add_to_gradient(const AbsDistMat& gradient,
                       DataType scale = DataType(1));
#ifdef LBANN_HAS_CUDNN
  /** Add to the GPU gradient matrix. */
  void add_to_gradient(const cudnn::matrix& gradient,
                       DataType scale = DataType(1));
#endif // LBANN_HAS_CUDNN

  /** Add to the gradient staging matrix.
   *  When the gradient is needed, an allreduce is applied over the
   *  redundant communicator of the staging matrix and the result is
   *  added to the gradient. If the optimizer has a cuDNN manager, the
   *  data is copied to GPUs and added to the GPU staging matrix.
   */
  void add_to_gradient_staging(const AbsDistMat& gradient,
                               DataType scale = DataType(1));
#ifdef LBANN_HAS_CUDNN
  /** Add to the GPU gradient staging matrix.
   *  When the gradient is needed, an allreduce is applied over all
   *  the GPUs in the redundant communicator of the staging matrix and
   *  the result is added to the gradient.
   */
  void add_to_gradient_staging(const cudnn::matrix& gradient,
                               DataType scale = DataType(1));
#endif // LBANN_HAS_CUDNN
  /** Start allreduce on the gradient staging matrix.
   *  If an allreduce is not needed or if it has already started, this
   *  function does nothing. This may call a non-blocking allreduce.
   */
  void start_gradient_staging_allreduce();

  /** Get number of gradient sources.
   *  This is the number of objects that contribute to the gradient
   *  but have not added their contributions yet.
   */
  int get_num_gradient_sources() const { return m_gradient_sources.size(); }
  /** Add a gradient source.
   *  Objects that depend on the weights being optimized and which
   *  contribute to the gradient should add themselves as a gradient
   *  source.
   */
  void add_gradient_source(const void* source);
  /** Remove a gradient source.
   *  Objects that contribute to the gradient should remove themselves
   *  as gradient sources when they add to the gradient. If there are
   *  no more gradient sources remaining, an allreduce is started on
   *  the gradient staging matrix.
   */
  void remove_gradient_source(const void* source);

  /** Setup optimizer. */
  virtual void setup(weights& w);

  /** Apply an optimization step. */
  void step();
  /** Perform the computation in an optimization step.
   *  It can be assumed that values and gradient are the same size and
   *  have the same matrix distribution.
   */
  virtual void step_compute(AbsDistMat& values,
                            const AbsDistMat& gradient) = 0;
#ifdef LBANN_HAS_CUDNN
  /** Perform the computation in an optimization step on GPU.
   *  The default implementation is to transfer data to CPU and call
   *  step_compute.
   */
  virtual void step_compute_gpu(cudnn::matrix& values,
                                const cudnn::matrix& gradient_d);
#endif // LBANN_HAS_CUDNN

  /** Get the time spent in step(). */
  double get_step_time() const { return m_step_time; }
  /** Reset stats counters. */
  virtual void reset_counters() {
    m_step_time = 0.0;
  }

  // For checkpointing
  virtual void set_states_on_host() {}
  virtual void set_states_on_device() {}

 protected:
  // For checkpointing
  void set_mat_state_on_host(AbsDistMat* state, const std::vector<DataType*>& state_d);
  void set_mat_state_on_device(AbsDistMat* state, std::vector<DataType*>& state_d);

 protected:

  /** LBANN communicator. */
  lbann_comm *m_comm;

  /** cuDNN manager. */
  cudnn::cudnn_manager* m_cudnn;

  /** Weights being optimized. */
  weights* m_weights;

  /** Learning rate. */
  DataType m_learning_rate;

  /** Gradient matrix. */
  AbsDistMat* m_gradient;
#ifdef LBANN_HAS_CUDNN
  /** GPU gradient matrix. */
  cudnn::matrix m_gradient_d;
#endif // LBANN_HAS_CUDNN

 private:

  /** Sources of gradient contributions.
   *  This set contains pointers to objects (i.e. layers and objective
   *  function terms) which depend on the weights being optimized and
   *  which contribute to the gradient. Objects should add themselves
   *  to the set as they request the weights and they should remove
   *  themselves as they add their gradient contribution. Once this
   *  set is empty, it is safe to perform an allreduce on the gradient
   *  staging matrix.
   */
  std::unordered_set<const void*> m_gradient_sources;

  /** Gradient staging matrix.
   *  When the gradient is needed, an allreduce is applied over the
   *  redundant communicator of the staging matrix and the result is
   *  added to the gradient matrix.
   */
  AbsDistMat* m_gradient_staging;
#ifdef LBANN_HAS_CUDNN
  /** GPU memory for gradient staging matrix.
   *  When the gradient is needed, an allreduce is applied over the
   *  GPUs and over the redundant communicator of the staging matrix
   *  and the result is added to the gradient matrix.
   */
  cudnn::matrix m_gradient_staging_d;
#endif // LBANN_HAS_CUDNN

  /** Whether the gradient staging matrix requires an allreduce. */
  bool m_gradient_allreduce_needed;
  /** Whether an allreduce on the gradient staging matrix has started. */
  bool m_gradient_allreduce_started;
  /** Whether an allreduce on the gradient staging matrix has been finished. */
  bool m_gradient_allreduce_finished;

  /** Running count of the time spent in step(). */
  double m_step_time = 0.0;

  /** The request for non-blocking allreduces. */
  Al::request m_gradient_allreduce_req;

//************************************************************************
// Checkpointing
//************************************************************************
 public:
  virtual bool save_to_checkpoint_shared(persist& p, std::string m_name);
  virtual bool load_from_checkpoint_shared(persist& p, std::string m_name);

  virtual bool save_to_checkpoint_distributed(persist& p, std::string m_name);
  virtual bool load_from_checkpoint_distributed(persist& p, std::string m_name);
};

} // namespace lbann

#endif // LBANN_OPTIMIZER_HPP
