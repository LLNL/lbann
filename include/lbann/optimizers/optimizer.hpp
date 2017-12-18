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
//
// lbann_optimizer .hpp .cpp - Abstract optimizer class
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_OPTIMIZER_HPP
#define LBANN_OPTIMIZER_HPP

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/weights/weights.hpp"
#include <string>

namespace lbann {

/** Abstract optimizer. */
class optimizer {
 public:

  /** Constructor. */
  optimizer(lbann_comm* comm, DataType learning_rate = DataType(0));

  /** Copy constructor. */
  optimizer(const optimizer& other);
  /** Copy assignment operator. */
  optimizer& operator=(const optimizer& other);
  /** Destructor. */
  virtual ~optimizer();
  /** Create a copy of the optimizer. */
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

  /** Get gradient matrix.
   *  The gradient is accumulated on the CPU.
   */
  AbsDistMat& get_gradient();
#ifdef LBANN_HAS_CUDNN
  /** Get gradient matrix on GPU.
   *  The gradient is accumulated on the GPU.
   */
  std::vector<DataType*> get_gradient_gpu();
#endif // LBANN_HAS_CUDNN

  /** Clear gradient matrix. */
  void clear_gradient();
  /** Add to the gradient matrix. */
  void add_to_gradient(const AbsDistMat& gradient,
                       DataType scale = DataType(1));
  /** Allreduce and add to gradient matrix.
   *  The input is added to a staging matrix. When the gradient is
   *  needed, an allreduce is applied over the redundant communicator
   *  of the gradient matrix and the result is added to the gradient.
   */
  void allreduce_and_add_to_gradient(const AbsDistMat& gradient,
                                     DataType scale = DataType(1));
#ifdef LBANN_HAS_CUDNN
  /** Add to the gradient matrix on GPU. */
  void add_to_gradient_gpu(std::vector<DataType*>& gradient,
                           DataType scale = DataType(1));
  /** Allreduce and add to gradient matrix on GPU.
   *  The input is added to a staging matrix. When the gradient is
   *  needed, an allreduce is applied over the redundant communicator
   *  of the gradient matrix and the result is added to the gradient.
   */
  void allreduce_and_add_to_gradient_gpu(std::vector<DataType*>& gradient,
                                         DataType scale = DataType(1));
#endif // LBANN_HAS_CUDNN

  /** Setup optimizer. */
  virtual void setup(weights& w);

  /** Apply an optimization step. */
  void step();
  /** Perform the computation in an optimization step.
   *  It can be assumed that values and gradient are the same size and
   *  have the same matrix distribution.
   */
  virtual void step_compute(AbsDistMat& values, const AbsDistMat& gradient) = 0;
#ifdef LBANN_HAS_CUDNN
  /** Perform the computation in an optimization step on GPU.
   *  The default implementation is to transfer data to CPU and call
   *  step_compute.
   */
  virtual void step_compute_gpu(std::vector<DataType*> values_d,
                                std::vector<DataType*> gradient_d);
#endif // LBANN_HAS_CUDNN

  /** Get the time spent in step(). */
  double get_step_time() const { return m_step_time; }
  /** Reset stats counters. */
  virtual void reset_counters() {
    m_step_time = 0.0;
  }

 protected:

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
  /** GPU memory for gradient matrix. */
  std::vector<DataType*> m_gradient_d;
#endif // LBANN_HAS_CUDNN

 private:

  /** Whether the CPU gradient matrix is non-zero. */
  bool m_cpu_gradient_is_nonzero;
  /** Whether the CPU staging matrix is non-zero. */
  bool m_cpu_staging_is_nonzero;
  /** Allreduce staging matrix.
   *  When the gradient is needed, an allreduce is applied over the
   *  redundant communicator of the staging matrix and the result is
   *  added to the gradient matrix.
   */
  AbsDistMat* m_staging;
#ifdef LBANN_HAS_CUDNN
  /** Whether the GPU gradient matrix is non-zero. */
  bool m_gpu_gradient_is_nonzero;
  /** Whether the GPU staging matrix is non-zero. */
  bool m_gpu_staging_is_nonzero;
  /** GPU memory for gradient staging matrix.
   *  When the gradient is needed, an allreduce is applied over the
   *  GPUs and over the redundant communicator of the staging matrix
   *  and the result is added to the gradient matrix.
   */
  std::vector<DataType*> m_staging_d;
#endif // LBANN_HAS_CUDNN

  /** Running count of the time spent in step(). */
  double m_step_time = 0.0;

//************************************************************************
// Checkpointing
//************************************************************************
 public:
  virtual bool save_to_checkpoint_shared(persist& p, std::string m_name);
  virtual bool load_from_checkpoint_shared(persist& p, std::string m_name);
};

} // namespace lbann

#endif // LBANN_OPTIMIZER_HPP
