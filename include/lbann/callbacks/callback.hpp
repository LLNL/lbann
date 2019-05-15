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
//
// lbann_callback .hpp - Base class for LBANN callbacks
////////////////////////////////////////////////////////////////////////////////

#ifndef __LBANN_CALLBACKS_CALLBACK_HPP_INCLUDED
#define __LBANN_CALLBACKS_CALLBACK_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/models/model.hpp"
#include "lbann/layers/layer.hpp"

namespace lbann {

/** @class lbann_callback
 *  @brief Base class for callbacks during training/testing.
 *
 *  The method of each callback is called at a given point during
 *  training or testing by the model. Implement whichever ones you
 *  care about.  Callbacks may be passed a lbann_summary instance,
 *  which they can use to log any relevant information.
 */
class lbann_callback {
public:

  /** @name Constructors and destructor */
  ///@{

  /** @brief Initialize a callback with an optional batch interval and
   *         summarizer.
   */
  lbann_callback(int batch_interval = 1,
                 lbann_summary *summarizer = nullptr) :
    m_batch_interval(std::max(batch_interval, 1)), m_summarizer(summarizer) {}
  lbann_callback(const lbann_callback&) = default;
  virtual ~lbann_callback() {}

  ///@}
  /** @name Polymorphic copy */
  ///@{

  virtual lbann_callback* copy() const = 0;

  ///@}
  /** @name Modifiers */
  ///@{

  void set_summarizer(lbann_summary *summarizer) {
    m_summarizer = summarizer;
  }

  /** @brief Called once to set up the callback (after all layers are
   *         set up).
   */
  virtual void setup(model *m) {};

  ///@}
  /** @name Callback hooks */
  ///@{

  /** @brief Called at the beginning of training. */
  virtual void on_train_begin(model *m) {}
  /** @brief Called at the end of training. */
  virtual void on_train_end(model *m) {}
  /** @brief Called at the end of every phase (multiple epochs) in a
   *         layer-wise model training
   */
  virtual void on_phase_end(model *m) {}
  /** @brief Called at the beginning of each epoch. */
  virtual void on_epoch_begin(model *m) {}
  /** @brief Called immediate after the end of each epoch. */
  virtual void on_epoch_end(model *m) {}
  /** @brief Called at the beginning of a (mini-)batch. */
  virtual void on_batch_begin(model *m) {}
  /** @brief Called immediately after the end of a (mini-)batch. */
  virtual void on_batch_end(model *m) {}
  /** @brief Called at the beginning of testing. */
  virtual void on_test_begin(model *m) {}
  /** @brief Called immediately after the end of testing. */
  virtual void on_test_end(model *m) {}
  /** @brief Called at the beginning of validation. */
  virtual void on_validation_begin(model *m) {}
  /** @brief Called immediately after the end of validation. */
  virtual void on_validation_end(model *m) {}
  /** @brief Called when a model begins forward propagation. */
  virtual void on_forward_prop_begin(model *m) {}
  /** @brief Called when a layer begins forward propagation. */
  virtual void on_forward_prop_begin(model *m, Layer *l) {}
  /** @brief Called when a model ends forward propagation. */
  virtual void on_forward_prop_end(model *m) {}
  /** @brief Called when a layer ends forward propagation. */
  virtual void on_forward_prop_end(model *m, Layer *l) {}
  /** @brief Called when a model begins backward propagation. */
  virtual void on_backward_prop_begin(model *m) {}
  /** @brief Called when a layer begins backward propagation. */
  virtual void on_backward_prop_begin(model *m, Layer *l) {}
  /** @brief Called when a model ends backward propagation. */
  virtual void on_backward_prop_end(model *m) {}
  /** @brief Called when a layer ends backward propagation. */
  virtual void on_backward_prop_end(model *m, Layer *l) {}
  /** @brief Called when a model begins optimization. */
  virtual void on_optimize_begin(model *m) {}
  /** @brief Called when weights begins optimization. */
  virtual void on_optimize_begin(model *m, weights *w) {}
  /** @brief Called when a model ends optimization. */
  virtual void on_optimize_end(model *m) {}
  /** @brief Called when weights ends optimization. */
  virtual void on_optimize_end(model *m, weights *w) {}

  /** @brief Called at the beginning of a (mini-)batch evaluation
   *         (validation / testing).
   */
  virtual void on_batch_evaluate_begin(model *m) {}
  /** @brief Called at the end of a (mini-)batch evaluation
   *         (validation / testing).
   */
  virtual void on_batch_evaluate_end(model *m) {}
  /** @brief Called when a model begins forward propagation for
   *         evaluation (validation / testing).
   */
  virtual void on_evaluate_forward_prop_begin(model *m) {}
  /** @brief Called when a layer begins forward propagation for
   *         evaluation (validation / testing).
   */
  virtual void on_evaluate_forward_prop_begin(model *m, Layer *l) {}
  /** @brief Called when a model ends forward propagation for
   *         evaluation (validation / testing).
   */
  virtual void on_evaluate_forward_prop_end(model *m) {}
  /** @brief Called when a layer ends forward propagation for
   *         evaluation (validation / testing).
   */
  virtual void on_evaluate_forward_prop_end(model *m, Layer *l) {}

  ///@}
  /** @name Queries */
  ///@{

  /** @brief Return the batch interval. */
  int get_batch_interval() const { return m_batch_interval; }

  /** @brief Return this callback's name. */
  virtual std::string name() const = 0;

  ///@}

protected:

  /** @brief Copy-assignment operator.
   *
   *  Performs a shallow (pointer) copy of the summarizer.
   */
  lbann_callback& operator=(const lbann_callback&) = default;

protected:
  /** @todo Make lbann_callback data private */

  /** @brief Batch methods should once every this many steps. */
  int m_batch_interval;
  /** @brief Optional summarizer for the callbacks to use. */
  lbann_summary *m_summarizer;
};

}  // namespace lbann

#endif  // __LBANN_CALLBACKS_CALLBACK_HPP_INCLUDED
