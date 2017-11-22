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
// lbann_callback .hpp - Base class for LBANN callbacks
////////////////////////////////////////////////////////////////////////////////

#ifndef __LBANN_CALLBACKS_CALLBACK_HPP_INCLUDED
#define __LBANN_CALLBACKS_CALLBACK_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/models/model.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/layers/activations/activation.hpp"
#include "lbann/layers/io/io_layer.hpp"
#include "lbann/layers/io/input/input_layer.hpp"
#include "lbann/layers/io/target/target_layer.hpp"
#include "lbann/layers/learning/learning.hpp"
#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/layers/transform/transform.hpp"

namespace lbann {

/**
 * Base class for callbacks during training/testing.
 * The method of each callback is called at a given point during training or
 * testing by the model. Implement whichever ones you care about.
 * Callbacks may be passed a lbann_summary instance, which they can use to log
 * whatever relevant information they need to.
 */
class lbann_callback {
 public:
  /** Initialize a callback with an optional batch interval and summarizer. */
  lbann_callback(int batch_interval = 1,
                 lbann_summary *summarizer = nullptr) :
    m_batch_interval(std::max(batch_interval, 1)), m_summarizer(summarizer) {}
  lbann_callback(const lbann_callback&) = default;
  lbann_callback& operator=(const lbann_callback&) = default;
  virtual ~lbann_callback() {}
  virtual lbann_callback* copy() const = 0;
  void set_summarizer(lbann_summary *summarizer) {
    m_summarizer = summarizer;
  }
  /** Called once to set up the callback (after all layers are set up). */
  virtual void setup(model *m) {};
  /** Called at the beginning of training. */
  virtual void on_train_begin(model *m) {}
  /** Called at the end of training. */
  virtual void on_train_end(model *m) {}
  /** Called at the end of every phase (multiple epochs) in a layer-wise model training */
  virtual void on_phase_end(model *m) {}
  /** Called at the beginning of each epoch. */
  virtual void on_epoch_begin(model *m) {}
  /** Called at the end of each epoch. */
  virtual void on_epoch_end(model *m) {}
  /** Called at the beginning of a (mini-)batch. */
  virtual void on_batch_begin(model *m) {}
  /** Called at the end of a (mini-)batch. */
  virtual void on_batch_end(model *m) {}
  /** Called at the beginning of testing. */
  virtual void on_test_begin(model *m) {}
  /** Called at the end of testing. */
  virtual void on_test_end(model *m) {}
  /** Called at the beginning of validation. */
  virtual void on_validation_begin(model *m) {}
  /** Called at the end of validation. */
  virtual void on_validation_end(model *m) {}
  /** Called when a model begins forward propagation. */
  virtual void on_forward_prop_begin(model *m) {}
  /** Called when a layer begins forward propagation. */
  virtual void on_forward_prop_begin(model *m, Layer *l) {}
  /** Called when a model ends forward propagation. */
  virtual void on_forward_prop_end(model *m) {}
  /** Called when a layer ends forward propagation. */
  virtual void on_forward_prop_end(model *m, Layer *l) {}
  /** Called when a model begins backward propagation. */
  virtual void on_backward_prop_begin(model *m) {}
  /** Called when a layer begins backward propagation. */
  virtual void on_backward_prop_begin(model *m, Layer *l) {}
  /** Called when a model ends backward propagation. */
  virtual void on_backward_prop_end(model *m) {}
  /** Called when a layer ends backward propagation. */
  virtual void on_backward_prop_end(model *m, Layer *l) {}

  /** Called at the beginning of a (mini-)batch evaluation (validation / testing). */
  virtual void on_batch_evaluate_begin(model *m) {}
  /** Called at the end of a (mini-)batch evaluation (validation / testing). */
  virtual void on_batch_evaluate_end(model *m) {}
  /** Called when a model begins forward propagation for evaluation (validation / testing). */
  virtual void on_evaluate_forward_prop_begin(model *m) {}
  /** Called when a layer begins forward propagation for evaluation (validation / testing). */
  virtual void on_evaluate_forward_prop_begin(model *m, Layer *l) {}
  /** Called when a model ends forward propagation for evaluation (validation / testing). */
  virtual void on_evaluate_forward_prop_end(model *m) {}
  /** Called when a layer ends forward propagation for evaluation (validation / testing). */
  virtual void on_evaluate_forward_prop_end(model *m, Layer *l) {}

  /** Return the batch interval. */
  int get_batch_interval() const { return m_batch_interval; }

  /** Return this callback's name. */
  virtual std::string name() const = 0;

 protected:
  /** Batch methods should once every this many steps. */
  int m_batch_interval;
  /** Optional summarizer for the callbacks to use. */
  lbann_summary *m_summarizer;
};

}  // namespace lbann

#endif  // __LBANN_CALLBACKS_CALLBACK_HPP_INCLUDED
