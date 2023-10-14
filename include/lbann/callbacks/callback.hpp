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
//
// callback .hpp - Base class for LBANN callbacks
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_HPP_INCLUDED

#include "lbann/utils/description.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/summary.hpp"

#include <google/protobuf/message.h>

#include <algorithm>
#include <string>

// Forward-declare protobuf classes
namespace lbann_data {
class Callback;
}

/** @brief A utility macro for easily adding default-constructed sub-class
 *  builders.*/
#define LBANN_ADD_DEFAULT_CALLBACK_BUILDER(Class, FunctionName)                \
  inline std::unique_ptr<callback_base> FunctionName(                          \
    const google::protobuf::Message&,                                          \
    std::shared_ptr<lbann_summary> const&)                                     \
  {                                                                            \
    return std::make_unique<Class>();                                          \
  }

namespace cereal {
class access;
} // namespace cereal

namespace lbann {

// Forward-declarations
class Layer;
class model;
class trainer;
class weights;

/** @class callback_base
 *  @brief Base class for callbacks during training/testing.
 *
 *  The method of each callback is called at a given point during
 *  training or testing by the model. Implement whichever ones you
 *  care about.  Callbacks may be passed a lbann_summary instance,
 *  which they can use to log any relevant information.
 */
class callback_base
{
public:
  /** @name Constructors and destructor */
  ///@{

  /** @brief Initialize a callback with an optional batch interval
   */
  callback_base(int batch_interval = 1)
    : m_batch_interval(std::max(batch_interval, 1))
  {}
  callback_base(const callback_base&) = default;
  virtual ~callback_base() = default;

  ///@}
  /** @name Polymorphic copy */
  ///@{

  virtual callback_base* copy() const = 0;

  ///@}
  /** @name Modifiers */
  ///@{

  /** @brief Called once to set up the callback on the trainer
   */
  virtual void setup(trainer* t){};

  /** @brief Called once to set up the callback on the model
   *         (after all layers are set up).
   */
  virtual void setup(model* m){};

  ///@}
  /** @name Callback hooks */
  ///@{

  /** @brief Called at the beginning of model setup. */
  virtual void on_setup_begin(model* m) {}
  /** @brief Called at the end of setup. */
  virtual void on_setup_end(model* m) {}
  /** @brief Called at the beginning of training. */
  virtual void on_train_begin(model* m) {}
  /** @brief Called at the end of training. */
  virtual void on_train_end(model* m) {}
  /** @brief Called at the end of every phase (multiple epochs) in a
   *         layer-wise model training
   */
  virtual void on_phase_end(model* m) {}
  /** @brief Called at the beginning of each epoch. */
  virtual void on_epoch_begin(model* m) {}
  /** @brief Called immediate after the end of each epoch. */
  virtual void on_epoch_end(model* m) {}
  /** @brief Called at the beginning of a (mini-)batch. */
  virtual void on_batch_begin(model* m) {}
  /** @brief Called immediately after the end of a (mini-)batch. */
  virtual void on_batch_end(model* m) {}
  /** @brief Called at the beginning of testing. */
  virtual void on_test_begin(model* m) {}
  /** @brief Called immediately after the end of testing. */
  virtual void on_test_end(model* m) {}
  /** @brief Called at the beginning of validation. */
  virtual void on_validation_begin(model* m) {}
  /** @brief Called immediately after the end of validation. */
  virtual void on_validation_end(model* m) {}
  /** @brief Called when a model begins forward propagation. */
  virtual void on_forward_prop_begin(model* m) {}
  /** @brief Called when a layer begins forward propagation. */
  virtual void on_forward_prop_begin(model* m, Layer* l) {}
  /** @brief Called when a model ends forward propagation. */
  virtual void on_forward_prop_end(model* m) {}
  /** @brief Called when a layer ends forward propagation. */
  virtual void on_forward_prop_end(model* m, Layer* l) {}
  /** @brief Called when a model begins backward propagation. */
  virtual void on_backward_prop_begin(model* m) {}
  /** @brief Called when a layer begins backward propagation. */
  virtual void on_backward_prop_begin(model* m, Layer* l) {}
  /** @brief Called when a model ends backward propagation. */
  virtual void on_backward_prop_end(model* m) {}
  /** @brief Called when a layer ends backward propagation. */
  virtual void on_backward_prop_end(model* m, Layer* l) {}
  /** @brief Called when a model begins optimization. */
  virtual void on_optimize_begin(model* m) {}
  /** @brief Called when weights begins optimization. */
  virtual void on_optimize_begin(model* m, weights* w) {}
  /** @brief Called when a model ends optimization. */
  virtual void on_optimize_end(model* m) {}
  /** @brief Called when weights ends optimization. */
  virtual void on_optimize_end(model* m, weights* w) {}

  /** @brief Called at the beginning of a (mini-)batch evaluation
   *         (validation / testing).
   */
  virtual void on_batch_evaluate_begin(model* m) {}
  /** @brief Called at the end of a (mini-)batch evaluation
   *         (validation / testing).
   */
  virtual void on_batch_evaluate_end(model* m) {}
  /** @brief Called when a model begins forward propagation for
   *         evaluation (validation / testing).
   */
  virtual void on_evaluate_forward_prop_begin(model* m) {}
  /** @brief Called when a layer begins forward propagation for
   *         evaluation (validation / testing).
   */
  virtual void on_evaluate_forward_prop_begin(model* m, Layer* l) {}
  /** @brief Called when a model ends forward propagation for
   *         evaluation (validation / testing).
   */
  virtual void on_evaluate_forward_prop_end(model* m) {}
  /** @brief Called when a layer ends forward propagation for
   *         evaluation (validation / testing).
   */
  virtual void on_evaluate_forward_prop_end(model* m, Layer* l) {}

  ///@}
  /** @name Queries */
  ///@{

  /** @brief Return the batch interval. */
  int get_batch_interval() const { return m_batch_interval; }

  /** @brief Return this callback's name. */
  virtual std::string name() const = 0;

  /** @brief Human-readable description. */
  virtual description get_description() const;

  ///@}
  /** @name Serialization */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  /** @brief Write a protobuf description of the callback. */
  void write_proto(lbann_data::Callback& proto) const;

protected:
  /** @brief Add callback specific data to prototext */
  virtual void write_specific_proto(lbann_data::Callback& proto) const = 0;

  ///@}

  /** @brief Build a standard directory hierarchy including trainer ID.
   */
  std::string get_multi_trainer_path(const model& m,
                                     const std::string& root_dir);

  /** @brief Build a standard directory hierachy including trainer,
   * execution context, and model information (in that order).
   */
  std::string get_multi_trainer_ec_model_path(const model& m,
                                              const std::string& root_dir);

  /** @brief Build a standard directory hierachy including trainer,
   * model information in that order.
   */
  std::string get_multi_trainer_model_path(const model& m,
                                           const std::string& root_dir);

protected:
  /** @brief Copy-assignment operator.
   *
   *  Performs a shallow (pointer) copy of the summarizer.
   */
  callback_base& operator=(const callback_base&) = default;

protected:
  /** @todo Make callback data private */

  /** @brief Batch methods should once every this many steps. */
  int m_batch_interval;
};

} // namespace lbann

#endif // LBANN_CALLBACKS_CALLBACK_HPP_INCLUDED
