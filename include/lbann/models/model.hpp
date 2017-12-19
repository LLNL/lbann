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
// lbann_model .hpp .cpp - Abstract class for neural network training models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_HPP
#define LBANN_MODEL_HPP

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/metrics/metric.hpp"
#include "lbann/weights/weights.hpp"
#include "lbann/optimizers/optimizer.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace lbann {

// Forward-declare this.
class lbann_callback;

/** Base class for LBANN models. */
class model {
 public:

  /** Constructor. */
  model(lbann_comm *comm,
        int mini_batch_size,
        objective_function *obj_fn,
        optimizer* default_optimizer = nullptr);

  /** Copy constructor. */
  model(const model& other);
  /** Copy assignment operator. */
  model& operator=(const model& other);
  /** Destructor. */
  virtual ~model();
  /** Copy model. */
  virtual model* copy() const = 0;

  /** Return the model's name. */
  virtual std::string name() const = 0;

  /** Set up the model. */
  virtual void setup();

  /** Add layer to model. */
  virtual void add_layer(Layer *layer);

  /** Add weights to model. */
  void add_weights(weights *w);

  /** Register a new callback for the model. */
  void add_callback(lbann_callback *cb);

  /** Get the list of callbacks for the model. */
  virtual std::vector<lbann_callback*>& get_callbacks() {
    return m_callbacks;
  }

  /** Register a new metric for the model. */
  void add_metric(metric *m);

  /** Construct an instance of the default optimizer.
   *  If there is no default optimizer, a null pointer is returned.
   */
  optimizer* create_optimizer() const;

  /** Return the model's objective function. */
  objective_function* get_objective_function() const {
    return m_objective_function;
  }

  /** Return the model's metrics. */
  virtual const std::vector<metric *>& get_metrics() const {
    return m_metrics;
  }

  /** Set the model's layers. */
  void set_layers(std::vector<Layer *>& layers);

  /** Return the model's layers. */
  virtual const std::vector<Layer *>& get_layers() const { return m_layers; }

  /** Replace the model's weights. */
  void replace_weights(std::vector<weights *>& w);

  /** Return the model's weights. */
  const std::vector<weights *>& get_weights() const { return m_weights; }

  /** Get the model's comm. */
  inline lbann_comm *get_comm() const {
    return m_comm;
  }

  /** Get the current epoch for the model. */
  inline int get_cur_epoch() const {
    return m_current_epoch;
  }
  /** Get the current step for the model. */
  inline int get_cur_step() const {
    return m_current_step;  /// @todo This should be renamed to get_cur_training step and replaced with one that returns the current based on execution mode
  }

  /** Get the current validation step for the model. */
  inline int get_cur_validation_step() const {
    return m_current_validation_step;
  }
  /** Get the current testing step for the model. */
  inline int get_cur_testing_step() const {
    return m_current_testing_step;
  }
  /** Set the model (and all layers') execution mode. */
  inline void set_execution_mode(execution_mode mode) {
    m_execution_mode = mode;
  }
  /** Get the model's execution mode. */
  inline execution_mode get_execution_mode() const {
    return m_execution_mode;
  }
  /** Set the model's current mini-batch size. */
  inline void set_current_mini_batch_size(int mini_batch_size) {
    m_current_mini_batch_size = mini_batch_size;
  }
  /** Get the model's current mini-batch size. */
  inline int get_current_mini_batch_size() const {
    return m_current_mini_batch_size;
  }
  /** Get the model's maximum mini-batch size. */
  inline int get_max_mini_batch_size() const {
    return m_max_mini_batch_size;
  }
  /** Get the model's effective mini-batch size. */
  inline int get_effective_mini_batch_size() const {
    return m_effective_mini_batch_size;
  }
  /** Set the model's effective mini-batch size. */
  inline void set_effective_mini_batch_size(int mini_batch_size) {
    m_effective_mini_batch_size = mini_batch_size;
  }

  /** Get the current phase (multiple epochs) in layer-wise model training. */
  inline int get_current_phase() const {
    return m_current_phase;
  }

  /**
   * Summarize statistics (e.g. timers, counters); these should be computable
   * quickly.
   */
  virtual void summarize_stats(lbann_summary& summarizer);
  /**
   * Summarize matrices (e.g. means); these are called less frequently and can
   * be more expensive.
   */
  virtual void summarize_matrices(lbann_summary& summarizer);

  /** Return true if the flag to stop training is set. */
  bool get_terminate_training() const {
    return m_terminate_training;
  }
  /** Set the terminate training flag (on or off). */
  void set_terminate_training(bool f) {
    m_terminate_training = f;
  }

  /** Train model. */
  virtual void train(int num_epochs);
  /** Evaluate model. */
  virtual void evaluate(execution_mode mode);

  /** Checkpoint model to given file descriptor, return number of bytes written */
  virtual bool save_to_checkpoint_shared(persist& p);
  /** Restore model by reading checkpoint from given file descriptor, return number of bytes read */
  virtual bool load_from_checkpoint_shared(persist& p);


 protected:

  /** The objective function used to train the model. */
  objective_function *m_objective_function;

  /** The model's current execution mode. */
  execution_mode m_execution_mode;
  /** Flag telling the model to terminate training. */
  bool m_terminate_training;
  /** Most recent/current epoch for the model. */
  int m_current_epoch;
  /** Most recent/current training step for the model. */
  int m_current_step;
  int m_current_validation_step;
  int m_current_testing_step;
  /**
   * Maximum possible minibatch size supported by layers in this model.
   * Note that this is local to the particular model, not across multiple
   * models.
   */
  int m_max_mini_batch_size;
  /** Size of the current mini-batch in the model. */
  int m_current_mini_batch_size;
  /**
   * The "effective" size of a minibatch.
   * This is the size of the minibatch across all models and used for e.g.
   * correctly averaging gradients from multiple models.
   */
  int m_effective_mini_batch_size;
  /** current phase (multiple of epoch counts) in training a model */
  int m_current_phase;
  /** Communicator for the model. */
  lbann_comm *m_comm;
  /** Current callbacks to process. */
  std::vector<lbann_callback *> m_callbacks;

  /** Default optimizer.
   *  If a layer needs to construct an optimizer during setup, it will
   *  make a copy of the default optimizer.
   */
  optimizer *m_default_optimizer;

  /** List of model metrics.
   *  A metric can be used to evaluate the performance of the model
   *  without affecting the training process.
   */
  std::vector<metric *> m_metrics;

  /** List of layers in model.
   *  The list is in execution order for forward propagation.
   */
  std::vector<Layer *> m_layers;
  /** List of weights in model. */
  std::vector<weights *> m_weights;

  /** Check if the model execution mode is valid. */
  virtual bool is_execution_mode_valid(execution_mode mode) const;
  /** Print out the description of a layer set up. */
  virtual std::string print_layer_description(const Layer* layer) const;
  /** Check if the layer execution order is topologically sorted. */
  virtual bool is_topologically_sorted() const;
  /** Remap pointers.
   *  Layer and weights pointers are remapped using the provided
   *  maps. If a pointer is not a key in the corresponding map, the
   *  pointer is not changed.
   */
  virtual void remap_pointers(const std::unordered_map<Layer *,Layer *>& layer_map,
                              const std::unordered_map<weights *,weights *>& weights_map);

  /** Set up topology of layer graph.
   *  Called in setup function. All layers in connected component of
   *  layer graph are added to the model and all parent/child
   *  relationships between layers are reciprocated.
   */
  virtual void setup_layer_topology();
  /** Set up layer execution order.
   *  Called in setup function.
   */
  virtual void setup_layer_execution_order() {}
  /** Set up layers.
   *  Called in setup function.
   */
  virtual void setup_layers();
  /** Set up weights.
   *  Called in setup function. All weights being used by layers or
   *  the objective function are added to the model and all unused
   *  weights are deleted.
   */
  virtual void setup_weights();

  /** Reset model pointer and execution mode. */
  virtual void reset_mode_and_model(execution_mode mode);
  /** Reset model statistics for an epoch. */
  virtual void reset_epoch_statistics(execution_mode mode);
  /** Evaluate model on a mini-batch */
  virtual bool evaluate_mini_batch(execution_mode mode);
  /** Train model on a mini-batch. */
  virtual bool train_mini_batch();

  /** Forward propagation step. */
  virtual void forward_prop(execution_mode mode);
  /** Backward propagation step. */
  virtual void backward_prop();
  /** Clear each layer's error signal tensor. */
  virtual void clear_error_signals();
  /** Update weights step. */
  virtual void update_weights();
  /** Update layers step. */
  virtual bool update_layers();

  ////////////////////////////////////////////////////////////
  // Callbacks
  ////////////////////////////////////////////////////////////

  /** Execute callbacks at start of training. */
  virtual void do_train_begin_cbs();
  /** Execute callbacks at end of training. */
  virtual void do_train_end_cbs();
  /** Execute callbacks at start of evaluation. */
  virtual void do_evaluate_begin_cbs(execution_mode mode);
  /** Execute callbacks at end of evaluation. */
  virtual void do_evaluate_end_cbs(execution_mode mode);
  /** Execute callbacks at start of epoch. */
  virtual void do_epoch_begin_cbs();
  /** Execute callbacks at end of epoch. */
  virtual void do_epoch_end_cbs();
  /** Execute callbacks at start of mini-batch. */
  virtual void do_batch_begin_cbs(execution_mode mode);
  /** Execute callbacks at end of mini-batch. */
  virtual void do_batch_end_cbs(execution_mode mode);
  /** Execute callbacks at start of model forward propagation. */
  virtual void do_model_forward_prop_begin_cbs(execution_mode mode);
  /** Execute callbacks at end of model forward propagation. */
  virtual void do_model_forward_prop_end_cbs(execution_mode mode);
  /** Execute callbacks at start of layer forward propagation. */
  virtual void do_layer_forward_prop_begin_cbs(execution_mode mode, Layer *l);
  /** Execute callbacks at end of layer forward propagation. */
  virtual void do_layer_forward_prop_end_cbs(execution_mode mode, Layer *l);
  /** Execute callbacks at start of model backward propagation. */
  virtual void do_model_backward_prop_begin_cbs();
  /** Execute callbacks at end of model backward propagation. */
  virtual void do_model_backward_prop_end_cbs();
  /** Execute callbacks at start of layer backward propagation. */
  virtual void do_layer_backward_prop_begin_cbs(Layer *l);
  /** Execute callbacks at end of layer backward propagation. */
  virtual void do_layer_backward_prop_end_cbs(Layer *l);
  /** Execute callbacks at start of model optimization. */
  virtual void do_model_optimize_begin_cbs();
  /** Execute callbacks at end of model optimization. */
  virtual void do_model_optimize_end_cbs();
  /** Execute callbacks at the start of weight optimization. */
  virtual void do_weight_optimize_begin_cbs(weights *w);
  /** Execute callbacks at the end of weight optimization. */
  virtual void do_weight_optimize_end_cbs(weights *w);

};

}  // namespace lbann

#endif  // LBANN_MODEL_HPP
