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
// model_planar .hpp .cpp - Planar neural network models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_MODEL_PLANAR_HPP
#define LBANN_MODEL_PLANAR_HPP

#include "lbann/models/model.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/layers/learning/learning.hpp"
#include <vector>
#include <string>

namespace lbann {

class planar_model : public model {
 public:
  typedef std::vector<Layer*> Layer_peers_t;
  typedef std::vector<Layer_peers_t> Layer_stack_t;
  // todo:
  // typedef std::vector<Layer_stack_t> Layer_groups_t;

  /// Constructor
  planar_model(lbann_comm *comm,
               int mini_batch_size,
               objective_functions::objective_function *obj_fn,
               optimizer *default_optimizer,
               int width);
  /** Copy constructor. */
  planar_model(const planar_model& other);

  /** Copy assignment operator. */
  planar_model& operator=(const planar_model& other);


  /// Destructor
  ~planar_model() override;

  /** Create copy. */
  planar_model* copy() const override { return new planar_model(*this); }

  /// Allow access to the model's layers.
  Layer_stack_t& get_layers() { return m_layers; }
  /// Allow the read-only access to the model's layers.
  const Layer_stack_t& get_layers() const { return m_layers; }
  /// Deep-copy layers
  void copy_layers(const Layer_stack_t& src_stack);
  /// shallow-copy layers
  void set_layers(const Layer_stack_t& new_stack);

  void add_layer(Layer *layer) override;

  /// Setup planar model
  void setup() override;

  std::string name() const override { return "planar_model"; }

  /// Set the model (and all layers') execution mode.
  void set_execution_mode(execution_mode mode) override;

  void summarize_stats(lbann_summary& summarizer) override;
  void summarize_matrices(lbann_summary& summarizer) override;

 protected:
  typedef std::unordered_map<Layer*, Layer*> Layer_map_t;

  /// Deallocate layer objects
  void delete_layers();

  static Layer* find_layer(const Layer_map_t& map_src_to_new, const Layer* const src_layer);

  /// Renew the layer linking pointers after copying layers
  virtual void renew_layer_links(const Layer_stack_t& src_stack,
                         const Layer_map_t& map_src_to_new) const;

  /** Following functions are used to add a set of layers at given horizontal level
   *  on a planar space. The layers are added either by duplicating a single layer
   *  or placing individual layers. */
  virtual void stackup_duplicate(Layer_peers_t& new_layer, int num_heads);

  void setup_subset();

  virtual bool check_layer_type_consistency(const Layer_peers_t& layer_peers) const;

  /// Ensure weight matriecs in heads at each level are the same
  virtual void equalize();

  void forward_prop_to_evaluate() override;
  bool update_io_layers() override;
  void forward_prop() override;
  void backward_prop() override;
  void update_optimizable_layers() override;

  /// Check if the model has a valid data set for the execution mode
  bool is_execution_mode_valid(execution_mode mode) const override;

  // Methods for calling every callback at different points.
  // These are currently dummy methods
  void setup_callbacks() override {};
  void do_train_begin_cbs() override {};
  void do_train_end_cbs() override {};
  void do_phase_end_cbs() override {};
  void do_epoch_begin_cbs() override {};
  void do_epoch_end_cbs() override {};
  void do_batch_begin_cbs() override {};
  void do_batch_end_cbs() override {};
  void do_test_begin_cbs() override {};
  void do_test_end_cbs() override {};
  void do_validation_begin_cbs() override {};
  void do_validation_end_cbs() override {};
  void do_model_forward_prop_begin_cbs() override {};
  void do_layer_forward_prop_begin_cbs(Layer *l) override {};
  void do_model_forward_prop_end_cbs() override {};
  void do_layer_forward_prop_end_cbs(Layer *l) override {};
  void do_model_backward_prop_begin_cbs() override {};
  void do_layer_backward_prop_begin_cbs(Layer *l) override {};
  void do_model_backward_prop_end_cbs() override {};
  void do_layer_backward_prop_end_cbs(Layer *l) override {};
  /// Evaluation phases (validation / testing)
  void do_batch_evaluate_begin_cbs() override {};
  void do_batch_evaluate_end_cbs() override {};
  void do_model_evaluate_forward_prop_begin_cbs() override {};
  void do_layer_evaluate_forward_prop_begin_cbs(Layer *l) override {};
  void do_model_evaluate_forward_prop_end_cbs() override {};
  void do_layer_evaluate_forward_prop_end_cbs(Layer *l) override {};

 protected:
  /// the maximum number of horizontal layers in the network
  int m_width;
  std::vector<int> m_head_counts;

  /// List of layers on the plane
  /// m_layers contains a set of horizontal layers for each level
  /// m_head_counts contains the number of horizontal layers for each level
  /// For now, the entries in m_head_counts are either 1 or m_width (to support
  /// the Siamese network)
  Layer_stack_t m_layers;
};

}  // namespace lbann

#endif  // LBANN_MODEL_PLANAR_HPP
