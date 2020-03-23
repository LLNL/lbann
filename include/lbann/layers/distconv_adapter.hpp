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

#ifndef LBANN_LAYERS_DISTCONV_ADAPTER_HPP_INCLUDED
#define LBANN_LAYERS_DISTCONV_ADAPTER_HPP_INCLUDED

#include "lbann/utils/distconv.hpp"

namespace lbann {

class Layer;

class distconv_adapter {
  friend class Layer;
public:
  distconv_adapter(Layer& layer);
  virtual ~distconv_adapter() = default;

  /** Get activation tensor corresponding to child layer. */
  virtual const dc::AbsTensor& get_activations(const Layer& child) const = 0;
  /** Get error signal tensor corresponding to parent layer. */
  virtual const dc::AbsTensor& get_error_signals(const Layer& parent) const = 0;

  virtual void setup_distributions(std::map<dc::Dist*, std::set<dc::Dist*>> &equivalents,
                                   std::set<dc::Dist*> &updated,
                                   std::set<dc::Dist*> &invariants);
  void impose_adjacent_distribution_constraints(
      std::map<dc::Dist*, std::set<dc::Dist*>> &equivalents);
  dc::Dist &get_prev_activations_dist();
  const dc::Dist &get_prev_activations_dist() const;
  dc::Dist &get_activations_dist();
  const dc::Dist &get_activations_dist() const;
  dc::Dist &get_prev_error_signals_dist();
  const dc::Dist &get_prev_error_signals_dist() const;
  dc::Dist &get_error_signals_dist();
  const dc::Dist &get_error_signals_dist() const;

  // Setup fp tensors
  virtual void setup_prev_activations() = 0;
  virtual void setup_original_prev_activations() = 0;
  virtual void setup_activations() = 0;
  virtual void setup_original_activations() = 0;
  virtual void setup_fp_tensors();

  // Setup bp tensors
  virtual void setup_prev_error_signals() = 0;
  virtual void setup_original_prev_error_signals() = 0;
  virtual void setup_error_signals() = 0;
  virtual void setup_original_error_signals() = 0;
  virtual void setup_bp_tensors();

  virtual void setup_layer(size_t workspace_capacity) {}

  virtual void fp_setup(El::Int mini_batch_size) = 0;
  virtual void bp_setup(El::Int mini_batch_size) = 0;

  virtual void ensure_prev_activations() = 0;
  virtual void copy_out_activations() = 0;
  virtual void ensure_prev_error_signals() = 0;
  virtual void copy_out_error_signals() = 0;


  bool parent_copy_required(size_t input_index) const;
  bool parent_shuffle_required(size_t input_index) const;
  bool child_copy_required(size_t output_index) const;
  bool child_shuffle_required(size_t output_index) const;

  virtual void dump_activations() const = 0;
  virtual void dump_original_activations()= 0;
  virtual void dump_error_signals() const = 0;
  virtual void dump_original_error_signals()= 0;

 protected:
  virtual Layer& layer();
  virtual const Layer& layer() const;
  std::string get_name() const;
  int get_num_dims() const;
  int get_num_spatial_dims() const;

  std::vector<dc::Dist> m_prev_activations_dists;
  std::vector<dc::Dist> m_activations_dists;
  std::vector<dc::Dist> m_prev_error_signals_dists;
  std::vector<dc::Dist> m_error_signals_dists;

  std::vector<bool> m_parent_copy_required;
  std::vector<bool> m_parent_shuffle_required;
  std::vector<bool> m_child_copy_required;
  std::vector<bool> m_child_shuffle_required;

 private:
  Layer& m_layer;

  void setup_tensor_shuffle();
};

} // namespace lbann

#endif // LBANN_LAYERS_DISTCONV_ADAPTER_HPP_INCLUDED
