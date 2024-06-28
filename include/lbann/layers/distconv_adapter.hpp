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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYERS_DISTCONV_ADAPTER_HPP_INCLUDED
#define LBANN_LAYERS_DISTCONV_ADAPTER_HPP_INCLUDED

#include <unordered_map>
#include <unordered_set>

#include "distconv/tensor/distribution.hpp"
#include "distconv/tensor/tensor.hpp"

// Provide access to El::Int
#include "El.hpp"

namespace lbann {

class Layer;

namespace dc {
using Dist = ::distconv::tensor::Distribution;
using AbsTensor = ::distconv::tensor::AbstractTensor;
} // namespace dc

class tensor_overlap_constraints
{
public:
  using dist_set = std::unordered_set<dc::Dist*>;
  using const_dist_set = std::unordered_set<const dc::Dist*>;

  tensor_overlap_constraints() = default;
  virtual ~tensor_overlap_constraints() = default;

  void mark_equivalent(dc::Dist& d1, dc::Dist& d2);
  void mark_updated(const dc::Dist& d);
  void mark_invariant(const dc::Dist& d);
  void update_name(const dc::Dist& d, std::string name);

  void find_valid_overlap();

private:
  std::unordered_map<const dc::Dist*, dist_set> m_equivalents;
  const_dist_set m_updated;
  const_dist_set m_invariants;
  std::unordered_map<const dc::Dist*, std::string> m_names;
};

class distconv_adapter
{
  friend class Layer;

public:
  distconv_adapter(Layer& layer);
  virtual ~distconv_adapter() = default;

  /** Get activation tensor corresponding to child layer. */
  virtual const dc::AbsTensor& get_activations(const Layer& child) const = 0;
  /** Get error signal tensor corresponding to parent layer. */
  virtual const dc::AbsTensor& get_error_signals(const Layer& parent) const = 0;

  virtual void setup_distributions(tensor_overlap_constraints& constraints);
  void
  impose_adjacent_overlap_constraints(tensor_overlap_constraints& constraints);

  dc::Dist& get_prev_activations_dist();
  const dc::Dist& get_prev_activations_dist() const;
  dc::Dist& get_activations_dist();
  const dc::Dist& get_activations_dist() const;
  dc::Dist& get_prev_error_signals_dist();
  const dc::Dist& get_prev_error_signals_dist() const;
  dc::Dist& get_error_signals_dist();
  const dc::Dist& get_error_signals_dist() const;

  virtual void setup_fp_tensors();
  virtual void setup_bp_tensors();

  virtual void setup_layer(size_t workspace_capacity) {}

  virtual void fp_setup() = 0;
  virtual void fp_postprocess() = 0;
  virtual void bp_setup() = 0;
  virtual void bp_postprocess() = 0;

  virtual bool parent_copy_required(size_t input_index) const;
  virtual bool parent_shuffle_required(size_t input_index) const;
  virtual bool child_copy_required(size_t output_index) const;
  virtual bool child_shuffle_required(size_t output_index) const;

  virtual void dump_activations() const = 0;
  virtual void dump_original_activations() = 0;
  virtual void dump_error_signals() const = 0;
  virtual void dump_original_error_signals() = 0;

protected:
  virtual Layer& layer();
  virtual const Layer& layer() const;
  std::string get_name() const;

  virtual void setup_prev_activations() = 0;
  virtual void setup_original_prev_activations() = 0;
  virtual void setup_activations() = 0;
  virtual void setup_original_activations() = 0;

  virtual void setup_prev_error_signals() = 0;
  virtual void setup_original_prev_error_signals() = 0;
  virtual void setup_error_signals() = 0;
  virtual void setup_original_error_signals() = 0;

  virtual void ensure_prev_activations() = 0;
  virtual void copy_out_activations() = 0;
  virtual void ensure_prev_error_signals() = 0;
  virtual void copy_out_error_signals() = 0;

  std::vector<dc::Dist> m_prev_activations_dists;
  std::vector<dc::Dist> m_activations_dists;
  std::vector<dc::Dist> m_prev_error_signals_dists;
  std::vector<dc::Dist> m_error_signals_dists;

private:
  Layer& m_layer;
  std::vector<bool> m_parent_copy_required;
  std::vector<bool> m_parent_shuffle_required;
  std::vector<bool> m_child_copy_required;
  std::vector<bool> m_child_shuffle_required;

  void setup_tensor_shuffle();
  void adjust_parallel_strategy();
};

} // namespace lbann

#endif // LBANN_LAYERS_DISTCONV_ADAPTER_HPP_INCLUDED
