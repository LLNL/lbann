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

#include "lbann/layers/distconv_adapter.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/utils/distconv.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

distconv_adapter::distconv_adapter(Layer& layer) : m_layer(layer)
{
  setup_tensor_shuffle();
}

Layer& distconv_adapter::layer() { return m_layer; }
const Layer& distconv_adapter::layer() const { return m_layer; }

std::string distconv_adapter::get_name() const { return layer().get_name(); }

dc::Dist& distconv_adapter::get_prev_activations_dist()
{
  return const_cast<dc::Dist&>(
    static_cast<const distconv_adapter&>(*this).get_prev_activations_dist());
}

const dc::Dist& distconv_adapter::get_prev_activations_dist() const
{
  size_t idx = 0;
  if (idx >= m_prev_activations_dists.size()) {
    LBANN_ERROR("Invalid access to previous activations distributions");
  }
  return m_prev_activations_dists[idx];
}

dc::Dist& distconv_adapter::get_activations_dist()
{
  return const_cast<dc::Dist&>(
    static_cast<const distconv_adapter&>(*this).get_activations_dist());
}

const dc::Dist& distconv_adapter::get_activations_dist() const
{
  size_t idx = 0;
  if (idx >= m_activations_dists.size()) {
    LBANN_ERROR("Invalid access to activations distributions");
  }
  return m_activations_dists[idx];
}

dc::Dist& distconv_adapter::get_prev_error_signals_dist()
{
  return const_cast<dc::Dist&>(
    static_cast<const distconv_adapter&>(*this).get_prev_error_signals_dist());
}

const dc::Dist& distconv_adapter::get_prev_error_signals_dist() const
{
  size_t idx = 0;
  if (idx >= m_prev_error_signals_dists.size()) {
    LBANN_ERROR("Invalid access to previous error signals distributions");
  }
  return m_prev_error_signals_dists[idx];
}

dc::Dist& distconv_adapter::get_error_signals_dist()
{
  return const_cast<dc::Dist&>(
    static_cast<const distconv_adapter&>(*this).get_error_signals_dist());
}

const dc::Dist& distconv_adapter::get_error_signals_dist() const
{
  size_t idx = 0;
  if (idx >= m_error_signals_dists.size()) {
    LBANN_ERROR("Invalid access to error signals distributions");
  }
  return m_error_signals_dists[idx];
}

void distconv_adapter::setup_fp_tensors()
{
  setup_original_prev_activations();
  setup_prev_activations();
  setup_activations();
  setup_original_activations();
}

void distconv_adapter::setup_bp_tensors()
{
  setup_original_prev_error_signals();
  setup_prev_error_signals();
  setup_error_signals();
  setup_original_error_signals();
}

bool distconv_adapter::parent_copy_required(size_t input_index) const
{
  if (input_index < m_parent_copy_required.size()) {
    return m_parent_copy_required.at(input_index);
  }
  else {
    LBANN_ERROR("Out of range error! parent_copy_required size: ",
                m_parent_copy_required.size(),
                ", index: ",
                input_index);
  }
}

bool distconv_adapter::parent_shuffle_required(size_t input_index) const
{
  if (input_index < m_parent_shuffle_required.size()) {
    return m_parent_shuffle_required.at(input_index);
  }
  else {
    LBANN_ERROR("Out of range error! parent_shuffle_required size: ",
                m_parent_shuffle_required.size(),
                ", index: ",
                input_index);
  }
}

bool distconv_adapter::child_copy_required(size_t output_index) const
{
  if (output_index < m_child_copy_required.size()) {
    return m_child_copy_required.at(output_index);
  }
  else {
    LBANN_ERROR("Out of range error! child_copy_required size: ",
                m_child_copy_required.size(),
                ", index: ",
                output_index);
  }
}

bool distconv_adapter::child_shuffle_required(size_t output_index) const
{
  if (output_index < m_child_shuffle_required.size()) {
    return m_child_shuffle_required.at(output_index);
  }
  else {
    LBANN_ERROR("Out of range error! child_shuffle_required size: ",
                m_child_shuffle_required.size(),
                ", index: ",
                output_index);
  }
}

void distconv_adapter::setup_tensor_shuffle()
{
  assert_always(layer().distconv_enabled());

  const auto& ps = layer().get_parallel_strategy();
  for (const auto& p : layer().get_parent_layers()) {
    m_parent_copy_required.push_back(!p->distconv_enabled());
    m_parent_shuffle_required.push_back((!p->distconv_enabled()) ||
                                        (ps != p->get_parallel_strategy()));
  }

  for (const auto& c : layer().get_child_layers()) {
    m_child_copy_required.push_back(!c->distconv_enabled());
    m_child_shuffle_required.push_back((!c->distconv_enabled()) ||
                                       (ps != c->get_parallel_strategy()));
  }

  std::stringstream ss;
  std::stringstream parent_copyin_ss;
  std::stringstream parent_shuffle_ss;
  for (int i = 0; i < layer().get_num_parents(); ++i) {
    if (m_parent_copy_required[i]) {
      parent_copyin_ss << " " << i;
    }
    if (m_parent_shuffle_required[i]) {
      parent_shuffle_ss << " " << i;
    }
  }
  std::stringstream child_copyout_ss;
  std::stringstream child_shuffle_ss;
  for (int i = 0; i < layer().get_num_children(); ++i) {
    if (m_child_copy_required[i]) {
      child_copyout_ss << " " << i;
    }
    if (m_child_shuffle_required[i]) {
      child_shuffle_ss << " " << i;
    }
  }
  if (!parent_copyin_ss.str().empty()) {
    ss << " parent copyin:" << parent_copyin_ss.str() << ",";
  }
  if (!parent_shuffle_ss.str().empty()) {
    ss << " parent shuffle:" << parent_shuffle_ss.str() << ",";
  }
  if (!child_copyout_ss.str().empty()) {
    ss << " child copyout:" << child_copyout_ss.str() << ",";
  }
  if (!child_shuffle_ss.str().empty()) {
    ss << " child shuffle:" << child_shuffle_ss.str();
  }
  if (ss.str().size() > 0) {
    dc::MPIRootPrintStreamDebug() << get_name() << ":" << ss.str();
  }
}

void distconv_adapter::adjust_parallel_strategy()
{
  auto& ps = layer().get_parallel_strategy();
  // The numerical attributes are 0 when not specified. Assume no
  // partitioning then.
  auto n = ps.sample_groups != 0 ? ps.sample_groups : 1;
  auto c = ps.channel_groups != 0 ? ps.channel_groups : 1;
  auto f = ps.filter_groups != 0 ? ps.filter_groups : 1;
  auto d = (dc::get_num_spatial_dims(layer()) == 3 && ps.depth_groups != 0)
             ? ps.depth_groups
             : 1;
  auto h = ps.height_groups != 0 ? ps.height_groups : 1;
  auto w = ps.width_groups != 0 ? ps.width_groups : 1;
  auto np = layer().get_comm()->get_procs_per_trainer();

  const auto spatial_prod = d * h * w;

  const auto layer_type = layer().get_type();

  // if only one process is used, do not parallelize
  if (np == 1) {
    n = c = f = h = w = d = 1;
  }

  if (layer_type == "convolution" || layer_type == "deconvolution") {
    if (c != f) {
      LBANN_ERROR(
        "The numbers of channel and filter decomposition should be the same.");
    }

    if (c != 1 || f != 1) {
      LBANN_ERROR(
        "Distconv does not support filter parallelization yet. Layer: ",
        get_name(),
        ", parallel strategy: ",
        ps);
    }
  }

  else if (layer_type == "channel-wise fully-connected" ||
           layer_type == "matmul") {
    if (c != f) {
      if (layer().get_comm()->am_trainer_master()) {
        LBANN_WARNING("The number of channel and filter decomposition should "
                      "be the same. Setting",
                      " the filter decomposition to channel decomposition: ",
                      c);
      }
      ps.filter_groups = c;
      f = c;
    }

    if (spatial_prod != 1) {
      LBANN_ERROR("Distributed channel-wise fully-connected or 3D matmul does "
                  "not support spatial (column-wise) ",
                  "parallelization: ",
                  get_name(),
                  ", parallel strategy: ",
                  ps);
    }
  }

  if (n * c * spatial_prod > np) {
    LBANN_ERROR("The number of MPI ranks must be at least as large as the "
                "number of processes implied by parallel strategy: ",
                ps);
  }
  // Put the remaining factor into the outer-most process dimension
  float rem = np / (float)(n * c * spatial_prod);
  n *= rem;
  ps.sample_splits *= rem;
  if (n * c * spatial_prod != np) {
    LBANN_ERROR("Can't determine factorization of the number of MPI ranks for "
                "parallel strategy: ",
                ps);
  }

  assert_always(spatial_prod * n * c == np);
  assert_always(spatial_prod * n * f == np);

  if (n != ps.sample_groups) {
    LBANN_MSG(
      "[",
      layer().get_name(),
      "]: Adjusting the parallel strategy to partition across the sample "
      "dimension to maximize parallelism: original sample groups = ",
      ps.sample_groups,
      " adjusted sample groups = ",
      n);
  }
  ps.sample_groups = n;
  ps.channel_groups = c;
  ps.filter_groups = f;
  ps.depth_groups = d;
  ps.height_groups = h;
  ps.width_groups = w;
  // If splits are not set, set them to be equal to the group numbers
  if (ps.sample_splits == 0)
    ps.sample_splits = n;
  if (ps.channel_splits == 0)
    ps.channel_splits = c;
  if (ps.filter_splits == 0)
    ps.filter_splits = f;
  if (ps.depth_splits == 0)
    ps.depth_splits = d;
  if (ps.height_splits == 0)
    ps.height_splits = h;
  if (ps.width_splits == 0)
    ps.width_splits = w;
}

void distconv_adapter::setup_distributions(
  tensor_overlap_constraints& constraints)
{
  const auto num_dims = dc::get_num_dims(layer());
  dc::Shape input_locale_shape(num_dims);
  dc::Shape input_split_shape(num_dims);
  dc::Shape output_locale_shape(num_dims);
  dc::Shape output_split_shape(num_dims);

  adjust_parallel_strategy();
  const auto& ps = layer().get_parallel_strategy();

  input_locale_shape[dc::get_sample_dim()] = ps.sample_groups;
  input_locale_shape[dc::get_channel_dim()] = ps.channel_groups;
  input_locale_shape[0] = ps.width_groups;
  input_locale_shape[1] = ps.height_groups;
  if (num_dims == 5)
    input_locale_shape[2] = ps.depth_groups;

  input_split_shape[dc::get_sample_dim()] = ps.sample_splits;
  input_split_shape[dc::get_channel_dim()] = ps.channel_splits;
  input_split_shape[0] = ps.width_splits;
  input_split_shape[1] = ps.height_splits;
  if (num_dims == 5)
    input_split_shape[2] = ps.depth_splits;

  output_locale_shape[dc::get_sample_dim()] = ps.sample_groups;
  output_locale_shape[dc::get_channel_dim()] = ps.filter_groups;
  output_locale_shape[0] = ps.width_groups;
  output_locale_shape[1] = ps.height_groups;
  if (num_dims == 5)
    output_locale_shape[2] = ps.depth_groups;

  output_split_shape[dc::get_sample_dim()] = ps.sample_splits;
  output_split_shape[dc::get_channel_dim()] = ps.filter_splits;
  output_split_shape[0] = ps.width_splits;
  output_split_shape[1] = ps.height_splits;
  if (num_dims == 5)
    output_split_shape[2] = ps.depth_splits;

  auto prev_activations_dist =
    dc::Dist::make_shared_distribution(input_locale_shape, input_split_shape);
  auto activations_dist =
    dc::Dist::make_shared_distribution(output_locale_shape, output_split_shape);
  auto prev_error_signals_dist = activations_dist;
  auto error_signals_dist = prev_activations_dist;

  m_prev_activations_dists.emplace_back(prev_activations_dist);
  m_activations_dists.emplace_back(activations_dist);
  m_prev_error_signals_dists.emplace_back(prev_error_signals_dist);
  m_error_signals_dists.emplace_back(error_signals_dist);

  std::string layer_name = layer().get_name();
  constraints.update_name(get_prev_activations_dist(), layer_name + " prev_activations ");
  constraints.update_name(get_activations_dist(), layer_name + " activations ");
  constraints.update_name(get_prev_error_signals_dist(), layer_name + " prev_error_signals ");
  constraints.update_name(get_error_signals_dist(), layer_name + " error_signals ");
}

void distconv_adapter::impose_adjacent_overlap_constraints(
  tensor_overlap_constraints& constraints)
{
  const auto& l = layer();
  const auto& ps = l.get_parallel_strategy();

  auto& x = get_prev_activations_dist();
  auto& y = get_activations_dist();
  auto& dx = get_error_signals_dist();
  auto& dy = get_prev_error_signals_dist();

  // TEMPORARY HACK. Each tensor should be able to have its own
  // distribution, however, the current design only allows for a
  // single distribution for all output tensors in each layer,
  // meaning the data and label tensors need to have the same
  // distribution. The data tensor is likely to have halo as the
  // next layer will be convolution, whereas the label won't need to
  // have halo. For now, ignore the child layer for the label data.

  if (l.get_type() == "input") {
    Layer* child = const_cast<Layer*>(l.get_child_layers()[0]);
    if (child->distconv_enabled() && child->get_parallel_strategy() == ps) {
      auto& child_x = child->get_distconv_adapter().get_prev_activations_dist();
      auto& child_dx = child->get_distconv_adapter().get_error_signals_dist();
      constraints.mark_equivalent(y, child_x);
      constraints.mark_equivalent(dy, child_dx);
    }
  }
  else {
    for (auto& child : l.get_child_layers()) {
      if (child->distconv_enabled() && child->get_parallel_strategy() == ps) {
        auto& child_x = const_cast<dc::Dist&>(
          child->get_distconv_adapter().get_prev_activations_dist());
        auto& child_dx = const_cast<dc::Dist&>(
          child->get_distconv_adapter().get_error_signals_dist());
        constraints.mark_equivalent(y, child_x);
        constraints.mark_equivalent(dy, child_dx);
      }
    }
  }
  for (auto& parent : l.get_parent_layers()) {
    if (parent->get_type() == "input") {
      const int child_index = parent->find_child_layer_index(l);
      if (child_index == 1)
        continue;
      assert_eq(child_index, 0);
    }
    if (parent->distconv_enabled() && parent->get_parallel_strategy() == ps) {
      auto& parent_y = const_cast<dc::Dist&>(
        parent->get_distconv_adapter().get_activations_dist());
      auto& parent_dy = const_cast<dc::Dist&>(
        parent->get_distconv_adapter().get_prev_error_signals_dist());
      constraints.mark_equivalent(x, parent_y);
      constraints.mark_equivalent(dx, parent_dy);
    }
  }
}

void tensor_overlap_constraints::mark_equivalent(dc::Dist& d1, dc::Dist& d2)
{
  // d1 -> d2
  if (m_equivalents.find(&d1) == m_equivalents.end()) {
    m_equivalents.insert(std::make_pair(&d1, dist_set()));
  }
  m_equivalents[&d1].insert(&d2);
  // d2 -> d1
  if (m_equivalents.find(&d2) == m_equivalents.end()) {
    m_equivalents.insert(std::make_pair(&d2, dist_set()));
  }
  m_equivalents[&d2].insert(&d1);
}

void tensor_overlap_constraints::mark_updated(const dc::Dist& d)
{
  m_updated.insert(&d);
}

void tensor_overlap_constraints::mark_invariant(const dc::Dist& d)
{
  m_invariants.insert(&d);
}

void tensor_overlap_constraints::update_name(const dc::Dist& d, std::string name)
{
  m_names[&d] = name;
}

void tensor_overlap_constraints::find_valid_overlap()
{
  while (m_updated.size() > 0) {
    const_dist_set updated_new;
    for (const auto d : m_updated) {
      auto equivalent_dists = m_equivalents.find(d);
      if (equivalent_dists == m_equivalents.end())
        continue;
      for (auto p : equivalent_dists->second) {
        if (d->get_overlap() != p->get_overlap()) {
          // p must have equal dist as d but is different.
          if (m_invariants.find(p) != m_invariants.end()) {
            // p can't be changed, so we can't solve the constraint.
            LBANN_ERROR("Incompatible overlap: ", m_names[d], *d, " <=> ", m_names[p], *p);
          }
          p->set_overlap(d->get_overlap());
          updated_new.insert(p);
        }
      }
    }
    m_updated = std::move(updated_new);
  }
}

} // namespace lbann
