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

#ifndef LBANN_LAYERS_LOSS_CROSS_ENTROPY_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_CROSS_ENTROPY_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/utils/distconv.hpp"

namespace lbann {

/** @brief Cross entropy loss function.
 *
 *  Given a predicted distribution @f$y@f$ and ground truth
 *  distribution @f$\hat{y}@f$,
 *  @f[ CE(y,\hat{y}) = - \sum\limits_{i} \hat{y}_i \log y_i @f]
 */
template <data_layout T_layout, El::Device Dev>
class cross_entropy_layer : public Layer {
public:

  cross_entropy_layer(lbann_comm *comm) : Layer(comm) {
    this->m_expected_num_parent_layers = 2;
  }

  cross_entropy_layer(const cross_entropy_layer& other)
    : Layer(other) {
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() :
                      nullptr);
  }

  cross_entropy_layer& operator=(const cross_entropy_layer& other) {
    Layer::operator=(other);
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() :
                      nullptr);
    return *this;
  }

  cross_entropy_layer* copy() const override { return new cross_entropy_layer(*this); }
  std::string get_type() const override { return "cross entropy"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  void setup_dims() override {
    Layer::setup_dims();
    set_output_dims({1});

#ifdef LBANN_HAS_DISTCONV
    if (using_distconv()) {
      return;
    }
#endif

    // Check that input dimensions match
    if (get_input_dims(0) != get_input_dims(1)) {
      const auto& parents = get_parent_layers();
      std::stringstream err;
      err << get_type() << " layer \"" << get_name() << "\" "
          << "has input tensors with different dimensions (";
      for (int i = 0; i < get_num_parents(); ++i) {
        const auto& dims = get_input_dims(i);
        err << (i > 0 ? ", " : "")
            << "layer \"" << parents[i]->get_name() << "\" outputs ";
        for (size_t j = 0; j < dims.size(); ++j) {
          err << (j > 0 ? " x " : "") << dims[j];
        }
      }
      err << ")";
      LBANN_ERROR(err.str());
    }

  }

  void setup_data() override {
    Layer::setup_data();

    // Initialize workspace
    const auto& prediction = get_prev_activations(0);
    switch (get_data_layout()) {
    case data_layout::DATA_PARALLEL:
      m_workspace.reset(new StarVCMat<Dev>(prediction.Grid(),
                                           prediction.Root()));
      break;
    case data_layout::MODEL_PARALLEL:
      m_workspace.reset(new StarMRMat<Dev>(prediction.Grid(),
                                           prediction.Root()));
      break;
    default: LBANN_ERROR("invalid data layout");
    }
#ifdef HYDROGEN_HAVE_CUB
    if (m_workspace->GetLocalDevice() == El::Device::GPU) {
      m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // HYDROGEN_HAVE_CUB

  }

  void fp_compute() override {

#ifdef LBANN_HAS_DISTCONV
    if (distconv_enabled()) {
      fp_compute_distconv();
      if (!early_terminate_last_iteration()) {
        return;
      }
      // fall through the normal code path to obtain reference results
    }
#endif

    // Initialize workspace
    const auto& prediction = get_prev_activations(0);
    m_workspace->AlignWith(prediction.DistData());
    m_workspace->Resize(1, prediction.Width());

    // Compute local contributions and accumulate
    /// @todo Consider reduce rather than allreduce
    local_fp_compute(get_local_prev_activations(0),
                     get_local_prev_activations(1),
                     m_workspace->Matrix());
    m_comm->allreduce(*m_workspace, m_workspace->RedundantComm());
    El::Copy(*m_workspace, get_activations());

#ifdef LBANN_HAS_DISTCONV
    if (distconv_enabled() && early_terminate_last_iteration() &&
        keep_original()) {
      dump_reference_activations();
    }
#endif // LBANN_HAS_DISTCONV
  }

  void bp_compute() override {

#ifdef LBANN_HAS_DISTCONV
    if (distconv_enabled()) {
      bp_compute_distconv();
      if (!early_terminate_last_iteration()) {
        return;
      }
    }
#endif // LBANN_HAS_DISTCONV

    // Initialize workspace
    const auto& prediction = get_prev_activations(0);
    m_workspace->AlignWith(prediction.DistData());
    El::Copy(get_prev_error_signals(), *m_workspace);

    // Compute local gradients
    local_bp_compute(get_local_prev_activations(0),
                     get_local_prev_activations(1),
                     m_workspace->LockedMatrix(),
                     get_local_error_signals(0),
                     get_local_error_signals(1));

#ifdef LBANN_HAS_DISTCONV
    if (distconv_enabled() && early_terminate_last_iteration() &&
        keep_original()) {
      dump_reference_error_signals();
    }
#endif // LBANN_HAS_DISTCONV
  }

private:

  /** Compute local contributions to cross entropy loss. */
  static void local_fp_compute(const AbsMat& local_prediction,
                               const AbsMat& local_ground_truth,
                               AbsMat& local_contribution);
  /** Compute local gradients. */
  static void local_bp_compute(const AbsMat& local_prediction,
                               const AbsMat& local_ground_truth,
                               const AbsMat& local_gradient_wrt_output,
                               AbsMat& local_gradient_wrt_prediction,
                               AbsMat& local_gradient_wrt_ground_truth);

  /** Workspace matrix. */
  std::unique_ptr<AbsDistMat> m_workspace;

#ifdef LBANN_HAS_DISTCONV
 protected:
  dc::CrossEntropy *m_cross_entropy;
  dc::TensorDev m_ground_truth_t;
  dc::TensorDev m_d_ground_truth_t;

  void fp_compute_distconv() {
    assert_always(distconv_enabled());
    m_cross_entropy->forward(m_prev_activations_t, m_ground_truth_t,
                             m_activations_t);
    copy_out_activations();
  }

  void bp_compute_distconv() {
    assert_always(distconv_enabled());
    m_cross_entropy->backward(m_prev_activations_t, m_ground_truth_t,
                              m_prev_error_signals_t,
                              m_error_signals_t, m_d_ground_truth_t);
    copy_out_error_signals();
  }

 public:
  void setup_tensor_distribution_init(
      std::map<const Layer*, std::array<dc::Dist, dc::num_dists>> &dists,
      std::map<dc::Dist*, std::set<dc::Dist*>> &invariants,
      std::set<dc::Dist*> &updated,
      std::set<dc::Dist*> &fixed) override {
    Layer::setup_tensor_distribution_init(dists, invariants, updated, fixed);
    if (!this->distconv_enabled()) return;

    // Output tensors share all dimensions except for the sample dimension
    auto activations_split = dists[this][1].get_split_shape();
    auto error_signals_split = dists[this][3].get_split_shape();
    for (int i = 0; i < activations_split.length() - 1; ++i) {
      activations_split[i] = 1;
      error_signals_split[i] = 1;
    }
    dists[this][1].set_split_shape(activations_split);
    dists[this][3].set_split_shape(error_signals_split);

    // No overlap supported yet
    const dc::IntVector no_overlap(dc::num_dims, 0);
    for (int i = 0; i < 4; ++i) {
      auto &dist = dists[this][i];
      dist.set_overlap(no_overlap);
      updated.insert(&dist);
      fixed.insert(&dist);
    }
  }

  dc::Shape get_activations_tensor_local_shape() const {
    auto input_shape = m_prev_activations_t.get_local_shape();
    for (int i = 0; i < input_shape.length() - 1; ++i) {
      input_shape[i] = 1;
    }
    return input_shape;
  }

  // NOTE: LBANN matrix is a 2-D matrix, while Distconv keeps the
  // original spatial and channel dimensions, so
  // get_output_tensor_shape() doesn't work here.
  void setup_activations_tensor(const std::array<dc::Dist, dc::num_dists> &dists,
                                bool allocate=true) override {
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
    auto output_tensor_shape = m_prev_activations_t.get_shape();
    for (int i = 0; i < output_tensor_shape.length() - 1; ++i) {
      output_tensor_shape[i] = 1;
    }
    const auto activations_local_shape =
        get_activations_tensor_local_shape();
    m_activations_t = dc::TensorDev(output_tensor_shape,
                                    loc, dists[1], activations_local_shape);
    if (allocate) {
      assert0(m_activations_t.allocate());
      m_activations_t.zero(dc::get_stream());
    }
  }

  void setup_activations_copyout_tensor(const std::array<dc::Dist, dc::num_dists> &dists)
      override {
    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
    const auto sample_dist = get_hydrogen_matrix_distribution();
    const auto output_tensor_shape = m_activations_t.get_shape();
    auto output_local_shape = output_tensor_shape;
    // Set the sample dimension as 0 so that its actual value is
    // calculated by Distconv
    output_local_shape[-1] = 0;
    m_activations_copyout = dc::TensorDev(output_tensor_shape, loc, sample_dist,
                                          output_local_shape);
    if (m_child_copy_out_required) {
      m_activations_shuffler = dc::get_tensor_shuffler(
          m_activations_t, m_activations_copyout);
      for (int i = 0; i < 3; ++i) {
        m_activations_shuffler_last_mb[i] = nullptr;
      }
    }
  }

  void setup_prev_error_signals_tensor(const std::array<dc::Dist, dc::num_dists> &dists)
      override {
    using namespace dc;
    const LocaleMPI loc(dc::get_mpi_comm(), false);
    const auto sample_dist = get_hydrogen_matrix_distribution();
    const auto output_tensor_shape = m_activations_t.get_shape();
    auto output_local_shape = output_tensor_shape;
    // Set the sample dimension as 0 so that its actual value is
    // calculated by Distconv
    output_local_shape[-1] = 0;

    if (m_child_copy_out_required || m_child_shuffle_required) {
      if (m_child_copy_out_required) {
        m_prev_error_signals_const_view = TensorDev(output_tensor_shape, loc,
                                                    sample_dist,
                                                    output_local_shape);
      } else {
        m_prev_error_signals_const_view =
            get_child_layers()[0]->get_error_signals_t(*this);
      }
      m_prev_error_signals_t = TensorDev(output_tensor_shape, loc,
                                         dists[3],
                                         m_activations_t.get_local_shape());
      assert0(m_prev_error_signals_t.allocate());
      m_prev_error_signals_t.zero(dc::get_stream());
      m_prev_error_signals_shuffler = get_tensor_shuffler(
          m_prev_error_signals_const_view, m_prev_error_signals_t);
      for (int i = 0; i < 3; ++i) {
        m_prev_error_signals_shuffler_last_mb[i] = nullptr;
      }
    } else {
      m_prev_error_signals_t = get_child_layers()[0]->get_error_signals_t(*this);
      assert_always(m_prev_error_signals_t.get_distribution() ==
                    dists[3]);
    }
  }

  void setup_tensors_fwd(const std::array<dc::Dist, dc::num_dists> &dists)
      override {
    Layer::setup_tensors_fwd(dists);
    if (!distconv_enabled()) return;
    setup_prev_activations_tensor(dists);
    setup_activations_tensor(dists);
    setup_activations_copyout_tensor(dists);
    m_ground_truth_t = get_parent_layers()[1]->get_activations_t(*this);
  }

  void setup_tensors_bwd(const std::array<dc::Dist, dc::num_dists> &dists)
      override {
    Layer::setup_tensors_bwd(dists);
    if (!distconv_enabled()) return;
    setup_prev_error_signals_tensor(dists);
    setup_error_signals_tensor(dists);
    setup_error_signals_copyout_tensor(dists);

    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);
    const auto &global_shape = m_ground_truth_t.get_shape();
    const auto &local_shape = m_ground_truth_t.get_local_shape();
    m_d_ground_truth_t = dc::TensorDev(
        global_shape, loc, dists[2], local_shape);
    assert0(m_d_ground_truth_t.allocate());
    m_d_ground_truth_t.zero(dc::get_stream());

    m_cross_entropy = new dc::CrossEntropy(dc::get_backend());
    m_cross_entropy->setup(m_prev_activations_t, m_ground_truth_t,
                           m_activations_t);
  }

  using Layer::get_error_signals_t;

  const dc::TensorDev &get_error_signals_t(const Layer &parent) const {
    const auto parents = get_parent_layers();
    assert_always(parents.size() == 2);
    for (int i = 0; i < (int)parents.size(); ++i) {
      if (parents[i] == &parent) {
        if (i == 0) {
          return m_error_signals_t;
        } else {
          return m_d_ground_truth_t;
        }
      }
    }
    LBANN_ERROR("No such parent found");
  }
#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_CROSS_ENTROPY_LAYER_INSTANTIATE
extern template class cross_entropy_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class cross_entropy_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class cross_entropy_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
extern template class cross_entropy_layer<
  data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_CROSS_ENTROPY_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_CROSS_ENTROPY_HPP_INCLUDED
