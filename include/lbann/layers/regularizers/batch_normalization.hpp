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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED

#include "lbann_config.hpp"
#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/models/model.hpp"
#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/cuda.hpp"
#include "lbann/utils/distconv.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

enum class batch_normalization_stats_aggregation {
  /** Statistics are aggregated only within a single rank. */
  local,
  /** Statistics are aggregated among every rank in a single node. */
  node_local,
  /** Statistics are aggregated among every rank in the model. */
  global
};

/** @brief
 *
 *  Each input channel is normalized across the mini-batch to have
 *  zero mean and unit standard deviation. Learned scaling factors and
 *  biases are then applied. This uses the standard approach of
 *  maintaining the running mean and standard deviation (with
 *  exponential decay) for use at test time. See:
 *
 *  Sergey Ioffe and Christian Szegedy. "Batch Normalization:
 *  Accelerating Deep Network Training by Reducing Internal Covariate
 *  Shift." In International Conference on Machine Learning,
 *  pp. 448-456. 2015.
 */
template <data_layout T_layout, El::Device Dev>
class batch_normalization_layer : public regularizer_layer {

private:

  /** Decay rate for the running statistics. */
  DataType m_decay;
  /** Small number to avoid division by zero. */
  DataType m_epsilon;
  /** Type of statistics aggregation to use. */
  batch_normalization_stats_aggregation m_stats_aggregation;
  /**
   * Cache of node-local num_per_sum results for node-local stats.
   * Indexed by effective mini-batch size.
   */
  std::unordered_map<El::Int, El::Int> m_num_per_sum_cache;

  /** Current minibatch means. */
  std::unique_ptr<AbsDistMat> m_mean;
  /** Current minibatch standard deviations. */
  std::unique_ptr<AbsDistMat> m_var;
  /** Gradient w.r.t. means. */
  std::unique_ptr <AbsDistMat> m_mean_gradient;
  /** Gradient w.r.t. standard deviations. */
  std::unique_ptr<AbsDistMat> m_var_gradient;
  /** Gradient w.r.t. scaling terms. */
  std::unique_ptr<AbsDistMat> m_scale_gradient;
  /** Gradient w.r.t. bias terms. */
  std::unique_ptr<AbsDistMat> m_bias_gradient;

public:
  /**
   * Set up batch normalization.
   * @param decay Controls the momentum of the running mean/standard
   * deviation averages.
   * @param epsilon A small number to avoid division by zero.
   * @param use_global_stats Whether to use global statistics when
   * training.
   */
  batch_normalization_layer(lbann_comm *comm,
                            DataType decay=0.9,
                            DataType epsilon=1e-5,
                            batch_normalization_stats_aggregation stats_aggregation =
                            batch_normalization_stats_aggregation::local)
    : regularizer_layer(comm),
      m_decay(decay),
      m_epsilon(epsilon),
      m_stats_aggregation(stats_aggregation) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "batch normalization only supports DATA_PARALLEL");
#ifdef LBANN_DETERMINISTIC
    // Force global computation.
    m_stats_aggregation = batch_normalization_stats_aggregation::global;
#endif
  }

  batch_normalization_layer(const batch_normalization_layer& other)
    : regularizer_layer(other),
      m_decay(other.m_decay),
      m_epsilon(other.m_epsilon),
      m_stats_aggregation(other.m_stats_aggregation),
      m_num_per_sum_cache(other.m_num_per_sum_cache),
      m_mean(other.m_mean ? other.m_mean->Copy() : nullptr),
      m_var(other.m_var ? other.m_var->Copy() : nullptr),
      m_mean_gradient(other.m_mean_gradient ?
                      other.m_mean_gradient->Copy() : nullptr),
      m_var_gradient(other.m_var_gradient ?
                     other.m_var_gradient->Copy() : nullptr),
      m_scale_gradient(other.m_scale_gradient ?
                       other.m_scale_gradient->Copy() : nullptr),
      m_bias_gradient(other.m_bias_gradient ?
                      other.m_bias_gradient->Copy() : nullptr) {}

  batch_normalization_layer& operator=(const batch_normalization_layer& other) {
    regularizer_layer::operator=(other);
    m_decay = other.m_decay;
    m_epsilon = other.m_epsilon;
    m_stats_aggregation = other.m_stats_aggregation;
    m_num_per_sum_cache = other.m_num_per_sum_cache;

    // Deep copy matrices
    m_mean.reset(other.m_mean ? other.m_mean->Copy() : nullptr);
    m_var.reset(other.m_var ? other.m_var->Copy() : nullptr);
    m_mean_gradient.reset(other.m_mean_gradient ?
                          other.m_mean_gradient->Copy() : nullptr);
    m_var_gradient.reset(other.m_var_gradient ?
                         other.m_var_gradient->Copy() : nullptr);
    m_scale_gradient.reset(other.m_scale_gradient ?
                           other.m_scale_gradient->Copy() : nullptr);
    m_bias_gradient.reset(other.m_bias_gradient ?
                          other.m_bias_gradient->Copy() : nullptr);

    return *this;
  }

  batch_normalization_layer* copy() const override { return new batch_normalization_layer(*this); }
  std::string get_type() const override { return "batch normalization"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto&& desc = regularizer_layer::get_description();
    desc.add("Decay", m_decay);
    desc.add("Epsilon", m_epsilon);
    switch (m_stats_aggregation) {
    case batch_normalization_stats_aggregation::local:
      desc.add("Statistics aggregation", "local");
      break;
    case batch_normalization_stats_aggregation::node_local:
      desc.add("Statistics aggregation", "node-local");
      break;
    case batch_normalization_stats_aggregation::global:
      desc.add("Statistics aggregation", "global");
      break;
    }
    return desc;
  }

protected:

  void setup_matrices(const El::Grid& grid) override {
    regularizer_layer::setup_matrices(grid);
    m_mean.reset(new StarMat<Dev>(grid));
    m_var.reset(new StarMat<Dev>(grid));
    m_mean_gradient.reset(new StarMat<Dev>(grid));
    m_var_gradient.reset(new StarMat<Dev>(grid));
    m_scale_gradient.reset(new StarMat<Dev>(grid));
    m_bias_gradient.reset(new StarMat<Dev>(grid));
  }

  void setup_dims() override {
    regularizer_layer::setup_dims();
    set_output_dims(get_input_dims());
  }

  void setup_data() override {
    regularizer_layer::setup_data();
    const auto& output_dims = get_output_dims();
    const auto& num_channels = output_dims[0];

    // Display warning if mini-batch size is small
    const auto& output = get_activations();
    const auto& mini_batch_size = output.Width();
    const auto& local_mini_batch_size = mini_batch_size / output.DistSize();
    if (m_stats_aggregation == batch_normalization_stats_aggregation::global
        && mini_batch_size <= 4) {
      std::stringstream err;
      err << "LBANN warning: "
          << get_type() << " layer \"" << get_name() << "\" "
          << "is using global statistics and "
          << "the mini-batch size (" << mini_batch_size << ") "
          << "may be too small to get good statistics";
      if (output.DistRank() == 0) {
        std::cerr << err.str() << std::endl;
      }
    } else if (m_stats_aggregation == batch_normalization_stats_aggregation::node_local
               && local_mini_batch_size*m_comm->get_procs_per_node() <= 4) {
      std::stringstream err;
      err << "LBANN warning: "
          << get_type() << " layer \"" << get_name() << "\" "
          << "is using node-local statistics and "
          << "the node-local mini-batch size ("
          << (local_mini_batch_size*m_comm->get_procs_per_node()) << ") "
          << "may be too small to get good statistics";
      if (output.DistRank() == 0) {
        std::cerr << err.str() << std::endl;
      }
    } else if (m_stats_aggregation == batch_normalization_stats_aggregation::local
               && local_mini_batch_size <= 4) {
      std::stringstream err;
      err << "LBANN warning: "
          << get_type() << " layer \"" << get_name() << "\" "
          << "is using local statistics and "
          << "the local mini-batch size (" << local_mini_batch_size << ") "
          << "may be too small to get good statistics";
      if (output.DistRank() == 0) {
        std::cerr << err.str() << std::endl;
      }
    }

    // Initialize default weights if none are provided
    if (this->m_weights.size() > 4) {
      std::stringstream err;
      err << "attempted to setup layer \"" << m_name << "\" "
          << "with an invalid number of weights";
      LBANN_ERROR(err.str());
    }
    this->m_weights.resize(4, nullptr);
    if (this->m_weights[0] == nullptr) {
      this->m_weights[0] = new weights(get_comm());
      std::unique_ptr<weights_initializer> init(new constant_initializer(DataType(1)));
      std::unique_ptr<optimizer> opt(m_model->create_optimizer());
      this->m_weights[0]->set_name(get_name() + "_scale");
      this->m_weights[0]->set_initializer(init);
      this->m_weights[0]->set_optimizer(opt);
      this->m_model->add_weights(this->m_weights[0]);
    }
    if (this->m_weights[1] == nullptr) {
      this->m_weights[1] = new weights(get_comm());
      std::unique_ptr<weights_initializer> init(new constant_initializer(DataType(0)));
      std::unique_ptr<optimizer> opt(m_model->create_optimizer());
      this->m_weights[1]->set_name(get_name() + "_bias");
      this->m_weights[1]->set_initializer(init);
      this->m_weights[1]->set_optimizer(opt);
      this->m_model->add_weights(this->m_weights[1]);
    }
    if (this->m_weights[2] == nullptr) {
      this->m_weights[2] = new weights(get_comm());
      this->m_weights[2]->set_name(get_name() + "_running_mean");
      std::unique_ptr<weights_initializer> init(new constant_initializer(DataType(0)));
      this->m_weights[2]->set_initializer(init);
      this->m_model->add_weights(this->m_weights[2]);
    }
    if (this->m_weights[3] == nullptr) {
      this->m_weights[3] = new weights(get_comm());
      this->m_weights[3]->set_name(get_name() + "_running_variance");
      std::unique_ptr<weights_initializer> init(new constant_initializer(DataType(1)));
      this->m_weights[3]->set_initializer(init);
      this->m_model->add_weights(this->m_weights[3]);
    }

    // Setup weights
    auto dist = get_prev_activations().DistData();
    dist.colDist = El::STAR;
    dist.rowDist = El::STAR;
    for (auto* w : this->m_weights) {
      w->set_dims(num_channels);
      w->set_matrix_distribution(dist);
    }

    // Initialize matrices
    El::Zeros(*m_mean,           num_channels, 1);
    El::Zeros(*m_var,            num_channels, 1);
    El::Zeros(*m_mean_gradient,  num_channels, 1);
    El::Zeros(*m_var_gradient,   num_channels, 1);
    El::Zeros(*m_scale_gradient, num_channels, 1);
    El::Zeros(*m_bias_gradient,  num_channels, 1);

    // Initialize freeze state
    for (auto&& w : this->m_weights) {
      if (m_frozen) {
        w->freeze();
      } else {
        w->unfreeze();
      }
    }
    for (auto&& w : this->m_weights) {
      if (w->is_frozen() != m_frozen) {
        std::stringstream err;
        err << (m_frozen ? "" : "un") << "frozen "
            << "layer \"" << get_name() << "\" has "
            << (w->is_frozen() ? "" : "un") << "frozen "
            << "weights \"" << w->get_name() << "\"";
        LBANN_ERROR(err.str());
      }
    }

  }

  void fp_compute() override;
  void bp_compute() override;

#ifdef LBANN_HAS_DISTCONV
 protected:
  void fp_compute_distconv();
  void bp_compute_distconv();

  dc::BatchNormalization *m_bn;
  dc::TensorDev m_mean_t;
  dc::TensorDev m_var_t;
  dc::TensorDev m_scale_t;
  dc::TensorDev m_bias_t;
  dc::TensorDev m_running_mean_t;
  dc::TensorDev m_running_var_t;
  dc::TensorDev m_mean_gradient_t;
  dc::TensorDev m_var_gradient_t;
  dc::TensorDev m_scale_gradient_t;
  dc::TensorDev m_bias_gradient_t;

  dc::LocaleMPI m_spatial_loc;

  void setup_tensors_fwd(const std::array<dc::Dist, 4> &dists) override {
    Layer::setup_tensors_fwd(dists);
    if (!distconv_enabled()) return;

    setup_prev_activations_tensor(dists);
    setup_activations_tensor(dists);
    setup_activations_copyout_tensor(dists);

    dc::MPIPrintStreamDebug()
        << "BN prev_activations: " << m_prev_activations_t
        << ", activations: " << m_activations_t;

    const int num_channels = this->get_output_dims()[0];
    // Sanity check that the shared tensors have the correct shape
    assert_ne(num_channels, 0);
    assert_eq(m_mean->Matrix().Width() * m_mean->Matrix().Height(),
              num_channels);

    dc::Array4 per_channel_stat_shape = {1, 1, num_channels, 1};
    dc::Dist shared_dist(dists[0].get_locale_shape());
    auto split_shape = dists[0].get_split_shape();
    // set all dimensions to be 1 except for the channel dimension
    auto pc = split_shape[-2];
    split_shape = 1;
    split_shape[-2] = pc;
    shared_dist.set_split_shape(split_shape);

    const dc::LocaleMPI loc(dc::get_mpi_comm(), false);

    // mean
    m_mean_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    assert0(dc::tensor::View(m_mean_t, this->m_mean->Buffer()));
    // var
    m_var_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    assert0(dc::tensor::View(m_var_t, this->m_var->Buffer()));
    // scale: view to weights[0]
    m_scale_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    // bias: view to weights[1]
    m_bias_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    // running_mean: view to weights[2]
    m_running_mean_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    // running_var: view to weights[3]
    m_running_var_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    // scale_gradient
    m_scale_gradient_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    assert0(dc::tensor::View(
        m_scale_gradient_t, this->m_scale_gradient->Buffer()));
    // bias_gradient
    m_bias_gradient_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    assert0(dc::tensor::View(
        m_bias_gradient_t, this->m_bias_gradient->Buffer()));
    // mean_gradient
    m_mean_gradient_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    assert0(dc::tensor::View(
        m_mean_gradient_t, this->m_mean_gradient->Buffer()));
    // var_gradient
    m_var_gradient_t = dc::TensorDev(per_channel_stat_shape, loc, shared_dist);
    assert0(dc::tensor::View(
        m_var_gradient_t, this->m_var_gradient->Buffer()));

    // spatial decomposition requires global communication
    // m_use_global_stats = true;
    if (dc::use_partial_aggregation_in_bn()) {
      m_spatial_loc = m_mean_t.get_spatial_locale();
    }
  }

  void setup_tensors_bwd(const std::array<dc::Dist, 4> &dists) override {
    Layer::setup_tensors_bwd(dists);
    if (!distconv_enabled()) return;

    setup_prev_error_signals_tensor(dists);
    setup_error_signals_tensor(dists);
    setup_error_signals_copyout_tensor(dists);

    dc::tensor::Array<4, bool> reduced_dims(false);
    if (m_use_global_stats) {
      reduced_dims = true;
    } else if (dc::use_partial_aggregation_in_bn()) {
      for (int i = 0; i < dc::TensorDev::num_spatial_dims; ++i) {
        reduced_dims[i] = true;
      }
    }

    m_bn = new dc::BatchNormalization(
        dc::get_backend(), m_decay, m_epsilon,
        reduced_dims);

    dc::MPIPrintStreamDebug()
        << "BN prev_error_signals: " << m_prev_error_signals_t
        << ", error_signals: " << m_error_signals_t << "\n";
  }
#endif

};

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
