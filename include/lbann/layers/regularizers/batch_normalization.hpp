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

#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/models/model.hpp"

namespace lbann {

/** Batch normalization layer.
 *  Each input channel is normalized across the mini-batch to have
 *  zero mean and unit standard deviation. Learned scaling factors and
 *  biases are then applied. See:
 *    Sergey Ioffe and Christian Szegedy. "Batch Normalization:
 *    Accelerating Deep Network Training by Reducing Internal
 *    Covariate Shift." ICML 2015.
 *  This uses the standard approach of maintaining the running mean
 *  and standard deviation (with exponential decay) for use at test
 *  time. See:
 *    https://cthorey.github.io/backpropagation/
 */
template <data_layout T_layout, El::Device Dev>
class batch_normalization_layer : public regularizer_layer {

 private:

  /** Decay rate for the running statistics. */
  DataType m_decay;
  /** Small number to avoid division by zero. */
  DataType m_epsilon;
  /** Whether to use global statistics when training. */
  bool m_use_global_stats;

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
                            bool use_global_stats = false)
    : regularizer_layer(comm),
      m_decay(decay),
      m_epsilon(epsilon),
      m_use_global_stats(use_global_stats) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "batch normalization only supports DATA_PARALLEL");
#ifdef LBANN_DETERMINISTIC
    // Force global computation.
    m_use_global_stats = true;
#endif
  }

  batch_normalization_layer(const batch_normalization_layer& other)
    : regularizer_layer(other),
      m_decay(other.m_decay),
      m_epsilon(other.m_epsilon),
      m_use_global_stats(other.m_use_global_stats),
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
    m_use_global_stats = other.m_use_global_stats;

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

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream ss;
    ss << " batch_normalization; "
       << "decay: " << m_decay
       << " epsilon : " << m_epsilon
       << " data_layout: " << get_data_layout_string(get_data_layout());
    return ss.str();
  }

  batch_normalization_layer* copy() const override { return new batch_normalization_layer(*this); }

  std::string get_type() const override { return "batch normalization"; }

  void setup_matrices(const El::Grid& grid) override {
    regularizer_layer::setup_matrices(grid);
    m_mean.reset(new StarMat<Dev>(grid));
    m_var.reset(new StarMat<Dev>(grid));
    m_mean_gradient.reset(new StarMat<Dev>(grid));
    m_var_gradient.reset(new StarMat<Dev>(grid));
    m_scale_gradient.reset(new StarMat<Dev>(grid));
    m_bias_gradient.reset(new StarMat<Dev>(grid));
  }

  data_layout get_data_layout() const override { return T_layout; }

  El::Device get_device_allocation() const override { return Dev; }

  void setup_data() override {
    regularizer_layer::setup_data();
    const auto& output_dims = get_output_dims();
    const auto& num_channels = output_dims[0];

    // Display warning if mini-batch size is small
    const auto& output = get_activations();
    const auto& mini_batch_size = output.Width();
    const auto& local_mini_batch_size = mini_batch_size / output.DistSize();
    if (m_use_global_stats && mini_batch_size <= 4) {
      std::stringstream err;
      err << "LBANN warning: "
          << get_type() << " layer \"" << get_name() << "\" "
          << "is using global statistics and "
          << "the mini-batch size (" << mini_batch_size << ") "
          << "may be too small to get good statistics";
      if (output.DistRank() == 0) {
        std::cerr << err.str() << std::endl;
      }
    } else if (!m_use_global_stats && local_mini_batch_size <= 4) {
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

};

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_BATCH_NORMALIZATION_HPP_INCLUDED
