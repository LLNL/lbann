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

#ifndef LBANN_LAYER_LEARNING_CHANNELWISE_SCALE_BIAS_HPP_INCLUDED
#define LBANN_LAYER_LEARNING_CHANNELWISE_SCALE_BIAS_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief Apply scale and bias to tensor channels.
 *
 *  The input tensor is sliced along the first tensor dimension (the
 *  "channel" dimension, assuming image data in CHW format) and scale
 *  and bias terms are applied independently to each slice. More
 *  precisely, given input and output tensors
 *  @f$ X,Y\in\mathbb{R}^{d_1\times\cdots\times d_n} @f$
 *  and scale and bias vectors @f$ a,b\in\mathbb{R}^{d_1} @f$:
 *  @f[
 *    Y_{i,j,\cdots} = a_i X_{i,j,\cdots} + b_i
 *  @f]
 *
 *  The scale and bias vectors are fused into a single weights tensor
 *  to reduce the number of gradient allreduces during backprop. In
 *  particular, the weights tensor is a
 *  @f$ \text{num_channels} \times 2 @f$ matrix, where the first
 *  column correspond to scale terms and the second column to bias
 *  terms.
 */
template <data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class channelwise_scale_bias_layer : public Layer {
public:

  channelwise_scale_bias_layer(lbann_comm *comm)
    : Layer(comm) {
    static_assert(Layout == data_layout::DATA_PARALLEL,
                  "channelwise_mean_layer only supports "
                  "data-parallel data layout");
  }

  channelwise_scale_bias_layer(const channelwise_scale_bias_layer& other)
    : Layer(other),
      m_weights_gradient(other.m_weights_gradient ?
                         other.m_weights_gradient->Copy() : nullptr) {}
  channelwise_scale_bias_layer& operator=(const channelwise_scale_bias_layer& other) {
    Layer::operator=(other);
    m_weights_gradient.reset(other.m_weights_gradient ?
                             other.m_weights_gradient->Copy() :
                             nullptr);
    return *this;
  }

  channelwise_scale_bias_layer* copy() const override {
    return new channelwise_scale_bias_layer(*this);
  }
  std::string get_type() const override { return "channel-wise scale/bias"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  void setup_matrices(const El::Grid& grid) override {
    Layer::setup_matrices(grid);
    m_weights_gradient.reset(new StarMat<Device>(grid));
  }

  void setup_data() override {
    Layer::setup_data();
    const El::Int num_channels = get_output_dims()[0];

    // Construct default weights if needed
    if (this->m_weights.size() < 1) {
      this->m_weights.push_back(new weights(get_comm()));
      std::vector<DataType> vals(2*num_channels, DataType{0});
      std::fill(vals.begin(), vals.begin()+num_channels, DataType{1});
      std::unique_ptr<weights_initializer> init(new value_initializer(vals));
      std::unique_ptr<optimizer> opt(m_model->create_optimizer());
      this->m_weights[0]->set_name(get_name() + "_weights");
      this->m_weights[0]->set_initializer(init);
      this->m_weights[0]->set_optimizer(opt);
      this->m_model->add_weights(this->m_weights[0]);
    }
    if (this->m_weights.size() != 1) {
      std::ostringstream err;
      err << "attempted to setup "
          << this->get_type() << " layer \"" << this->get_name() << "\" "
          << "with an invalid number of weights "
          << "(expected 1, "
          << "found " << this->m_weights.size() << ")";
      LBANN_ERROR(err.str());
    }

    // Setup weights
    auto dist = get_prev_activations().DistData();
    dist.colDist = El::STAR;
    dist.rowDist = El::STAR;
    m_weights[0]->set_dims({static_cast<int>(num_channels)},
                           {static_cast<int>(2)});
    m_weights[0]->set_matrix_distribution(dist);

    // Setup gradient w.r.t. weights
    m_weights_gradient->AlignWith(dist);
    m_weights_gradient->Resize(num_channels, 2);

  }

protected:
  void fp_compute() override;
  void bp_compute() override;

private:

  /** Objective function gradient w.r.t. weights. */
  std::unique_ptr<AbsDistMat> m_weights_gradient;

};

} // namespace lbann

#endif // LBANN_LAYER_LEARNING_CHANNELWISE_SCALE_BIAS_HPP_INCLUDED
