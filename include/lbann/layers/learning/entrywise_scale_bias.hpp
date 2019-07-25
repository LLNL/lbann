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

#ifndef LBANN_LAYER_LEARNING_ENTRYWISE_SCALE_BIAS_HPP_INCLUDED
#define LBANN_LAYER_LEARNING_ENTRYWISE_SCALE_BIAS_HPP_INCLUDED

#include "lbann/layers/layer.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief Apply scale and bias to tensor entries.
 *
 *  Scale and bias terms are applied independently to each tensor
 *  entry. More precisely, given input, output, scale, and bias
 *  tensors @f$ X,Y,A,B\in\mathbb{R}^{d_1\times\cdots\times d_n} @f$:
 *  @f[
 *    Y = A \circ X + B
 *  @f]
 *
 *  The scale and bias terms are fused into a single weights tensor to
 *  reduce the number of gradient allreduces during backprop. In
 *  particular, the weights tensor is a
 *  @f$ \text{size} \times 2 @f$ matrix, where the first
 *  column correspond to scale terms and the second column to bias
 *  terms.
 */
template <data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class entrywise_scale_bias_layer : public Layer {
public:

  entrywise_scale_bias_layer(lbann_comm *comm)
    : Layer(comm) {}

  entrywise_scale_bias_layer(const entrywise_scale_bias_layer& other)
    : Layer(other),
      m_weights_gradient(other.m_weights_gradient ?
                         other.m_weights_gradient->Copy() : nullptr) {}
  entrywise_scale_bias_layer& operator=(const entrywise_scale_bias_layer& other) {
    Layer::operator=(other);
    m_weights_gradient.reset(other.m_weights_gradient ?
                             other.m_weights_gradient->Copy() :
                             nullptr);
    return *this;
  }

  entrywise_scale_bias_layer* copy() const override {
    return new entrywise_scale_bias_layer(*this);
  }
  std::string get_type() const override { return "entry-wise scale/bias"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  void setup_matrices(const El::Grid& grid) override {
    Layer::setup_matrices(grid);
    auto dist = get_prev_activations().DistData();
    dist.rowDist = El::STAR;
    m_weights_gradient.reset(AbsDistMat::Instantiate(dist));
  }

  void setup_data() override {
    Layer::setup_data();
    const auto dims = get_output_dims();
    const El::Int size = get_output_size();

    // Construct default weights if needed
    if (this->m_weights.size() < 1) {
      this->m_weights.push_back(new weights(get_comm()));
      std::vector<DataType> vals(2*size, DataType{0});
      std::fill(vals.begin(), vals.begin()+size, DataType{1});
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
    dist.rowDist = El::STAR;
    m_weights[0]->set_dims(dims,
                           {static_cast<int>(2)});
    m_weights[0]->set_matrix_distribution(dist);

    // Setup gradient w.r.t. weights
    m_weights_gradient->AlignWith(dist);
    m_weights_gradient->Resize(size, 2);

  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    Layer::fp_setup_outputs(mini_batch_size);

#if 0 /// @todo See https://github.com/LLNL/lbann/issues/1123

    // Check that input and weights tensors are aligned
    /// @todo Realign weights tensor if misaligned
    bool aligned = true;
    try {
      const auto& x = get_prev_activations();
      const auto& w = m_weights[0]->get_values();
      aligned = (x.ColAlign() == w.ColAlign()
                 && x.RowAlign() == w.RowAlign());
    }
    catch (const exception& e) {
      // An exception is thrown if you try accessing weights values
      // before they are initialized. We don't care if this case is
      // aligned, so it's safe to ignore.
    }
    if (!aligned) {
      std::ostringstream err;
      err << this->get_type() << " layer \"" << this->get_name() << "\" "
          << "has misaligned input and weights matrices";
      LBANN_ERROR(err.str());
    }

#endif // 0

  }

  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) {
    Layer::bp_setup_gradient_wrt_inputs(mini_batch_size);
    m_weights_gradient->Empty(false);
    m_weights_gradient->AlignWith(get_prev_activations());
    m_weights_gradient->Resize(get_input_size(), mini_batch_size);
  }

protected:
  void fp_compute() override;
  void bp_compute() override;

private:

  /** Objective function gradient w.r.t. weights. */
  std::unique_ptr<AbsDistMat> m_weights_gradient;

};

} // namespace lbann

#endif // LBANN_LAYER_LEARNING_ENTRYWISE_SCALE_BIAS_HPP_INCLUDED
