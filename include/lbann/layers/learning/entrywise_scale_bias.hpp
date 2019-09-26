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

#include "lbann/layers/data_type_layer.hpp"
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
template <typename TensorDataType, data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class entrywise_scale_bias_layer : public data_type_layer<TensorDataType> {
public:

  entrywise_scale_bias_layer(lbann_comm *comm)
    : data_type_layer<TensorDataType>(comm) {}

  entrywise_scale_bias_layer(const entrywise_scale_bias_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_weights_gradient(other.m_weights_gradient ?
                         other.m_weights_gradient->Copy() : nullptr) {}
  entrywise_scale_bias_layer& operator=(const entrywise_scale_bias_layer& other) {
    data_type_layer<TensorDataType>::operator=(other);
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
    data_type_layer<TensorDataType>::setup_matrices(grid);
    auto dist = this->get_prev_activations().DistData();
    dist.rowDist = El::STAR;
    m_weights_gradient.reset(El::AbstractDistMatrix<TensorDataType>::Instantiate(dist));
  }

  void setup_data() override {
    data_type_layer<TensorDataType>::setup_data();

    // Initialize output dimensions
    this->set_output_dims(this->get_input_dims());
    const auto output_dims = this->get_output_dims();
    const El::Int output_size = this->get_output_size();

    // Construct default weights if needed
    // Note: Scale is initialized to 1 and bias to 0
    if (this->m_weights.empty()) {
      auto w = make_unique<weights<TensorDataType>>(this->get_comm());
      std::vector<DataType> vals(2*output_size, TensorDataType{0});
      std::fill(vals.begin(), vals.begin()+output_size, TensorDataType{1});
      auto init = make_unique<value_initializer>(vals);
      std::unique_ptr<optimizer<TensorDataType>> opt(this->m_model->create_optimizer());
      w->set_name(this->get_name() + "_weights");
      w->set_initializer(std::move(init));
      w->set_optimizer(std::move(opt));
      this->m_weights.push_back(w.get());
      this->m_model->add_weights(std::move(w));
    }
    if (this->m_weights.size() != 1) {
      LBANN_ERROR("attempted to setup ",
                  this->get_type()," layer \"",this->get_name(),"\" ",
                  "with an invalid number of weights ",
                  "(expected 1, found ",this->m_weights.size(),")");
    }

    // Setup weights
    auto dist = this->get_prev_activations().DistData();
    dist.rowDist = El::STAR;
    this->get_weights()[0]->set_dims(output_dims,
                                     {static_cast<int>(2)});
    this->get_weights()[0]->set_matrix_distribution(dist);

    // Setup gradient w.r.t. weights
    m_weights_gradient->AlignWith(dist);
    m_weights_gradient->Resize(output_size, 2);

  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    data_type_layer<TensorDataType>::fp_setup_outputs(mini_batch_size);

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

  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override {
    data_type_layer<TensorDataType>::bp_setup_gradient_wrt_inputs(mini_batch_size);
    m_weights_gradient->Empty(false);
    m_weights_gradient->AlignWith(this->get_prev_activations());
    m_weights_gradient->Resize(this->get_input_size(), 2);
  }

protected:
  void fp_compute() override;
  void bp_compute() override;

private:

  /** Objective function gradient w.r.t. weights. */
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> m_weights_gradient;


  template <typename U>
  friend void fp_compute_impl(entrywise_scale_bias_layer<U, Layout, Device>& l);
  template <typename U>
  friend void bp_compute_impl(entrywise_scale_bias_layer<U, Layout, Device>& l);
};

#ifndef LBANN_ENTRYWISE_SCALE_BIAS_LAYER_INSTANTIATE
extern template class entrywise_scale_bias_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class entrywise_scale_bias_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class entrywise_scale_bias_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
extern template class entrywise_scale_bias_layer<
  data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_ENTRYWISE_SCALE_BIAS_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_LEARNING_ENTRYWISE_SCALE_BIAS_HPP_INCLUDED
