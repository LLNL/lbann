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

#ifndef LBANN_LAYERS_MISC_ROWWISE_WEIGHTS_NORMS_HPP_INCLUDED
#define LBANN_LAYERS_MISC_ROWWISE_WEIGHTS_NORMS_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/weights/weights_helpers.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class rowwise_weights_norms_layer : public data_type_layer<TensorDataType> {
public:

  rowwise_weights_norms_layer();
  rowwise_weights_norms_layer(const rowwise_weights_norms_layer& other) = default;
  rowwise_weights_norms_layer& operator=(const rowwise_weights_norms_layer& other) = default;

  rowwise_weights_norms_layer* copy() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

protected:

  void setup_dims(DataReaderMetaData& dr_metadata) override;

  void fp_compute() override;
  void bp_compute() override;

private:

  using LocalMat = El::Matrix<TensorDataType, Device>;
  LocalMat m_local_norms;

  static void row_sqsums(const LocalMat& mat, LocalMat& row_sqsums);
  static void sqrt(LocalMat& mat);
  static void divide(LocalMat& numer, const LocalMat& denom);
  static void row_axpy(
    TensorDataType alpha,
    const LocalMat& a_vec,
    const LocalMat& x_mat,
    TensorDataType beta,
    LocalMat& y_mat);

};

// =========================================================
// Implementation
// =========================================================

template <typename TensorDataType, data_layout Layout, El::Device Device>
rowwise_weights_norms_layer<TensorDataType,Layout,Device>::rowwise_weights_norms_layer()
  : data_type_layer<TensorDataType>(nullptr) {
  this->m_expected_num_parent_layers = 0;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
rowwise_weights_norms_layer<TensorDataType,Layout,Device>* rowwise_weights_norms_layer<TensorDataType,Layout,Device>::copy() const {
  return new rowwise_weights_norms_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string rowwise_weights_norms_layer<TensorDataType,Layout,Device>::get_type() const {
  return "row-wise weights norms";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout rowwise_weights_norms_layer<TensorDataType,Layout,Device>::get_data_layout() const {
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device rowwise_weights_norms_layer<TensorDataType,Layout,Device>::get_device_allocation() const {
  return Device;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType,Layout,Device>::setup_dims(DataReaderMetaData& dr_metadata) {
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);

  // Make sure weights have already been setup by another layer
  if (this->has_weights() != 1) {
    LBANN_ERROR("attempted to setup ",
                this->get_type()," layer \"",this->get_name(),"\" ",
                "with an invalid number of weights ",
                "(expected 1, found ",this->num_weights(),")");
  }
  if (this->get_weights(0).get_matrix_height() <= 0) {
    LBANN_ERROR("attempted to setup ",
                this->get_type()," layer \"",this->get_name(),"\" ",
                "with weights \"",this->get_weights(0).get_name(),"\" ",
                "before weights have been setup ",
                "(consider using hint_layer to force ",
                "another layer to setup the weights first)");
  }

  // Output dimensions are height of weights matrix
  this->set_output_dims(this->get_weights(0).get_matrix_height_dims());

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType,Layout,Device>::fp_compute() {

  // Weights data
  using WeightsType = data_type_weights<TensorDataType>;
  const auto& w = dynamic_cast<const WeightsType&>(this->get_weights(0));
  const auto& weights_matrix = w.get_values();

  // Output tensor
  auto& output = this->get_activations();
  output.AlignWith(weights_matrix);
  if (weights_matrix.LocalHeight() != output.LocalHeight()) {
    LBANN_ERROR(
      "data matrices for ",
      this->get_type()," layer \"",this->get_name(),"\" ",
      "and weights \"",w.get_name(),"\" ",
      "are not aligned or have invalid layouts");
  }

  // Workspace buffers
  /// @todo Synchronize
  m_local_norms.Resize(weights_matrix.LocalHeight(), 1);
  LocalMat ones;
  El::Ones(ones, output.LocalWidth(), 1);

  // Compute norm of each row in weights matrix
  this->row_sqsums(weights_matrix.LockedMatrix(), m_local_norms);
  El::AllReduce(m_local_norms, weights_matrix.RowComm(), El::mpi::SUM);
  this->sqrt(m_local_norms);
  El::Gemm(
    El::NORMAL,
    El::TRANSPOSE,
    El::TypeTraits<TensorDataType>::One(),
    m_local_norms,
    ones,
    El::TypeTraits<TensorDataType>::Zero(),
    output.Matrix());

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rowwise_weights_norms_layer<TensorDataType,Layout,Device>::bp_compute() {

  // Weights data
  using WeightsType = data_type_weights<TensorDataType>;
  auto& w = dynamic_cast<WeightsType&>(this->get_weights(0));
  const auto& weights_matrix = w.get_values();
  auto&& opt = w.get_optimizer();
  if (opt == nullptr) { return; }
  TensorDataType alpha, beta;
  auto& weights_matrix_grad = opt->get_gradient_buffer(beta, alpha, false);

  // Gradient w.r.t. output tensor
  // Note: Assume output grad and weights data are aligned
  const auto& output_grad = this->get_prev_error_signals();
  if (weights_matrix.LocalHeight() != output_grad.LocalHeight()) {
    LBANN_ERROR(
      "data matrices for ",
      this->get_type()," layer \"",this->get_name(),"\" ",
      "and weights \"",w.get_name(),"\" ",
      "are not aligned or have invalid layouts");
  }

  // Workspace buffers
  LocalMat workspace(output_grad.LocalHeight(), 1);
  LocalMat ones;
  El::Ones(ones, output_grad.LocalWidth(), 1);

  // dw/dL = w / norm(w) * sum(dy/dL)
  El::Gemm(
    El::NORMAL,
    El::NORMAL,
    El::TypeTraits<TensorDataType>::One(),
    output_grad.LockedMatrix(),
    ones,
    El::TypeTraits<TensorDataType>::Zero(),
    workspace);
  El::AllReduce(workspace, output_grad.RowComm(), El::mpi::SUM);
  this->divide(workspace, m_local_norms);
  this->row_axpy(
    alpha,
    workspace,
    weights_matrix.LockedMatrix(),
    beta,
    weights_matrix_grad.Matrix());

}

LBANN_DEFINE_LAYER_BUILDER(rowwise_weights_norms);

#ifndef LBANN_ROWWISE_WEIGHTS_NORMS_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                 \
  extern template class rowwise_weights_norms_layer<          \
    T, data_layout::DATA_PARALLEL, Device>;
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_ROWWISE_WEIGHTS_NORMS_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_MISC_ROWWISE_WEIGHTS_NORMS_HPP_INCLUDED
