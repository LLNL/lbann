////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYERS_LOSS_MEAN_SQUARED_ERROR_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_MEAN_SQUARED_ERROR_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
//#include "lbann/utils/distconv.hpp"
#include "lbann/proto/layers.pb.h"

#ifdef LBANN_HAS_DISTCONV
#include "lbann/utils/distconv.hpp"
#include "distconv/dnn_backend/mean_squared_error.hpp"
#endif

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
namespace dc {
using Backend = ::distconv::BackendDNNLib;
using MeanSquaredError = ::distconv::MeanSquaredError<Backend>;
} // namespace dc

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class mean_squared_error_distconv_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;
  mean_squared_error_distconv_adapter(Layer& layer)
      : data_type_distconv_adapter<TensorDataType>(layer) {}
  virtual ~mean_squared_error_distconv_adapter() = default;
  void setup_distributions(tensor_overlap_constraints &constraints) override;
  dc::Shape get_prev_activations_shape(int index) const override;
  dc::Shape get_activations_shape(int index) const override;
  dc::Shape get_activations_local_shape(int index) const override;
  void setup_layer(size_t workspace_capacity) override;
  std::unique_ptr<dc::MeanSquaredError> m_mean_squared_error;
};
#endif // LBANN_HAS_DISTCONV

/** @brief Mean squared error
 *
 *  Given a prediction @f$y@f$ and ground truth @f$\hat{y}@f$,
 *  @f[
 *    MSE(y,\hat{y})
 *      = \frac{1}{n} \sum\limits_{i=1}^{n} (y_i - \hat{y}_i)^2
 *  @f]
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class mean_squared_error_layer : public data_type_layer<TensorDataType> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:

  mean_squared_error_layer(lbann_comm *comm) : data_type_layer<TensorDataType>(comm) {
    this->m_expected_num_parent_layers = 2;
  }

  mean_squared_error_layer(const mean_squared_error_layer& other)
    : data_type_layer<TensorDataType>(other) {
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() :
                      nullptr);
  }

  mean_squared_error_layer& operator=(const mean_squared_error_layer& other) {
    data_type_layer<TensorDataType>::operator=(other);
    m_workspace.reset(other.m_workspace ?
                      other.m_workspace->Copy() :
                      nullptr);
    return *this;
  }

  mean_squared_error_layer* copy() const override { return new mean_squared_error_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "mean squared error"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

#ifdef LBANN_HAS_ONNX
  void fill_onnx_node(onnx::GraphProto& graph) const override;
#endif // LBANN_HAS_ONNX

  void setup_dims(DataReaderMetaData& dr_metadata) override;//  {
//     data_type_layer<TensorDataType>::setup_dims(dr_metadata);
//     this->set_output_dims({1});

// #ifdef LBANN_HAS_DISTCONV
//     // In the current implementation of mean squared error in Distconv, we
//     // do not use the reshape layer and just assumes both inputs have
//     // the matching shape. Therefore, the following check on the input
//     // dimensions would fail. We could address this by either 1)
//     // implementing the reshape layer, or 2) giving a proper shape to
//     // the ground-truth data.
//     //
//     if (this->distconv_enabled()) {
//       return;
//     }
// #endif

//     // Check that input dimensions match
//     if (this->get_input_dims(0) != this->get_input_dims(1)) {
//       const auto& parents = this->get_parent_layers();
//       std::stringstream err;
//       err << get_type() << " layer \"" << this->get_name() << "\" "
//           << "has input tensors with different dimensions (";
//       for (int i = 0; i < this->get_num_parents(); ++i) {
//         const auto& dims = this->get_input_dims(i);
//         err << (i > 0 ? ", " : "")
//             << "layer \"" << parents[i]->get_name() << "\" outputs ";
//         for (size_t j = 0; j < dims.size(); ++j) {
//           err << (j > 0 ? " x " : "") << dims[j];
//         }
//       }
//       err << ")";
//       LBANN_ERROR(err.str());
//     }

//   }

  void setup_data(size_t max_mini_batch_size) override;//  {
//     data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

//     // Initialize workspace
//     const auto& input_dist = this->get_prev_activations(0).DistData();
//     m_workspace.reset(AbsDistMatrixType::Instantiate(*input_dist.grid,
//                                               input_dist.root,
//                                               El::STAR,
//                                               input_dist.rowDist,
//                                               (input_dist.blockHeight == 1
//                                                && input_dist.blockWidth == 1 ?
//                                                El::ELEMENT : El::BLOCK),
//                                               input_dist.device));
// #ifdef HYDROGEN_HAVE_CUB
//     if (m_workspace->GetLocalDevice() == El::Device::GPU) {
//       m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
//     }
// #endif // HYDROGEN_HAVE_CUB

//   }

  void fp_compute() override;//  {

// #ifdef LBANN_HAS_DISTCONV
//     if (this->distconv_enabled()) {
//       fp_compute_distconv();
//       return;
//     }
// #endif // LBANN_HAS_DISTCONV

//     // Initialize workspace
//     m_workspace->Empty();
//     m_workspace->AlignWith(this->get_prev_activations());
//     m_workspace->Resize(1, this->get_prev_activations().Width());

//     // Compute local contributions and accumulate
//     /// @todo Consider reduce rather than allreduce
//     local_fp_compute();
//     this->get_comm()->allreduce(*m_workspace, m_workspace->RedundantComm());
//     El::Copy(*m_workspace, this->get_activations());

//     // Clean up
//     m_workspace->Empty();

//   }

  void bp_compute() override;//  {

// #ifdef LBANN_HAS_DISTCONV
//     if (this->distconv_enabled()) {
//       bp_compute_distconv();
//       return;
//     }
// #endif // LBANN_HAS_DISTCONV

//     // Initialize workspace
//     m_workspace->Empty();
//     m_workspace->AlignWith(this->get_prev_activations());
//     El::Copy(this->get_prev_error_signals(), *m_workspace);

//     // Compute local gradients
//     local_bp_compute();

//     // Clean up
//     m_workspace->Empty();

//   }

protected:

  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  mean_squared_error_layer()
    : mean_squared_error_layer(nullptr)
  {}

private:

  /** Compute local contributions to mean squared error loss. */
  void local_fp_compute();
  /** Compute local gradients. */
  void local_bp_compute();

  /** Workspace matrix. */
  std::unique_ptr<AbsDistMatrixType> m_workspace;

#ifdef LBANN_HAS_DISTCONV
  friend class mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>;
 protected:
  bool is_distconv_supported() const override {
    return Dev == El::Device::GPU && T_layout == data_layout::DATA_PARALLEL;
  }

  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override {
    this->get_distconv_adapter_ptr() = std::make_unique<
      mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>>(*this);
  }

  mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() override;
  const mean_squared_error_distconv_adapter<TensorDataType, T_layout, Dev>& get_distconv_adapter() const override;

  void fp_compute_distconv() {
    assert_always(this->distconv_enabled());
    get_distconv_adapter().m_mean_squared_error->forward(this->get_distconv_adapter().get_prev_activations(0),
                                                         this->get_distconv_adapter().get_prev_activations(1),
                                                         this->get_distconv_adapter().get_activations());
  }

  void bp_compute_distconv() {
    assert_always(this->distconv_enabled());
    get_distconv_adapter().m_mean_squared_error->backward(this->get_distconv_adapter().get_prev_activations(0),
                                                          this->get_distconv_adapter().get_prev_activations(1),
                                                          this->get_distconv_adapter().get_prev_error_signals(0),
                                                          this->get_distconv_adapter().get_error_signals(0),
                                                          this->get_distconv_adapter().get_error_signals(1));
  }
#endif // LBANN_HAS_DISTCONV
};

#ifndef LBANN_MEAN_SQUARED_ERROR_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                     \
  extern template class mean_squared_error_layer<   \
    T, data_layout::DATA_PARALLEL, Device>;         \
  extern template class mean_squared_error_layer<   \
    T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_MEAN_SQUARED_ERROR_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_MEAN_SQUARED_ERROR_HPP_INCLUDED
