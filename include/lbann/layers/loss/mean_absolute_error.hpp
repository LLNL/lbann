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

#ifndef LBANN_LAYERS_LOSS_MEAN_ABSOLUTE_ERROR_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_MEAN_ABSOLUTE_ERROR_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

/** @brief Mean absolute error
 *
 *  Given a prediction @f$y@f$ and ground truth @f$\hat{y}@f$,
 *  @f[
 *    MAE(y,\hat{y})
 *      = \frac{1}{n} \sum\limits_{i=1}^{n} | y_i - \hat{y}_i |
 *  @f]
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class mean_absolute_error_layer : public data_type_layer<TensorDataType>
{
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:
  mean_absolute_error_layer(lbann_comm* comm)
    : data_type_layer<TensorDataType>(comm)
  {
    this->m_expected_num_parent_layers = 2;
  }

  mean_absolute_error_layer(const mean_absolute_error_layer& other)
    : data_type_layer<TensorDataType>(other)
  {
    m_workspace.reset(other.m_workspace ? other.m_workspace->Copy() : nullptr);
  }

  mean_absolute_error_layer& operator=(const mean_absolute_error_layer& other)
  {
    data_type_layer<TensorDataType>::operator=(other);
    m_workspace.reset(other.m_workspace ? other.m_workspace->Copy() : nullptr);
    return *this;
  }

  mean_absolute_error_layer* copy() const override
  {
    return new mean_absolute_error_layer(*this);
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "mean absolute error"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS;
  }

  void setup_dims(DataReaderMetaData& dr_metadata) override;

  void setup_data(size_t max_mini_batch_size) override;

  void fp_compute() override;

  void bp_compute() override;

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  mean_absolute_error_layer() : mean_absolute_error_layer(nullptr) {}

private:
  /** Compute local contributions to mean absolute error loss. */
  void local_fp_compute();
  /** Compute local gradients. */
  void local_bp_compute();

  /** Workspace matrix. */
  std::unique_ptr<AbsDistMatrixType> m_workspace;
};

template <typename T, data_layout L, El::Device D>
void mean_absolute_error_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_mean_absolute_error();
}

#ifndef LBANN_MEAN_ABSOLUTE_ERROR_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                                                \
  extern template class mean_absolute_error_layer<T,                           \
                                                  data_layout::DATA_PARALLEL,  \
                                                  Device>;                     \
  extern template class mean_absolute_error_layer<T,                           \
                                                  data_layout::MODEL_PARALLEL, \
                                                  Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_MEAN_ABSOLUTE_ERROR_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_MEAN_ABSOLUTE_ERROR_HPP_INCLUDED
