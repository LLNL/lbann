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

#ifndef LBANN_LAYERS_MISC_ROWWISE_WEIGHTS_NORMS_HPP_INCLUDED
#define LBANN_LAYERS_MISC_ROWWISE_WEIGHTS_NORMS_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"
#include "lbann/utils/exception.hpp"
#include "lbann/weights/weights_helpers.hpp"

namespace lbann {

/** @brief L2 norm of each row of a weights matrix.
 *
 *  @warning This layer is experimental and finnicky. It is intended
 *  for use with the matrix weights from a fully-connected layer, and
 *  other use-cases may have strange behavior.
 *
 *  Given a weights object, this layer computes the L2 norm for each
 *  row of the underlying matrix. Note that the internal matrix may
 *  have different dimensions than the logical weight dimensions.
 *
 *  This layer expects to have one weights object. During setup, that
 *  weights object should be initialized by another layer before this
 *  layer's setup phase. Setting a "hint layer" may be necessary to
 *  enforce this ordering.
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class rowwise_weights_norms_layer : public data_type_layer<TensorDataType>
{
public:
  rowwise_weights_norms_layer();
  rowwise_weights_norms_layer(const rowwise_weights_norms_layer& other) =
    default;
  rowwise_weights_norms_layer&
  operator=(const rowwise_weights_norms_layer& other) = default;

  rowwise_weights_norms_layer* copy() const override;

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | WEIGHTS;
  }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  void setup_dims(DataReaderMetaData& dr_metadata) override;

  void fp_compute() override;
  void bp_compute() override;

private:
  using LocalMat = El::Matrix<TensorDataType, Device>;
  LocalMat m_local_norms;

  static void row_sqsums(const LocalMat& mat, LocalMat& row_sqsums);
  static void sqrt(LocalMat& mat);
  static void divide(LocalMat& numer, const LocalMat& denom);
  static void row_axpy(TensorDataType alpha,
                       const LocalMat& a_vec,
                       const LocalMat& x_mat,
                       TensorDataType beta,
                       LocalMat& y_mat);
};

template <typename T, data_layout L, El::Device D>
void rowwise_weights_norms_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_rowwise_weights_norms();
}

#ifndef LBANN_ROWWISE_WEIGHTS_NORMS_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class rowwise_weights_norms_layer<                           \
    T,                                                                         \
    data_layout::DATA_PARALLEL,                                                \
    Device>;                                                                   \
  extern template class rowwise_weights_norms_layer<                           \
    T,                                                                         \
    data_layout::MODEL_PARALLEL,                                               \
    Device>;
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_ROWWISE_WEIGHTS_NORMS_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_MISC_ROWWISE_WEIGHTS_NORMS_HPP_INCLUDED
