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

#define LBANN_BATCHWISE_REDUCE_SUM_LAYER_INSTANTIATE
#include "lbann/layers/transform/batchwise_reduce_sum.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
batchwise_reduce_sum_layer<TensorDataType, Layout, Device>::
  batchwise_reduce_sum_layer()
  : data_type_layer<TensorDataType>(nullptr)
{}

template <typename TensorDataType, data_layout Layout, El::Device Device>
batchwise_reduce_sum_layer<TensorDataType, Layout, Device>*
batchwise_reduce_sum_layer<TensorDataType, Layout, Device>::copy() const
{
  return new batchwise_reduce_sum_layer(*this);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::string
batchwise_reduce_sum_layer<TensorDataType, Layout, Device>::get_type() const
{
  return "batch-wise reduce-sum";
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
data_layout
batchwise_reduce_sum_layer<TensorDataType, Layout, Device>::get_data_layout()
  const
{
  return Layout;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
El::Device batchwise_reduce_sum_layer<TensorDataType, Layout, Device>::
  get_device_allocation() const
{
  return Device;
}

template <typename T, data_layout L, El::Device D>
void batchwise_reduce_sum_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_batchwise_reduce_sum();
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void batchwise_reduce_sum_layer<TensorDataType, Layout, Device>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);
  this->set_output_dims(this->get_input_dims());
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void batchwise_reduce_sum_layer<TensorDataType, Layout, Device>::fp_compute()
{

  // Data tensors
  // Note: Assume input and output are aligned.
  const auto& input = this->get_prev_activations();
  const auto& local_input = input.LockedMatrix();
  auto& local_output = this->get_local_activations();

  // Temporary buffers
  using LocalMat = El::Matrix<TensorDataType, Device>;
  LocalMat sums, ones;
  sums.Resize(local_input.Height(), 1);
  El::Ones(ones, local_input.Width(), 1);

  // Local sums
  if (local_input.IsEmpty()) {
    El::Zero(sums);
  }
  else {
    El::Gemm(El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             local_input,
             ones,
             El::TypeTraits<TensorDataType>::Zero(),
             sums);
  }

  // Global sums
  El::AllReduce(sums, input.RowComm(), El::mpi::SUM);

  // Write to output tensor
  if (!local_output.IsEmpty()) {
    El::Gemm(El::NORMAL,
             El::TRANSPOSE,
             El::TypeTraits<TensorDataType>::One(),
             sums,
             ones,
             El::TypeTraits<TensorDataType>::Zero(),
             local_output);
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void batchwise_reduce_sum_layer<TensorDataType, Layout, Device>::bp_compute()
{

  // Data tensors
  // Note: Assume input grad and output grad are aligned.
  const auto& output_grad = this->get_prev_error_signals();
  const auto& local_output_grad = output_grad.LockedMatrix();
  auto& local_input_grad = this->get_local_error_signals();

  // Temporary buffers
  using LocalMat = El::Matrix<TensorDataType, Device>;
  LocalMat sums, ones;
  sums.Resize(local_output_grad.Height(), 1);
  El::Ones(ones, local_output_grad.Width(), 1);

  // Local sums
  if (local_output_grad.IsEmpty()) {
    El::Zero(sums);
  }
  else {
    El::Gemm(El::NORMAL,
             El::NORMAL,
             El::TypeTraits<TensorDataType>::One(),
             local_output_grad,
             ones,
             El::TypeTraits<TensorDataType>::Zero(),
             sums);
  }

  // Global sums
  El::AllReduce(sums, output_grad.RowComm(), El::mpi::SUM);

  // Write to output tensor
  if (!local_input_grad.IsEmpty()) {
    El::Gemm(El::NORMAL,
             El::TRANSPOSE,
             El::TypeTraits<TensorDataType>::One(),
             sums,
             ones,
             El::TypeTraits<TensorDataType>::Zero(),
             local_input_grad);
  }
}

#define PROTO_DEVICE(T, Device)                                                \
  template class batchwise_reduce_sum_layer<T,                                 \
                                            data_layout::DATA_PARALLEL,        \
                                            Device>;                           \
  template class batchwise_reduce_sum_layer<T,                                 \
                                            data_layout::MODEL_PARALLEL,       \
                                            Device>;
#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
