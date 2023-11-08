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

#define LBANN_MINI_BATCH_SIZE_LAYER_INSTANTIATE
#include "lbann/layers/misc/mini_batch_size.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

template <typename T, data_layout L, El::Device D>
mini_batch_size_layer<T, L, D>::mini_batch_size_layer(lbann_comm* comm)
  : data_type_layer<T>(comm)
{
  this->m_expected_num_parent_layers = 0;
}

template <typename T, data_layout L, El::Device D>
auto mini_batch_size_layer<T, L, D>::copy() const -> mini_batch_size_layer*
{
  return new mini_batch_size_layer(*this);
}

template <typename T, data_layout L, El::Device D>
std::string mini_batch_size_layer<T, L, D>::get_type() const
{
  return "mini-batch size";
}

template <typename T, data_layout L, El::Device D>
data_layout mini_batch_size_layer<T, L, D>::get_data_layout() const
{
  return L;
}

template <typename T, data_layout L, El::Device D>
El::Device mini_batch_size_layer<T, L, D>::get_device_allocation() const
{
  return D;
}

template <typename T, data_layout L, El::Device D>
void mini_batch_size_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_mini_batch_size();
}

template <typename T, data_layout L, El::Device D>
void mini_batch_size_layer<T, L, D>::setup_dims()
{
  data_type_layer<T>::setup_dims();
  this->set_output_dims({1});
}

template <typename T, data_layout L, El::Device D>
void mini_batch_size_layer<T, L, D>::setup_data(size_t max_mini_batch_size)
{
  data_type_layer<T>::setup_data(max_mini_batch_size);
  m_mini_batch_size = max_mini_batch_size;
}

template <typename T, data_layout L, El::Device D>
void mini_batch_size_layer<T, L, D>::fp_compute()
{
  El::Fill(this->get_activations(), El::To<T>(m_mini_batch_size));
}

#define PROTO_DEVICE(T, Device)                                                \
  template class mini_batch_size_layer<T, data_layout::DATA_PARALLEL, Device>; \
  template class mini_batch_size_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
