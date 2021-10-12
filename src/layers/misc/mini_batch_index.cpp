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

#define LBANN_MINI_BATCH_INDEX_LAYER_INSTANTIATE
#include "lbann/layers/misc/mini_batch_index.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

template <typename T, data_layout L, El::Device D>
mini_batch_index_layer<T, L, D>::mini_batch_index_layer(lbann_comm* comm)
  : data_type_layer<T>(comm)
{
  this->m_expected_num_parent_layers = 0;
}

template <typename T, data_layout L, El::Device D>
auto mini_batch_index_layer<T, L, D>::copy() const -> mini_batch_index_layer*
{
  return new mini_batch_index_layer(*this);
}

template <typename T, data_layout L, El::Device D>
std::string mini_batch_index_layer<T, L, D>::get_type() const
{
  return "mini-batch index";
}

template <typename T, data_layout L, El::Device D>
data_layout mini_batch_index_layer<T, L, D>::get_data_layout() const
{
  return L;
}

template <typename T, data_layout L, El::Device D>
El::Device mini_batch_index_layer<T, L, D>::get_device_allocation() const
{
  return D;
}

template <typename T, data_layout L, El::Device D>
void mini_batch_index_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_mini_batch_index();
}

template <typename T, data_layout L, El::Device D>
void mini_batch_index_layer<T, L, D>::setup_dims(
  DataReaderMetaData& dr_metadata)
{
  data_type_layer<T>::setup_dims(dr_metadata);
  this->set_output_dims({1});
}

template <typename T, data_layout L, El::Device D>
void mini_batch_index_layer<T, L, D>::fp_compute()
{
  using CPUMatType = El::Matrix<T, El::Device::CPU>;

  // Get output matrix
  auto& output = this->get_activations();
  auto& local_output = output.Matrix();
  const auto& local_width = local_output.Width();

  // Create temporary matrix if output matrix is not on CPU
  CPUMatType local_output_v;
  if (local_output.GetDevice() == El::Device::CPU) {
    El::View(local_output_v, local_output);
  }
  else {
    local_output_v.Resize(1, local_width);
  }
#ifdef LBANN_HAS_CALIPER
    CALI_MARK_BEGIN("populate_matrix");
#endif
  // Populate matrix on CPU
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    local_output_v(0, col) = El::To<T>(output.GlobalCol(col));
  }
#ifdef LBANN_HAS_CALIPER
    CALI_MARK_END("populate_matrix");
#endif
  // Copy result from CPU if needed
  if (!local_output_v.Viewing()) {
    El::Copy(local_output_v, local_output);
  }
}

#define PROTO_DEVICE(T, Device)                                                \
  template class mini_batch_index_layer<T,                                     \
                                        data_layout::DATA_PARALLEL,            \
                                        Device>;                               \
  template class mini_batch_index_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
