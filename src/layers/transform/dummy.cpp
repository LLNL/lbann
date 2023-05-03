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

#define LBANN_DUMMY_LAYER_INSTANTIATE
#include "lbann/layers/transform/dummy.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"
#include "lbann/utils/tensor_impl.hpp"
#include <lbann/utils/memory.hpp>

namespace lbann {

LBANN_LAYER_DEFAULT_BUILDER(dummy)

template <typename T, data_layout L, El::Device D>
void dummy_layer<T, L, D>::set_error_signal(
  std::unique_ptr<dummy_layer<T, L, D>::AbsDistMatrixType> signal)
{
  this->m_error_signal = std::move(signal);
}

template <typename T, data_layout L, El::Device D>
void dummy_layer<T, L, D>::bp_compute()
{
  if (this->m_error_signal)
    do_tensor_copy(*this->m_error_signal, this->get_error_signals());
}

template <typename T, data_layout L, El::Device D>
void dummy_layer<T, L, D>::write_specific_proto(lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_dummy();
}

#define PROTO_DEVICE(T, Device)                                                \
  template class dummy_layer<T, data_layout::DATA_PARALLEL, Device>;           \
  template class dummy_layer<T, data_layout::MODEL_PARALLEL, Device>;          \
  LBANN_LAYER_BUILDER_ETI(dummy, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
