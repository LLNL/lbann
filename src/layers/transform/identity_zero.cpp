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

#define LBANN_IDENTITY_ZERO_LAYER_INSTANTIATE
#include "lbann/layers/transform/identity_zero.hpp"
#include "lbann/utils/exception.hpp"

// LBANN_ASSERT_MSG_HAS_FIELD
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/proto_common.hpp"
#include "lbann/utils/protobuf.hpp"

namespace lbann {

template <typename T, data_layout L, El::Device D>
void identity_zero_layer<T, L, D>::setup_dims()
{
  data_type_layer<T>::setup_dims();
  this->set_output_dims(this->get_input_dims());

  // Check that input dimensions match
  const auto& output_dims = this->get_output_dims();
  for (int i = 0; i < this->get_num_parents(); ++i) {
    if (this->get_input_dims(i) != output_dims) {
      const auto& parents = this->get_parent_layers();
      std::ostringstream err;
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "has input tensors with incompatible dimensions (";
      for (int j = 0; j < this->get_num_parents(); ++j) {
        const auto& dims = this->get_input_dims(j);
        err << (j > 0 ? ", " : "") << "layer \"" << parents[j]->get_name()
            << "\" outputs ";
        for (size_t k = 0; k < dims.size(); ++k) {
          err << (k > 0 ? " x " : "") << dims[k];
        }
      }
      err << ")";
      LBANN_ERROR(err.str());
    }
  }
}

template <typename T, data_layout L, El::Device D>
void identity_zero_layer<T, L, D>::fp_compute()
{
  if (this->is_frozen()) {
    El::Zero(this->get_activations());
  }
  else {
    El::Copy(this->get_prev_activations(), this->get_activations());
  }
}

template <typename T, data_layout L, El::Device D>
void identity_zero_layer<T, L, D>::bp_compute()
{
  if (this->is_frozen()) {
    El::Zero(this->get_error_signals());
  }
  else {
    El::Copy(this->get_prev_error_signals(), this->get_error_signals());
  }
}

template <typename T, data_layout L, El::Device D>
void identity_zero_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_identity_zero();
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer>
build_identity_zero_layer_from_pbuf(lbann_comm* comm,
                                    lbann_data::Layer const& proto_layer)
{
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, identity_zero);
  using LayerType = identity_zero_layer<TensorDataType, Layout, Device>;

  return std::make_unique<LayerType>(comm);
}

#define PROTO_DEVICE(T, Device)                                                \
  template class identity_zero_layer<T, data_layout::DATA_PARALLEL, Device>;   \
  template class identity_zero_layer<T, data_layout::MODEL_PARALLEL, Device>;  \
  LBANN_LAYER_BUILDER_ETI(identity_zero, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
