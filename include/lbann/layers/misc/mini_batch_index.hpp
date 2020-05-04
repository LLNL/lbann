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

#ifndef LBANN_LAYERS_MISC_MINI_BATCH_INDEX_HPP_INCLUDED
#define LBANN_LAYERS_MISC_MINI_BATCH_INDEX_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Mini-batch index.
 *
 *  Output tensor is a 1D tensor with a single entry containing the
 *  mini-batch sample. Each sample in a model's mini-batch has a
 *  unique index in [0, mini_batch_size).
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class mini_batch_index_layer : public data_type_layer<TensorDataType> {
public:

  mini_batch_index_layer(lbann_comm* comm) : data_type_layer<TensorDataType>(comm) {
    this->m_expected_num_parent_layers = 0;
  }

  mini_batch_index_layer* copy() const override { return new mini_batch_index_layer(*this); }
  std::string get_type() const override { return "mini-batch index"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

protected:

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    this->set_output_dims({1});
  }

  void fp_compute() override {
    using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

    // Get output matrix
    auto& output = this->get_activations();
    auto& local_output = output.Matrix();
    const auto& local_width = local_output.Width();

    // Create temporary matrix if output matrix is not on CPU
    CPUMatType local_output_v;
    if (local_output.GetDevice() == El::Device::CPU) {
      El::View(local_output_v, local_output);
    } else {
      local_output_v.Resize(1, local_width);
    }

    // Populate matrix on CPU
    LBANN_OMP_PARALLEL_FOR
    for (El::Int col = 0; col < local_width; ++col) {
      local_output_v(0, col) = El::To<TensorDataType>(output.GlobalCol(col));
    }

    // Copy result from CPU if needed
    if (!local_output_v.Viewing()) {
      El::Copy(local_output_v, local_output);
    }

  }

};

#ifndef LBANN_MINI_BATCH_INDEX_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device) \
  extern template class mini_batch_index_layer<T, data_layout::DATA_PARALLEL, Device>; \
  extern template class mini_batch_index_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_MINI_BATCH_INDEX_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_MISC_MINI_BATCH_INDEX_HPP_INCLUDED
