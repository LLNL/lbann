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

#ifndef LBANN_LAYER_SORT_HPP_INCLUDED
#define LBANN_LAYER_SORT_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

/** @brief Sort tensor entries */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class sort_layer : public data_type_layer<TensorDataType>
{
  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "sort layer only supports DATA_PARALLEL");

public:
  sort_layer(lbann_comm* comm, bool descending = false)
    : data_type_layer<TensorDataType>(comm), m_descending(descending)
  {}
  sort_layer(const sort_layer& other);

  sort_layer& operator=(const sort_layer& other);

  sort_layer* copy() const override { return new sort_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "sort"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

  description get_description() const override
  {
    auto desc = data_type_layer<TensorDataType>::get_description();
    desc.add("Descending", m_descending);
    return desc;
  }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  sort_layer() : sort_layer(nullptr) {}

  void setup_dims() override;

  void setup_data(size_t max_mini_batch_size) override;

  void fp_setup_outputs(El::Int mini_batch_size) override;

  void fp_compute() override;
  void bp_compute() override;

private:
  /** Whether values are sorted by descending order. */
  bool m_descending;

  /** Input indices corresponding to output entries.
   *  @todo Switch to distributed integer matrix once it's supported
   *  in Hydrogen.
   */
  std::unique_ptr<El::AbstractMatrix<El::Int>> m_indices;
};

template <typename T, data_layout L, El::Device D>
void sort_layer<T, L, D>::write_specific_proto(lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_sort();
  msg->set_descending(m_descending);
}

#ifndef LBANN_SORT_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class sort_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_SORT_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_SORT_HPP_INCLUDED
