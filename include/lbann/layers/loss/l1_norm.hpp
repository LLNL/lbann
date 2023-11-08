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

#ifndef LBANN_LAYERS_LOSS_L1_NORM_HPP_INCLUDED
#define LBANN_LAYERS_LOSS_L1_NORM_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

/** @brief L1 vector norm
 *
 *  @f[ \lVert x\rVert_1 = \sum\limits_{i} | x_i | @f]
 */
template <typename TensorDataType, data_layout T_layout, El::Device Dev>
class l1_norm_layer : public data_type_layer<TensorDataType>
{
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:
  l1_norm_layer(lbann_comm* comm) : data_type_layer<TensorDataType>(comm) {}

  l1_norm_layer(const l1_norm_layer& other)
    : data_type_layer<TensorDataType>(other),
      m_workspace(other.m_workspace ? other.m_workspace->Copy() : nullptr)
  {}
  l1_norm_layer& operator=(const l1_norm_layer& other)
  {
    data_type_layer<TensorDataType>::operator=(other);
    m_workspace.reset(other.m_workspace ? other.m_workspace->Copy() : nullptr);
    return *this;
  }

  l1_norm_layer* copy() const override { return new l1_norm_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "L1 norm"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override
  {
    return ERROR_SIGNALS | PREV_ACTIVATIONS;
  }

  void setup_dims() override
  {
    data_type_layer<TensorDataType>::setup_dims();
    this->set_output_dims({1});
  }

  void setup_data(size_t max_mini_batch_size) override
  {
    data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);

    // Initialize workspace
    auto dist = this->get_prev_activations().DistData();
    dist.colDist = El::STAR;
    m_workspace.reset(AbsDistMatrixType::Instantiate(dist));
#ifdef HYDROGEN_HAVE_CUB
    if (m_workspace->GetLocalDevice() == El::Device::GPU) {
      m_workspace->Matrix().SetMemoryMode(1); // CUB memory pool
    }
#endif // HYDROGEN_HAVE_CUB
  }

  void fp_compute() override;

  void bp_compute() override;

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  l1_norm_layer() : l1_norm_layer(nullptr) {}

private:
  /** Compute local contributions to L2 norm. */
  void local_fp_compute();
  /** Compute local gradients. */
  void local_bp_compute();

  /** Workspace matrix. */
  std::unique_ptr<AbsDistMatrixType> m_workspace;
};

#ifndef LBANN_L1_NORM_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device)                                                \
  extern template class l1_norm_layer<T, data_layout::DATA_PARALLEL, Device>;  \
  extern template class l1_norm_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_L1_NORM_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LOSS_L1_NORM_HPP_INCLUDED
