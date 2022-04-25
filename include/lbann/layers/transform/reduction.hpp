////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYER_REDUCTION_HPP_INCLUDED
#define LBANN_LAYER_REDUCTION_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

enum class reduction_mode {INVALID, SUM, AVERAGE};

/** @brief Reduce tensor to scalar
 *
 *  @todo Reduction over specified dimensions.
 */
template <typename TensorDataType,
          data_layout Layout = data_layout::DATA_PARALLEL,
          El::Device Device = El::Device::CPU>
class reduction_layer : public data_type_layer<TensorDataType> {
private:

  /** Reduction mode. */
  reduction_mode m_mode;

public:

  reduction_layer(reduction_mode mode=reduction_mode::SUM)
    : data_type_layer<TensorDataType>(nullptr),
      m_mode(mode) {
    if (mode == reduction_mode::INVALID) {
      LBANN_ERROR("invalid reduction mode");
    }
  }

  reduction_layer* copy() const override { return new reduction_layer(*this); }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "reduction"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  description get_description() const override {
    auto desc = data_type_layer<TensorDataType>::get_description();
    std::string mode_str;
    switch (m_mode) {
    case reduction_mode::SUM:     mode_str = "sum";     break;
    case reduction_mode::AVERAGE: mode_str = "average"; break;
    case reduction_mode::INVALID:
    default:
      mode_str = "invalid";
    }
    desc.add("Mode", mode_str);
    return desc;
  }

protected:

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    this->set_output_dims({1});
  }

  void fp_compute() override {

    // Constants
    const auto one = El::TypeTraits<TensorDataType>::One();
    const auto zero = El::TypeTraits<TensorDataType>::Zero();

    // Data matrices
    using LocalMat = El::Matrix<TensorDataType, Device>;
    const auto& input = this->get_prev_activations();
    auto& output = this->get_activations();

    // Create workspace buffers
    LocalMat local_reduction, ones;
    const auto& col_comm = input.ColComm();
    const auto col_rank = El::mpi::Rank(col_comm);
    const auto owner_rank = output.RowOwner(0);
    if (col_rank == owner_rank) {
      El::View(local_reduction, output.Matrix());
    }
    else {
      local_reduction.Resize(1, input.LocalWidth());
    }
    El::Ones(ones, input.LocalHeight(), 1);

    // Compute local reductions
    switch (m_mode) {
    case reduction_mode::SUM:
      El::Gemv(
        El::TRANSPOSE,
        one,
        input.LockedMatrix(),
        ones,
        zero,
        local_reduction);
      break;
    case reduction_mode::AVERAGE:
      El::Gemv(
        El::TRANSPOSE,
        one / El::To<TensorDataType>(input.Height()),
        input.LockedMatrix(),
        ones,
        zero,
        local_reduction);
      break;
    default:
      LBANN_ERROR("invalid reduction mode");
    }

    // Accumulate local reductions in output matrix
    /// @todo Replace with Reduce when supported in Hydrogen.
    El::AllReduce(local_reduction, col_comm, El::mpi::SUM);

  }

  void bp_compute() override {

    // Constants
    const auto one = El::TypeTraits<TensorDataType>::One();
    const auto zero = El::TypeTraits<TensorDataType>::Zero();

    // Data matrices
    using LocalMat = El::Matrix<TensorDataType, Device>;
    const auto& output_grad = this->get_prev_error_signals();
    auto& input_grad = this->get_error_signals();

    // Create workspace buffers
    LocalMat local_output_grad, ones;
    const auto& col_comm = input_grad.ColComm();
    const auto col_rank = El::mpi::Rank(col_comm);
    const auto owner_rank = output_grad.RowOwner(0);
    if (col_rank == owner_rank) {
      El::LockedView(local_output_grad, output_grad.LockedMatrix());
    }
    else {
      local_output_grad.Resize(1, input_grad.LocalWidth());
    }
    /** @todo (tym1 3/12/21): We are working around a bug in Hydrogen.
     *  Broadcast with Matrix<T,D> is not instatiated. */
    El::Broadcast(
      static_cast<El::AbstractMatrix<TensorDataType>&>(local_output_grad),
      col_comm,
      owner_rank);
    El::Ones(ones, input_grad.LocalHeight(), 1);

    // Populate error signals
    switch (m_mode) {
    case reduction_mode::SUM:
      El::Gemm(
        El::NORMAL,
        El::NORMAL,
        one,
        ones,
        local_output_grad,
        zero,
        input_grad.Matrix());
      break;
    case reduction_mode::AVERAGE:
      El::Gemm(
        El::NORMAL,
        El::NORMAL,
        one / El::To<TensorDataType>(input_grad.Height()),
        ones,
        local_output_grad,
        zero,
        input_grad.Matrix());
      break;
    default:
      LBANN_ERROR("invalid reduction mode");
    }

  }

};


#ifndef LBANN_REDUCTION_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                 \
  extern template class reduction_layer<        \
    T, data_layout::DATA_PARALLEL, Device>;     \
  extern template class reduction_layer<        \
    T, data_layout::MODEL_PARALLEL, Device>
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_REDUCTION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_REDUCTION_HPP_INCLUDED
