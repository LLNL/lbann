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

#ifndef LBANN_LAYER_CROSS_GRID_SUM_SLICE_HPP_INCLUDED
#define LBANN_LAYER_CROSS_GRID_SUM_SLICE_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

template <typename TensorDataType, El::Device Dev>
class cross_grid_sum_slice_layer : public data_type_layer<TensorDataType>
{
public:
  cross_grid_sum_slice_layer(lbann_comm* comm)
    : data_type_layer<TensorDataType>(comm)
  {
    this->m_expected_num_parent_layers = -1; // No limit on parents
    this->m_expected_num_child_layers = -1;  // No limit on children
  }

  cross_grid_sum_slice_layer* copy() const override
  {
    return new cross_grid_sum_slice_layer(*this);
  }
  std::string get_type() const override { return "cross_grid_sum_slice"; }
  data_layout get_data_layout() const override
  {
    return data_layout::DATA_PARALLEL;
  }
  El::Device get_device_allocation() const override { return Dev; }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  El::SyncInfo<Dev> syncSubGridCommunication = El::SyncInfo<Dev>();

  void setup_pointers() override
  {
    data_type_layer<TensorDataType>::setup_pointers();
    if (this->get_num_parents() < 1) {
      LBANN_ERROR(get_type(),
                  " layer \"",
                  this->get_name(),
                  "\" has no parent layers");
    }
  }

  void setup_dims(DataReaderMetaData& dr_metadata) override
  {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);

    // Slice along last dimension
    int subgridCommSize = El::mpi::Size(this->get_subgrid_comm());
    const auto input_dims = this->get_input_dims();
    std::vector<int> output_dims_slice(input_dims);
    output_dims_slice.back() = int(output_dims_slice.back() / subgridCommSize);

    for (int i = 0; i < this->get_num_children(); ++i)
      this->set_output_dims(output_dims_slice, i);
  }

  void fp_compute() override
  {
    auto const subgrid_comm_rank = El::mpi::Rank(this->get_subgrid_comm());
    auto const subgrid_comm_size = El::mpi::Size(this->get_subgrid_comm());

    auto& output = this->get_activations(subgrid_comm_rank);
    auto& input = this->get_prev_activations(subgrid_comm_rank);
    // El::Copy(input, output);

    auto* const output_cast = dynamic_cast<
      El::DistMatrix<TensorDataType, El::STAR, El::VC, El::ELEMENT, Dev>*>(
      &output);

    auto const sync_info_output =
      El::SyncInfoFromMatrix(output_cast->LockedMatrix());

    const El::Int mloc = input.LocalHeight();
    const El::Int nloc = input.LocalWidth();

    El::Matrix<TensorDataType, Dev> prev_allreduce(mloc, nloc),
      after_allreduce(mloc, nloc);

    El::Copy(input.LockedMatrix(), prev_allreduce);

    El::mpi::AllReduce(prev_allreduce.Buffer(),
                       after_allreduce.Buffer(),
                       mloc * nloc,
                       El::mpi::SUM,
                       this->get_subgrid_comm(),
                       sync_info_output);

    const auto input_dims = this->get_input_dims();
    int last_dim = input_dims.back();

    int last_dim_start_point = 1;
    for (int i = 0; i < int(input_dims.size()) - 1; ++i) {
      last_dim_start_point = last_dim_start_point * input_dims[i];
    }

    if (last_dim % subgrid_comm_size != 0)
      LBANN_ERROR("cross_grid_sum_slice layer: last dimension should be "
                  "divided by the number of branches in subgraph");

    int const last_dim_index =
      int(last_dim / subgrid_comm_size) * subgrid_comm_rank;

    El::copy::util::InterleaveMatrix(
      (last_dim / subgrid_comm_size),
      input.LocalWidth() * last_dim_start_point,
      after_allreduce.LockedBuffer(last_dim_index, 0),
      1,
      last_dim,
      output_cast->Buffer(),
      1,
      (last_dim / subgrid_comm_size),
      sync_info_output);
  }

  void fp_setup_outputs(El::Int mini_batch_size) override
  {

    if (this->get_num_children() < 1) {
      return;
    }
    // Determine distributed matrix alignment

    // Initialize output tensors
    for (int i = 0; i < this->get_num_children(); ++i) {
      auto& output = this->get_activations(i);
      output.Empty(false);
      output.Resize(this->get_output_size(i), mini_batch_size);
    }
  }

  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override
  {
    auto childs = this->get_child_layers();
    auto const subgrid_comm_rank = El::mpi::Rank(this->get_subgrid_comm());
    auto const subgrid_comm_size = El::mpi::Size(this->get_subgrid_comm());
    const auto input_dims = this->get_input_dims();
    int last_dim = input_dims.back();

    auto& input_grad = this->get_error_signals(subgrid_comm_rank);
    const auto& gradient_wrt_output =
      this->get_prev_error_signals(subgrid_comm_rank);

    using MatrixType =
      El::DistMatrix<TensorDataType, El::STAR, El::VC, El::ELEMENT, Dev>;
    auto const* const gradient_wrt_output_cast =
      dynamic_cast<const MatrixType*>(&gradient_wrt_output);

    auto* const gradient_wrt_input_cast =
      dynamic_cast<MatrixType*>(&input_grad);

    int mloc = gradient_wrt_output_cast->LocalHeight();
    int nloc = gradient_wrt_output_cast->LocalWidth();
    int per_grid_last_dim = last_dim / subgrid_comm_size;

    El::Matrix<TensorDataType, Dev> temp_input(nloc, mloc),
      temp_output(nloc * (mloc / per_grid_last_dim), last_dim),
      transposed(nloc * (mloc / per_grid_last_dim), per_grid_last_dim),
      transposed_output(last_dim, nloc * (mloc / per_grid_last_dim));

    El::Copy(gradient_wrt_output_cast->LockedMatrix(), temp_input);
    temp_input.Resize(per_grid_last_dim, nloc * (mloc / per_grid_last_dim));

    El::Transpose(temp_input, transposed);

    El::mpi::AllGather(transposed.Buffer(),
                       mloc * nloc,
                       temp_output.Buffer(),
                       mloc * nloc,
                       this->get_subgrid_comm(),
                       syncSubGridCommunication);

    El::Transpose(temp_output, transposed_output);
    transposed_output.Resize(mloc * subgrid_comm_size, nloc);

    // gradient_wrt_input_cast->Resize(this->get_input_size(), mini_batch_size);
    for(int i=0; i<childs.size();i++)
    {
      auto* const gradient_wrt_input_cast_layer = dynamic_cast<
        El::DistMatrix<TensorDataType, El::STAR, El::VC, El::ELEMENT, Dev>*>(
        &this->get_error_signals(i));
      gradient_wrt_input_cast_layer->Resize(this->get_input_size(), mini_batch_size);
    }
    El::Copy(transposed_output, gradient_wrt_input_cast->Matrix());
  }

  void bp_compute() final {}
};

#ifndef LBANN_CROSS_GRID_SUM_SLICE_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class cross_grid_sum_slice_layer<T, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_CROSS_GRID_SUM_SLICE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_CROSS_GRID_SUM_SLICE_HPP_INCLUDED
