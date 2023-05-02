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

#ifndef LBANN_LAYER_CROSS_GRID_SUM_HPP_INCLUDED
#define LBANN_LAYER_CROSS_GRID_SUM_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

template <typename TensorDataType, El::Device Dev>
class cross_grid_sum_layer final : public data_type_layer<TensorDataType>
{
public:
  cross_grid_sum_layer(lbann_comm* comm) : data_type_layer<TensorDataType>(comm)
  {
    this->m_expected_num_parent_layers = -1; // No limit on parents
    this->m_expected_num_child_layers = -1;  // No limit on children
  }

  cross_grid_sum_layer* copy() const final
  {
    return new cross_grid_sum_layer(*this);
  }
  std::string get_type() const final { return "cross_grid_sum"; }
  data_layout get_data_layout() const final
  {
    return data_layout::DATA_PARALLEL;
  }
  El::Device get_device_allocation() const final { return Dev; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

private:
  void setup_pointers() final
  {
    data_type_layer<TensorDataType>::setup_pointers();
    if (this->get_num_parents() < 1) {
      LBANN_ERROR(get_type(),
                  " layer \"",
                  this->get_name(),
                  "\" has no parent layers");
    }
  }

  void setup_dims(DataReaderMetaData& dr_metadata) final
  {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);
    this->set_output_dims(this->get_input_dims());

    // print dims
#ifdef LBANN_DEBUG
    {
      const auto& dims_print = this->get_input_dims();
      auto const dims_size = dims_print.size();
      for (auto ii = 0UL; ii < dims_size; ++ii) {
        std::cout << "Index:" << ii << " dim" << dims_print[ii] << "\n";
      }
    }
#endif // LBANN_DEBUG

    // Check that input dimensions match
    const auto& output_dims = this->get_output_dims();
    for (int i = 0; i < this->get_num_parents(); ++i) {
      if (this->get_input_dims(i) != output_dims) {
        const auto& parents = this->get_parent_layers();
        std::stringstream err;
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

  void fp_compute() final
  {
    int const rank = El::mpi::Rank(this->get_subgrid_comm());

    auto& output = this->get_activations(rank);
    auto& input = this->get_prev_activations(rank);
    El::Copy(input, output);

    auto* const output_cast = dynamic_cast<
      El::DistMatrix<TensorDataType, El::STAR, El::VC, El::ELEMENT, Dev>*>(
      &output);

    auto const syncInfoOutput =
      El::SyncInfoFromMatrix(output_cast->LockedMatrix());

    const El::Int mloc = output_cast->LocalHeight();
    const El::Int nloc = output_cast->LocalWidth();

    El::Matrix<TensorDataType, Dev> temp_output(mloc, nloc);

    El::Copy(output_cast->LockedMatrix(), temp_output);

    El::mpi::AllReduce(temp_output.Buffer(),
                       output_cast->Buffer(),
                       mloc * nloc,
                       El::mpi::SUM,
                       this->get_subgrid_comm(),
                       syncInfoOutput);
  }

  void fp_setup_outputs(El::Int mini_batch_size) final
  {

    if (this->get_num_children() < 1) {
      return;
    }

    // Initialize output tensors
    for (int i = 0; i < this->get_num_children(); ++i) {

      auto& output = this->get_activations(i);
      output.Empty(false);
      output.Resize(this->get_output_size(i), mini_batch_size);
    }
  }

  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) final
  {
    int rank = El::mpi::Rank(this->get_subgrid_comm());
    const auto& gradient_wrt_output = this->get_prev_error_signals(rank);
    auto& gradient_wrt_input = this->get_error_signals(rank);
    El::Copy(gradient_wrt_output, gradient_wrt_input);

    auto* const gradient_wrt_input_cast = dynamic_cast<
      El::DistMatrix<TensorDataType, El::STAR, El::VC, El::ELEMENT, Dev>*>(
      &gradient_wrt_input);

    const El::Int mloc = gradient_wrt_input_cast->LocalHeight();
    const El::Int nloc = gradient_wrt_input_cast->LocalWidth();

    El::Matrix<TensorDataType, Dev> temp_output(mloc, nloc);

    El::Copy(gradient_wrt_input_cast->LockedMatrix(), temp_output);

    El::AllReduce(gradient_wrt_input, this->get_subgrid_comm(), El::mpi::SUM);
  }

  void bp_compute() final {}
};

#ifndef LBANN_CROSS_GRID_SUM_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class cross_grid_sum_layer<T, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE

#endif // LBANN_CROSS_GRID_SUM_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_CROSS_GRID_SUM_HPP_INCLUDED
