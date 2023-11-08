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

#ifndef LBANN_LAYERS_MISC_VARIANCE_IMPL_HPP_INCLUDED
#define LBANN_LAYERS_MISC_VARIANCE_IMPL_HPP_INCLUDED

#include "lbann/layers/misc/variance.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void variance_layer<TensorDataType, Layout, Device>::setup_data(
  size_t max_mini_batch_size)
{
  data_type_layer<TensorDataType>::setup_data(max_mini_batch_size);
  auto dist_data = this->get_prev_activations().DistData();
  dist_data.colDist = El::STAR;
  m_means.reset(AbsDistMatrixType::Instantiate(dist_data));
  m_workspace.reset(AbsDistMatrixType::Instantiate(dist_data));
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void variance_layer<TensorDataType, Layout, Device>::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();
  this->set_output_dims({1});
  if (this->get_input_size() <= 1) {
    std::stringstream err;
    const auto& parents = this->get_parent_layers();
    const auto& dims = this->get_input_dims();
    err << get_type() << " layer \"" << this->get_name() << "\" "
        << "expects an input tensor with at least two entries, "
        << "but parent layer \"" << parents[0]->get_name() << "\" "
        << "outputs a tensor with dimensions ";
    for (size_t i = 0; i < dims.size(); ++i) {
      err << (i > 0 ? " x " : "") << dims[i];
    }
    LBANN_ERROR(err.str());
  }
}

} // namespace lbann

#endif // LBANN_LAYERS_MISC_VARIANCE_IMPL_HPP_INCLUDED
