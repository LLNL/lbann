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

#ifndef LBANN_LAYERS_MISC_ONE_HOT_IMPL_HPP_INCLUDED
#define LBANN_LAYERS_MISC_ONE_HOT_IMPL_HPP_INCLUDED

#include "lbann/layers/misc/one_hot.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void one_hot_layer<TensorDataType, Layout, Device>::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();

  // Make sure input tensor is scalar
  if (this->get_input_size() != 1) {
    const auto input_dims = this->get_input_dims();
    std::ostringstream dim_ss;
    for (size_t i = 0; i < input_dims.size(); ++i) {
      dim_ss << (i > 0 ? "x" : "") << input_dims[i];
    }
    LBANN_ERROR(get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "received an input tensor with invalid dimensions ",
                "(expected 1, got ",
                dim_ss.str(),
                ")");
  }
}

} // namespace lbann

#endif // LBANN_LAYERS_MISC_ONE_HOT_IMPL_HPP_INCLUDED
