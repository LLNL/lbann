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

#ifndef LBANN_LAYERS_IMAGE_BILINEAR_RESIZE_IMPL_HPP_INCLUDED
#define LBANN_LAYERS_IMAGE_BILINEAR_RESIZE_IMPL_HPP_INCLUDED

#include "lbann/layers/image/bilinear_resize.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void bilinear_resize_layer<TensorDataType, Layout, Device>::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();

  // Get input dimensions
  auto dims = this->get_input_dims();
  const auto& num_dims = dims.size();

  // Check that dimensions are valid
  std::stringstream err;
  if (num_dims < 2) {
    err << get_type() << " layer \"" << this->get_name() << "\" "
        << "expects input with at least two dimensions, "
        << "but input dimensions are ";
    for (size_t i = 0; i < num_dims; ++i) {
      err << (i > 0 ? " x " : "") << dims[i];
    }
    LBANN_ERROR(err.str());
  }
  else if (m_height <= 0) {
    err << get_type() << " layer \"" << this->get_name() << "\" "
        << "attempted to resize with "
        << "negative height (" << m_height << ")";
    LBANN_ERROR(err.str());
  }
  else if (m_width <= 0) {
    err << get_type() << " layer \"" << this->get_name() << "\" "
        << "attempted to resize with "
        << "negative width (" << m_width << ")";
    LBANN_ERROR(err.str());
  }

  // Resize output tensor
  dims[num_dims - 2] = m_height;
  dims[num_dims - 1] = m_width;
  this->set_output_dims(dims);
}

} // namespace lbann

#endif // LBANN_LAYERS_IMAGE_BILINEAR_RESIZE_IMPL_HPP_INCLUDED
