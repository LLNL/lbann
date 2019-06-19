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

#ifndef LBANN_LAYERS_IMAGE_BILINEAR_RESIZE_HPP_INCLUDED
#define LBANN_LAYERS_IMAGE_BILINEAR_RESIZE_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

/** @brief Resize image with bilinear interpolation.
 *
 *  Tensors are assumed to be image data in CHW format. Gradients are
 *  not propagated during backprop.
 */
template <data_layout Layout, El::Device Device>
class bilinear_resize_layer : public Layer {
public:

  bilinear_resize_layer(lbann_comm *comm, El::Int height, El::Int width)
    : Layer(comm), m_height(height), m_width(width) {
    static_assert(Layout == data_layout::DATA_PARALLEL,
                  "bilinear_resize_layer only supports DATA_PARALLEL");
  }

  bilinear_resize_layer* copy() const override {
    return new bilinear_resize_layer(*this);
  }
  std::string get_type() const override { return "bilinear resize"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  void fp_compute() override;

protected:

  void setup_dims() override {
    Layer::setup_dims();

    // Get input dimensions
    auto dims = get_input_dims();
    const auto& num_dims = dims.size();

    // Check that dimensions are valid
    std::stringstream err;
    if (num_dims < 2) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "expects input with at least two dimensions, "
          << "but input dimensions are ";
      for (size_t i = 0; i < num_dims; ++i) {
        err << (i > 0 ? " x " : "") << dims[i];
      }
      LBANN_ERROR(err.str());
    } else if (m_height <= 0) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "attempted to resize with "
          << "negative height (" << m_height << ")";
      LBANN_ERROR(err.str());
    } else if (m_width <= 0) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "attempted to resize with "
          << "negative width (" << m_width << ")";
      LBANN_ERROR(err.str());
    }

    // Resize output tensor
    dims[num_dims-2] = m_height;
    dims[num_dims-1] = m_width;
    set_output_dims(dims);

  }

private:

  /** Output image height.
   *  Data is assumed to be in CHW format.
   */
  El::Int m_height;
  /** Output image width.
   *  Data is assumed to be in CHW format.
   */
  El::Int m_width;

};

} // namespace lbann

#endif // LBANN_LAYERS_IMAGE_BILINEAR_RESIZE_HPP_INCLUDED
