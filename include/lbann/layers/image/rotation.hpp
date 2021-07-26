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

#ifndef LBANN_LAYERS_IMAGE_ROTATION_HPP_INCLUDED
#define LBANN_LAYERS_IMAGE_ROTATION_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** brief Rotate a image around its center with a defined angle clockwise.
 *
 *  Tensors are assumed to be image data in CHW format. 
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class rotation_layer : public data_type_layer<TensorDataType> {
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "rotation_layer only supports DATA_PARALLEL");
  static_assert(Device == El::Device::CPU, 
                "rotation_layer only supports CPU");
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}


public:

  rotation_layer(lbann_comm *comm)
    : data_type_layer<TensorDataType>(comm) {
		 this->m_expected_num_parent_layers = 2;
  }

  rotation_layer* copy() const override {
    return new rotation_layer(*this);
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "rotation"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  void fp_compute() override;

protected:

  friend class cereal::access;
  rotation_layer()
    : rotation_layer(nullptr)
  {}

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);

    // Get input dimensions
    auto dims = this->get_input_dims(0);
    const auto& angle_dims = this->get_input_dims(1);
	
    const auto& num_dims = dims.size();
    const auto& num_angle_dims = angle_dims.size();

    // Check that dimensions are valid
    std::stringstream err;
    if (num_dims != 3) {
      err << get_type() << " layer \"" << this->get_name() << "\" "
          << "expects 3D input in CHW format, "
          << "but input dimensions are ";
      for (size_t i = 0; i < num_dims; ++i) {
        err << (i > 0 ? " x " : "") << dims[i];
      }
      LBANN_ERROR(err.str());
    } 
  }
};

#ifndef LBANN_ROTATION_LAYER_INSTANTIATE
#define PROTO(T) \
  extern template class rotation_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>

#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_ROTATION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_IMAGE_ROTATION_HPP_INCLUDED
