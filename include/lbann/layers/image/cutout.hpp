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

#ifndef LBANN_LAYERS_IMAGE_CUTOUT_HPP_INCLUDED
#define LBANN_LAYERS_IMAGE_CUTOUT_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"

namespace lbann {

/** @brief Cutout a square from an image
 *
 *  Expects two inputs: a 3D image tensor in CHW format and a scalar
 *  length of the cutout square.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class cutout_layer : public data_type_layer<TensorDataType> {
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "cutout_layer only supports DATA_PARALLEL");
  static_assert(Device == El::Device::CPU,
                "cutou_layer only supports CPU");
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}


public:

  cutout_layer(lbann_comm *comm)
    : data_type_layer<TensorDataType>(comm) {
		 this->m_expected_num_parent_layers = 2;
  }

  cutout_layer* copy() const override {
    return new cutout_layer(*this);
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "cutout"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  void fp_compute() override;

protected:

  friend class cereal::access;
  cutout_layer()
    : cutout_layer(nullptr)
  {}

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    data_type_layer<TensorDataType>::setup_dims(dr_metadata);

    // Get input dimensions
    auto dims = this->get_input_dims(0);
    const auto& cutout_length = this->get_input_dims(1);

    // Check that dimensions are valid
    if (dims.size() != 3) {
      std::ostringstream ss;
      for (size_t i = 0; i < dims.size(); ++i) {
        ss << (i > 0 ? " x " : "") << dims[i];
      }
      LBANN_ERROR(this->get_type()," layer \"",this->get_name(),"\" ",
        "expects a 3D input in CHW format, ",
        "but input dimensions are ",ss.str());
    }
    if (cutout_length.size() > 1 || cutout_length[0] != 1) {
      std::ostringstream ss;
      for (size_t i = 0; i < cutout_length.size(); ++i) {
        ss << (i > 0 ? " x " : "") << cutout_length[i];
      }
      LBANN_ERROR(
        this->get_type()," layer \"",this->get_name(),"\" ",
        "expects a scalar input for the cutout length, ",
        "but input dimensions are ",ss.str());
    }
  }
};

#ifndef LBANN_CUTOUT_LAYER_INSTANTIATE
#define PROTO(T) \
  extern template class cutout_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>

#include "lbann/macros/instantiate.hpp"
#undef PROTO
#endif // LBANN_CUTOUT_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_IMAGE_CUTOUT_HPP_INCLUDED
