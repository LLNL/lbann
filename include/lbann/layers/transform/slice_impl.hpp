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

#ifndef LBANN_LAYER_SLICE_IMPL_HPP_INCLUDED
#define LBANN_LAYER_SLICE_IMPL_HPP_INCLUDED

#include "lbann/data_ingestion/data_coordinator.hpp"
#include "lbann/layers/transform/slice.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void slice_layer<TensorDataType, Layout, Device>::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();

  // Setup the slice points if they are to be established by the data reader
  // TODO: Move this responsibility to another component (input layer)
  if (m_set_slice_points_from_data_reader) {
    std::vector<size_t> slice_points;
    std::string slice_point_method_name = "'get_slice_points_from_reader'";

    LBANN_WARNING("slice_points_from_reader is deprecated and will be removed "
                  "in a future version.");

    const data_coordinator& dc = get_const_trainer().get_data_coordinator();
    const DataReaderMetaData& dr_metadata = dc.get_dr_metadata();
    for (auto& slice_point : dr_metadata.slice_points.at(m_var_category)) {
      slice_points.push_back(slice_point);
    }

    if (slice_points.size() < 2u) {
      LBANN_ERROR(slice_point_method_name, " is not supported by the reader.");
      return;
    }
    m_slice_points = std::move(slice_points);
  }

  // Check that slice parameters are valid
  const auto& input_dims = this->get_input_dims();
  const size_t num_outputs = this->get_num_children();
  if (m_slice_dim >= input_dims.size()) {
    std::ostringstream err;
    err << this->get_type() << " layer \"" << this->get_name() << "\" "
        << "is slicing along dimension " << m_slice_dim << ", "
        << "but it has a " << input_dims.size() << "-D input tensor "
        << "(parent layer \"" << this->get_parent_layers()[0]->get_name()
        << "\" "
        << "outputs with dimensions ";
    for (size_t d = 0; d < input_dims.size(); ++d) {
      err << (d > 0 ? " x " : "") << input_dims[d];
    }
    err << ")";
    LBANN_ERROR(err.str());
  }
  if (m_slice_points.size() <= num_outputs) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has ",
                num_outputs,
                " children, "
                "but only ",
                m_slice_points.size(),
                " slice points");
  }
  if (!std::is_sorted(m_slice_points.begin(), m_slice_points.end())) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has unsorted slice points");
  }
  if (m_slice_points.back() > static_cast<size_t>(input_dims[m_slice_dim])) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "has a slice point of ",
                m_slice_points.back(),
                ", ",
                "which is outside the expected range "
                "[0 ",
                input_dims[m_slice_dim],
                "]");
  }

  // Model-parallel implementation only supports flat data
  if (Layout == data_layout::MODEL_PARALLEL && input_dims.size() != 1) {
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "attempted to slice along dimension ",
                m_slice_dim,
                ", ",
                "but model-parallel slice layer only supports flat data");
  }

  // Set output tensor dimensions
  auto output_dims = input_dims;
  for (size_t i = 0; i < num_outputs; ++i) {
    output_dims[m_slice_dim] = m_slice_points[i + 1] - m_slice_points[i];
    this->set_output_dims(output_dims, i);
  }
}

} // namespace lbann

#endif // LBANN_LAYER_SLICE_IMPL_HPP_INCLUDED
