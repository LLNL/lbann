////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYER_SLICE_HPP_INCLUDED
#define LBANN_LAYER_SLICE_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** Slice layer.
 *  This layer slices an input tensor along a specified axis.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class slice_layer : public transform_layer {
 private:

  /** Tensor dimension to slice. */
  int m_slice_axis;
  /** Slice points for each child layer. */
  std::vector<int> m_slice_points;

  /** View into region of input tensor. */
  AbsDistMat *m_input_region_v;
  /** View into region of output tensor. */
  AbsDistMat *m_output_region_v;

 public:

  slice_layer(lbann_comm *comm,
              int slice_axis,
              std::vector<int> slice_points)
    : transform_layer(comm),
      m_slice_axis(slice_axis),
      m_slice_points(slice_points),
      m_input_region_v(nullptr),
      m_output_region_v(nullptr) {

    // Slice layer has no limit on children
    m_expected_num_child_layers = -1;

  }

  slice_layer(const slice_layer& other)
    : transform_layer(other),
      m_slice_axis(other.m_slice_axis),
      m_slice_points(other.m_slice_points),
      m_input_region_v(other.m_input_region_v),
      m_output_region_v(other.m_output_region_v) {
    // Deep copy matrices
    if (m_input_region_v != nullptr) {
      m_input_region_v = m_input_region_v->Copy();
    }
    if (m_output_region_v != nullptr) {
      m_output_region_v = m_output_region_v->Copy();
    }
  }

  slice_layer& operator=(const slice_layer& other) {
    transform_layer::operator=(other);
    m_slice_axis = other.m_slice_axis;
    m_slice_points = other.m_slice_points;

    // Deep copy matrices
    if (m_input_region_v != nullptr)  { delete m_input_region_v; }
    if (m_output_region_v != nullptr) { delete m_output_region_v; }
    m_input_region_v = other.m_input_region_v;
    m_output_region_v = other.m_output_region_v;
    if (m_input_region_v != nullptr) {
      m_input_region_v = m_input_region_v->Copy();
    }
    if (m_output_region_v != nullptr) {
      m_output_region_v = m_output_region_v->Copy();
    }

  }

  virtual ~slice_layer() override {
    if (m_input_region_v != nullptr)  { delete m_input_region_v; }
    if (m_output_region_v != nullptr) { delete m_output_region_v; }
  }

  slice_layer* copy() const override { return new slice_layer(*this); }
  std::string get_type() const override { return "slice"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
    s << " slice; slice_axis: "
      << m_slice_axis << " children: ";
    for (size_t h=0; h<this->m_child_layers.size(); h++) {
      s << this->m_child_layers[h]->get_name() << " " << this->m_child_layers[h]->get_type() << " ";
    }
    s << " slice_points: ";
    for (size_t h=0; h<this->m_slice_points.size(); h++) {
      s << this->m_slice_points[h] << " ";
    }
    s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    s << " device alloc: " << this->get_device_allocation_string(get_device_allocation());
    return s.str();
  }

  void setup_pointers() override {
    transform_layer::setup_pointers();
    if (get_num_children() <= 0) {
      LBANN_ERROR("slice layer has no children");
    }
  }

  void setup_matrices(const El::Grid& grid) override {
    transform_layer::setup_matrices(grid);
    if (m_input_region_v != nullptr)  { delete m_input_region_v; }
    if (m_output_region_v != nullptr) { delete m_output_region_v; }
    const auto& input = get_prev_activations();
    m_input_region_v = input.Construct(input.Grid(), input.Root());
    m_output_region_v = input.Construct(input.Grid(), input.Root());
  }

  void setup_dims() override {
    transform_layer::setup_dims();
    std::stringstream err;

    const auto& input_dims = get_input_dims();
    const auto& num_outputs = get_num_children();

    // Check that slice parameters are valid
    if (m_slice_axis >= (int) input_dims.size()) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "cannot slice along axis " << m_slice_axis << " "
          << "since it only has " << input_dims.size() << " dimensions";
      LBANN_ERROR(err.str());
    }
    if ((int) m_slice_points.size() <= num_outputs) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "requires more slice points than output tensors "
          << "(found " << m_slice_points.size() << " slice points "
          << "and " << m_child_layers.size() << " output tensors)";
      LBANN_ERROR(err.str());
    }
    if (!std::is_sorted(m_slice_points.begin(), m_slice_points.end())) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "has unsorted slice points";
      LBANN_ERROR(err.str());
    }
    if (m_slice_points.front() < 0
        || m_slice_points.back() > input_dims[m_slice_axis]) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "expects slice points in the range "
          << "[0, " << input_dims[m_slice_axis] << "], "
          << "but found an invalid slice point ";
      if (m_slice_points.front() < 0) {
        err << "(" << m_slice_points.front() << ")";
      } else {
        err << "(" << m_slice_points.back() << ")";
      }
      LBANN_ERROR(err.str());
    }

    // Set output tensor dimensions
    auto output_dims = input_dims;
    for (int i = 0; i < num_outputs; ++i) {
      output_dims[m_slice_axis] = m_slice_points[i+1] - m_slice_points[i];
      set_output_dims(output_dims, i);
    }

  }

  /** Get slice points. */
  std::vector<int>& get_slice_points() { return m_slice_points; }
  /** Get slice points (const). */
  const std::vector<int>& get_slice_points() const { return m_slice_points; }

  protected:

  void fp_compute() override {

    // Input tensor
    const auto& input = get_prev_activations();
    const int width = input.Width();

    // Get number of contiguous regions in a tensor slice of width 1
    const auto& input_dims = get_input_dims();
    const int num_regions = std::accumulate(input_dims.begin(),
                                            input_dims.begin() + m_slice_axis,
                                            1,
                                            std::multiplies<int>());
    const int unit_region_size = std::accumulate(input_dims.begin() + m_slice_axis + 1,
                                                 input_dims.end(),
                                                 1,
                                                 std::multiplies<int>());

    // Get stride between contiguous regions in input tensor slices
    const int input_slice_dim = input_dims[m_slice_axis];
    const int input_region_stride = input_slice_dim * unit_region_size;

    // Populate output tensors with slices of input tensor
    for (int i = 0; i < get_num_children(); ++i) {
      auto& output = get_activations(i);

      // Get stride between contiguous regions in output tensor slices
      const int output_slice_dim = m_slice_points[i+1] - m_slice_points[i];
      const int output_region_stride = output_slice_dim * unit_region_size;

      // Get position of first contiguous region in input tensor
      const int input_region_start = m_slice_points[i] * unit_region_size;
      const int input_region_end = m_slice_points[i+1] * unit_region_size;

      // Populate current output tensor
      output.Resize(num_regions * output_region_stride, width);
      for (int region = 0; region < num_regions; ++region) {
        El::LockedView(*m_input_region_v,
                       input,
                       El::IR(input_region_start + region * input_region_stride,
                              input_region_end + region * input_region_stride),
                       El::ALL);
        El::View(*m_output_region_v,
                 output,
                 El::IR(region * output_region_stride,
                        (region+1) * output_region_stride),
                 El::ALL);
        El::Copy(*m_input_region_v, *m_output_region_v);
      }
    }

  }

  void bp_compute() override {

    // Gradient w.r.t. input
    auto& gradient_wrt_input = get_error_signals();

    // Get number of contiguous regions in a tensor slice of width 1
    const auto& input_dims = get_input_dims();
    const int num_regions = std::accumulate(input_dims.begin(),
                                            input_dims.begin() + m_slice_axis,
                                            1,
                                            std::multiplies<int>());
    const int unit_region_size = std::accumulate(input_dims.begin() + m_slice_axis + 1,
                                                 input_dims.end(),
                                                 1,
                                                 std::multiplies<int>());

    // Get stride between contiguous regions in input tensor slices
    const int input_slice_dim = input_dims[m_slice_axis];
    const int input_region_stride = input_slice_dim * unit_region_size;

    // Populate gradient w.r.t. input with slices of gradient w.r.t. output
    for (int i = 0; i < get_num_children(); ++i) {
      const auto& gradient_wrt_output = get_prev_error_signals(i);

      // Get stride between contiguous regions in output tensor slices
      const int output_slice_dim = m_slice_points[i+1] - m_slice_points[i];
      const int output_region_stride = output_slice_dim * unit_region_size;

      // Get position of first contiguous region in input tensor
      const int input_region_start = m_slice_points[i] * unit_region_size;
      const int input_region_end = m_slice_points[i+1] * unit_region_size;

      // Populate slice of gradient w.r.t. input
      for (int region = 0; region < num_regions; ++region) {
        El::View(*m_input_region_v,
                 gradient_wrt_input,
                 El::IR(input_region_start + region * input_region_stride,
                        input_region_end + region * input_region_stride),
                 El::ALL);
        El::LockedView(*m_output_region_v,
                       gradient_wrt_output,
                       El::IR(region * output_region_stride,
                              (region+1) * output_region_stride),
                       El::ALL);
        El::Copy(*m_output_region_v, *m_input_region_v);
      }

    }

  }

};

} // namespace lbann

#endif // LBANN_LAYER_SLICE_HPP_INCLUDED
