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

#ifndef LBANN_LAYER_SLICE_HPP_INCLUDED
#define LBANN_LAYER_SLICE_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief Slice tensor along a specified dimension.
 *
 *  Suppose we slice a @f$ D_1 \times\cdots\times D_n @f$ input tensor
 *  along the dimension @f$ k @f$. We specify slice points
 *  @f$ s_1,\cdots,s_\ell @f$, which are strictly increasing and have
 *  @f$ s_1 = 0 @f$ and @f$ s_\ell=D_k @f$. The @f$ i @f$th output
 *  tensor is then a
 *  @f$ D_1 \times\cdots
 *    \times D_{i-1}\times (s_i - s_{i-1}) \times D_{i+1} \times
 *    \cdots\times D_n @f$
 *  tensor.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class slice_layer : public transform_layer {
public:

  slice_layer(lbann_comm *comm,
              El::Int slice_dim,
              std::vector<El::Int> slice_points)
    : transform_layer(comm),
      m_slice_dim(slice_dim),
      m_slice_points(slice_points) {
    this->m_expected_num_child_layers = -1; // No limit on children
  }

  slice_layer(const slice_layer& other)
    : transform_layer(other),
      m_slice_dim(other.m_slice_dim),
      m_slice_points(other.m_slice_points) {
    m_input_v.reset(other.m_input_v ? other.m_input_v->Copy() : nullptr);
    m_output_v.reset(other.m_output_v ? other.m_output_v->Copy() : nullptr);
  }

  slice_layer& operator=(const slice_layer& other) {
    transform_layer::operator=(other);
    m_slice_dim = other.m_slice_dim;
    m_slice_points = other.m_slice_points;
    m_input_v.reset(other.m_input_v ? other.m_input_v->Copy() : nullptr);
    m_output_v.reset(other.m_output_v ? other.m_output_v->Copy() : nullptr);
  }

  slice_layer* copy() const override { return new slice_layer(*this); }
  std::string get_type() const override { return "slice"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  /** Get slice points. */
  std::vector<El::Int>& get_slice_points() { return m_slice_points; }
  /** Get slice points (const). */
  std::vector<El::Int> get_slice_points() const { return m_slice_points; }

  description get_description() const override {
    auto&& desc = transform_layer::get_description();
    desc.add("Slice dimension", m_slice_dim);
    std::stringstream ss;
    for (size_t i = 0; i < m_slice_points.size(); ++i) {
      ss << (i > 0 ? ", " : "") << m_slice_points[i];
    }
    desc.add("Slice points", ss.str());
    return desc;
  }

protected:

  void setup_matrices(const El::Grid& grid) override {
    transform_layer::setup_matrices(grid);
    const auto& input = get_prev_activations();
    m_input_v.reset(input.Construct(input.Grid(), input.Root()));
    m_output_v.reset(input.Construct(input.Grid(), input.Root()));
  }

  void setup_dims() override {
    transform_layer::setup_dims();
    const auto& input_dims = get_input_dims();
    const auto& num_outputs = get_num_children();

    // Check that slice parameters are valid
    std::stringstream err;
    if (m_slice_dim < 0 || m_slice_dim >= (El::Int) input_dims.size()) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "has " << input_dims.size() << " dimensions, "
          << "but attempted to slice along dimension " << m_slice_dim;
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
        || m_slice_points.back() > input_dims[m_slice_dim]) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "expects slice points in the range "
          << "[0, " << input_dims[m_slice_dim] << "], "
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
      output_dims[m_slice_dim] = m_slice_points[i+1] - m_slice_points[i];
      set_output_dims(output_dims, i);
    }

  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    const auto& num_outputs = get_num_children();
    const auto& input_dims = get_input_dims();

    // Divide input tensor into unit slices along slice dimension
    // Note: Each unit slice is divided into contiguous "unit blocks"
    const auto& input_num_unit_slices = input_dims[m_slice_dim];
    const auto& blocks_per_slice
      = std::accumulate(&input_dims[0], &input_dims[m_slice_dim],
                        1, std::multiplies<int>());
    const auto& unit_block_size
      = std::accumulate(input_dims.begin() + m_slice_dim + 1,
                        input_dims.end(),
                        1, std::multiplies<int>());
    const auto& input_block_stride = (input_num_unit_slices
                                      * unit_block_size);

    // Populate output tensors with slices of input tensor
    const auto& input = get_prev_activations();
    for (int i = 0; i < num_outputs; ++i) {
      const auto& output_dims = get_output_dims(i);
      const auto& output_size = get_output_size(i);
      auto& output = get_activations(i);
      output.Empty(false);

      // Divide output tensor into unit slices
      const auto& output_num_unit_slices = output_dims[m_slice_dim];

      // Merge unit slices and get first contiguous input block
      const auto& block_size = output_num_unit_slices * unit_block_size;
      const auto& input_block_offset = m_slice_points[i] * unit_block_size;
      El::LockedView(*m_input_v, input,
                     El::IR(input_block_offset,
                            input_block_offset + block_size),
                     El::ALL);

      // Populate output tensor one block at a time
      // Note: If there is only one block, output can be a view
      if (blocks_per_slice > 1) {
        output.AlignWith(*m_input_v);
        output.Resize(output_size, mini_batch_size);
        for (int block = 0; block < blocks_per_slice; ++block) {
          const auto& input_offset = (input_block_offset
                                      + block * input_block_stride);
          const auto& output_offset = block * block_size;
          El::LockedView(*m_input_v, input,
                         El::IR(input_offset, input_offset + block_size),
                         El::ALL);
          El::View(*m_output_v, output,
                   El::IR(output_offset, output_offset + block_size),
                   El::ALL);
          El::Copy(*m_input_v, *m_output_v);
        }
      } else {
        El::LockedView(output, *m_input_v);
      }

    }

  }

  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override {
    const auto& num_outputs = get_num_children();
    const auto& input_dims = get_input_dims();

    // Initialize gradient w.r.t. input tensor
    auto& gradient_wrt_input = get_error_signals();
    gradient_wrt_input.Empty(false);
    gradient_wrt_input.AlignWith(get_prev_activations());
    gradient_wrt_input.Resize(get_input_size(), mini_batch_size);
    if (m_slice_points[0] != 0
        || m_slice_points[num_outputs] != input_dims[m_slice_dim]) {
      El::Zero(gradient_wrt_input);
    }

    // Divide input tensor into unit slices along slice dimension
    // Note: Each unit slice is divided into contiguous "unit blocks"
    const auto& input_num_unit_slices = input_dims[m_slice_dim];
    const auto& blocks_per_slice
      = std::accumulate(&input_dims[0], &input_dims[m_slice_dim],
                        1, std::multiplies<int>());
    const auto& unit_block_size
      = std::accumulate(input_dims.begin() + m_slice_dim + 1,
                        input_dims.end(),
                        1, std::multiplies<int>());
    const auto& input_block_stride = (input_num_unit_slices
                                      * unit_block_size);

    // Populate slices of gradient w.r.t. input tensor
    for (int i = 0; i < num_outputs; ++i) {
      const auto& output_dims = get_output_dims(i);
      const auto& gradient_wrt_output = get_prev_error_signals(i);

      // Divide output tensor into unit slices
      const auto& output_num_unit_slices = output_dims[m_slice_dim];

      // Merge unit slices
      const auto& block_size = output_num_unit_slices * unit_block_size;
      const auto& input_block_offset = m_slice_points[i] * unit_block_size;

      // Populate gradient w.r.t. input tensor one block at a time
      for (int block = 0; block < blocks_per_slice; ++block) {
        const auto& input_offset = (input_block_offset
                                    + block * input_block_stride);
        const auto& output_offset = block * block_size;
        El::LockedView(*m_output_v, gradient_wrt_output,
                       El::IR(output_offset, output_offset + block_size),
                       El::ALL);
        El::View(*m_input_v, gradient_wrt_input,
                 El::IR(input_offset, input_offset + block_size),
                 El::ALL);
        El::Copy(*m_output_v, *m_input_v);
      }

    }

  }

  void fp_compute() override {}
  void bp_compute() override {}

private:

  /** Tensor dimension to slice. */
  El::Int m_slice_dim;
  /** Slice points for each child layer. */
  std::vector<El::Int> m_slice_points;

  /** View into input tensor. */
  std::unique_ptr<AbsDistMat> m_input_v;
  /** View into output tensor. */
  std::unique_ptr<AbsDistMat> m_output_v;

};

} // namespace lbann

#endif // LBANN_LAYER_SLICE_HPP_INCLUDED
