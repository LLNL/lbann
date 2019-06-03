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

#ifndef LBANN_LAYER_CROP_HPP_INCLUDED
#define LBANN_LAYER_CROP_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief Crop tensor.
 *
 *  Extract a crop from an @f$ N @f$-D tensor. The second input tensor
 *  is interpreted as a normalized crop position in @f$ [0,1)^N
 *  @f$. For images in CHW format, a position of (0,0,0) corresponds
 *  to the red-top-left corner and (1,1,1) to the blue-bottom-right
 *  corner. The crop size is determined at setup.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class crop_layer : public transform_layer {
public:

  crop_layer(lbann_comm *comm,
             std::vector<int> dims)
    : transform_layer(comm) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "crop layer only supports DATA_PARALLEL");
    set_output_dims(dims);
    this->m_expected_num_parent_layers = 2;
  }

  crop_layer(const crop_layer& other)
    : transform_layer(other),
      m_input_v(other.m_input_v ?
                other.m_input_v->Copy() : nullptr),
      m_output_v(other.m_output_v ?
                 other.m_output_v->Copy() : nullptr),
      m_crop_pos_v(other.m_crop_pos_v ?
                   other.m_crop_pos_v->Copy() : nullptr){}
  crop_layer& operator=(const crop_layer& other) {
    transform_layer::operator=(other);
    m_input_v.reset(other.m_input_v ?
                    other.m_input_v->Copy() : nullptr);
    m_output_v.reset(other.m_output_v ?
                     other.m_output_v->Copy() : nullptr);
    m_crop_pos_v.reset(other.m_crop_pos_v ?
                       other.m_crop_pos_v->Copy() : nullptr);
    return *this;
  }

  crop_layer* copy() const override { return new crop_layer(*this); }
  std::string get_type() const override { return "crop"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  void setup_matrices(const El::Grid& grid) override {
    transform_layer::setup_matrices(grid);
    const auto& input = get_prev_activations();
    const auto& dist = input.DistData();
    m_input_v.reset(input.Construct(input.Grid(), input.Root()));
    m_output_v.reset(input.Construct(input.Grid(), input.Root()));

    /// @todo Setup the input tensor with this data distribution
    m_crop_pos_v.reset(AbsDistMat::Instantiate(*dist.grid,
                                               dist.root,
                                               El::STAR,
                                               dist.rowDist,
                                               (dist.blockWidth == 1 ?
                                                El::ELEMENT : El::BLOCK),
                                               El::Device::CPU));

  }

  void setup_dims() override {
    transform_layer::setup_dims();
    std::stringstream err;

    // Make sure input tensors have valid dimensions
    const auto& input_dims = get_input_dims(0);
    const auto& loc_dims = get_input_dims(1);
    const auto& output_dims = get_output_dims();
    if (input_dims.size() != output_dims.size()) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "expects a crop input tensor with "
          << output_dims.size() << " dimensions, "
          << "but parent layer "
          << "\"" << m_parent_layers[0]->get_name() << "\" "
          << "outputs a tensor with "
          << input_dims.size() << " dimensions";
      LBANN_ERROR(err.str());
    }
    if (loc_dims.size() != 1 || loc_dims[0] != (int) input_dims.size()) {
      err << get_type() << " layer \"" << get_name() << "\" "
          << "expects a 1D crop position tensor with "
          << output_dims.size() << " entries, "
          << "but parent layer "
          << "\"" << m_parent_layers[1]->get_name() << "\" "
          << "outputs a tensor with dimensions ";
      for (size_t i = 0; i < loc_dims.size(); ++i) {
        err << (i > 0 ? " x " : "") << loc_dims[i];
      }
      LBANN_ERROR(err.str());
    }

  }

protected:

  void fp_compute() override {
    switch (get_input_dims().size()) {
    case 3: fp_compute_3d(); break;
    default: fp_compute_nd();
    }
  }

  void bp_compute() override {
    switch (get_input_dims().size()) {
    case 3: bp_compute_3d(); break;
    default: bp_compute_nd();
    }
  }

private:
  /** View into input tensor. */
  std::unique_ptr<AbsDistMat> m_input_v;
  /** View into output tensor. */
  std::unique_ptr<AbsDistMat> m_output_v;
  /** View into crop positions. */
  std::unique_ptr<AbsDistMat> m_crop_pos_v;

  /** Forward prop implementation for n-dimensional tensors. */
  void fp_compute_nd() {

    // Input and output tensors
    const auto& input = get_prev_activations(0);
    auto& output = get_activations();

    // Tensor dimensions
    const auto& input_dims = get_input_dims(0);
    const auto& output_dims = get_output_dims();
    const El::Int num_dims = output_dims.size();
    const auto& local_width = input.LocalWidth();
    const auto& region_size = output_dims.back();

    // Get crop position
    m_crop_pos_v->Empty(false);
    m_crop_pos_v->AlignWith(input);
    const auto& input1 = get_prev_activations(1);
    if (m_crop_pos_v->DistData() == input1.DistData()) {
      El::LockedView(*m_crop_pos_v, input1);
    } else {
      El::Copy(input1, *m_crop_pos_v);
    }
    const auto& local_crop_pos = m_crop_pos_v->LockedMatrix();

    // Crop each local mini-batch sample
    // BVE_FIXME LBANN_OMP_PARALLEL_FOR
    for (El::Int local_col = 0; local_col < local_width; ++local_col) {
      const auto& col = input.GlobalCol(local_col);

      // Determine crop position
      std::vector<El::Int> crop_offsets;
      for (El::Int d = 0; d < num_dims; ++d) {
        const auto& pos = local_crop_pos(d, local_col);
        if (pos < DataType(0) || pos > DataType(1)) {
          std::stringstream err;
          err << "crop position not in range [0,1] (pos=(";
          for (El::Int i = 0; i < local_crop_pos.Height(); ++i) {
            err << (i > 0 ? "," : "") << local_crop_pos(i, local_col);
          }
          err << "))";
          LBANN_ERROR(err.str());
        }
        const El::Int num_offsets = input_dims[d] - output_dims[d] + 1;
        crop_offsets.push_back(std::min(El::Int(pos * num_offsets),
                                        num_offsets - 1));
      }

      // Copy contiguous regions from input tensor to output tensor
      std::vector<El::Int> output_pos(num_dims, 0);
      while (output_pos[0] < output_dims[0]) {

        // Copy region from input tensor to output tensor
        auto input_index = output_pos[0] + crop_offsets[0];
        auto output_index = output_pos[0];
        for (El::Int d = 1; d < num_dims; ++d) {
          input_index = (input_dims[d] * input_index
                         + output_pos[d] + crop_offsets[d]);
          output_index = output_dims[d] * output_index + output_pos[d];
        }
        El::LockedView(*m_input_v,
                       input,
                       El::IR(input_index, input_index + region_size),
                       El::IR(col));
        El::View(*m_output_v,
                 output,
                 El::IR(output_index, output_index + region_size),
                 El::IR(col));
        El::Copy(*m_input_v, *m_output_v);

        // Move to next contiguous region
        output_pos.back() += region_size;
        for (El::Int d = num_dims-1; d >= 1; --d) {
          if (output_pos[d] >= output_dims[d]) {
            output_pos[d] = 0;
            ++output_pos[d-1];
          }
        }

      }

    }

  }

  /** Backward prop implementation for n-dimensional tensors. */
  void bp_compute_nd() {

    // Clear error signals
    El::Zero(get_error_signals(0));
    El::Zero(get_error_signals(1));

    // Input and gradient tensors
    const auto& gradient_wrt_output = get_prev_error_signals();
    auto& gradient_wrt_input = get_error_signals(0);
    const auto& local_crop_pos = m_crop_pos_v->LockedMatrix();

    // Tensor dimensions
    const auto& input_dims = get_input_dims(0);
    const auto& output_dims = get_output_dims();
    const El::Int num_dims = output_dims.size();
    const auto& local_width = gradient_wrt_input.LocalWidth();
    const auto& region_size = output_dims.back();

    // Populate error signal for each local mini-batch sample
    // BVE_FIXME LBANN_OMP_PARALLEL_FOR
    for (El::Int local_col = 0; local_col < local_width; ++local_col) {
      const auto& col = gradient_wrt_input.GlobalCol(local_col);

      // Determine crop position
      std::vector<El::Int> crop_offsets;
      for (El::Int d = 0; d < num_dims; ++d) {
        const auto& pos = local_crop_pos(d, local_col);
        if (pos < DataType(0) || pos > DataType(1)) {
          std::stringstream err;
          err << "crop position not in range [0,1] (pos=(";
          for (El::Int i = 0; i < local_crop_pos.Height(); ++i) {
            err << (i > 0 ? "," : "") << local_crop_pos(i, local_col);
          }
          err << "))";
          LBANN_ERROR(err.str());
        }
        const El::Int num_offsets = input_dims[d] - output_dims[d] + 1;
        crop_offsets.push_back(std::min(El::Int(pos * num_offsets),
                                        num_offsets - 1));
      }

      // Populate contiguous regions in gradient w.r.t. input tensor
      std::vector<El::Int> output_pos(num_dims, 0);
      while (output_pos[0] < output_dims[0]) {

        // Copy region
        auto input_index = output_pos[0] + crop_offsets[0];
        auto output_index = output_pos[0];
        for (El::Int d = 1; d < num_dims; ++d) {
          input_index = (input_dims[d] * input_index
                         + output_pos[d] + crop_offsets[d]);
          output_index = output_dims[d] * output_index + output_pos[d];
        }
        El::LockedView(*m_output_v,
                       gradient_wrt_output,
                       El::IR(output_index, output_index + region_size),
                       El::IR(col));
        El::View(*m_input_v,
                 gradient_wrt_input,
                 El::IR(input_index, input_index + region_size),
                 El::IR(col));
        El::Copy(*m_output_v, *m_input_v);

        // Move to next contiguous region
        output_pos.back() += region_size;
        for (El::Int d = num_dims-1; d >= 1; --d) {
          if (output_pos[d] >= output_dims[d]) {
            output_pos[d] = 0;
            ++output_pos[d-1];
          }
        }

      }

    }

  }

  /** Forward prop implementation for 3D tensors.
   *  E.g. image data.
   */
  void fp_compute_3d();
  /** Backward prop implementation for 3D tensors.
   *  E.g. image data.
   */
  void bp_compute_3d();

};

} // namespace lbann

#endif // LBANN_LAYER_CROP_HPP_INCLUDED
