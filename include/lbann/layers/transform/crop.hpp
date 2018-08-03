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

#ifndef LBANN_LAYER_CROP_HPP_INCLUDED
#define LBANN_LAYER_CROP_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** Crop layer.
 *  This layer extracts a crop from an input tensor, namely the
 *  activations tensor from the first parent layer. The position of
 *  the crop is controlled by the second parent layer, which should
 *  output one value in [0,1) for each tensor dimension.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class crop_layer : public transform_layer {
 private:

  /** View into region of input tensor. */
  AbsDistMat* m_input_region_v;
  /** View into region of output tensor. */
  AbsDistMat* m_output_region_v;

 public:

  crop_layer(lbann_comm *comm,
             std::vector<int> dims)
    : transform_layer(comm),
      m_input_region_v(nullptr),
      m_output_region_v(nullptr) {
    static_assert(Dev == El::Device::CPU,
                  "crop layer currently only supports CPU");
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "crop layer currently only supports DATA_PARALLEL");

    // Crop dimensions
    set_output_dims(dims);
 
    // Parent layers for original tensor and crop position
    m_expected_num_parent_layers = 2;

  }

  crop_layer(const crop_layer& other)
    : transform_layer(other),
      m_input_region_v(other.m_input_region_v),
      m_output_region_v(other.m_output_region_v) {
    if (m_input_region_v != nullptr) {
      m_input_region_v = m_input_region_v->Copy();
    }
    if (m_output_region_v != nullptr) {
      m_output_region_v = m_output_region_v->Copy();
    }
  }

  crop_layer& operator=(const crop_layer& other) {
    transform_layer::operator=(other);
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

  virtual ~crop_layer() override {
    if (m_input_region_v != nullptr)  { delete m_input_region_v; }
    if (m_output_region_v != nullptr) { delete m_output_region_v; }
  }

  crop_layer* copy() const override { return new crop_layer(*this); }
  std::string get_type() const override { return "crop"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

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
    if (this->using_gpus()) {
      fp_compute_gpu();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() override {
    if (this->using_gpus()) {
      bp_compute_gpu();
    } else {
      bp_compute_cpu();
    }
  }

  void fp_compute_cpu() {

    // Tensor dimensions
    const auto& input_dims = get_input_dims(0);
    const auto& output_dims = get_output_dims();
    const IntType num_dims = output_dims.size();

    // Input and output tensors
    const auto& local_crop_pos = get_local_prev_activations(1);
    const auto& local_input = get_local_prev_activations(0);
    auto& local_output = get_local_activations();

    // Crop each mini-batch sample
    #pragma omp parallel for
    for (IntType s = 0; s < local_input.Width(); ++s) {

      // Determine crop position
      std::vector<int> crop_offsets;
      for (IntType d = 0; d < num_dims; ++d) {
        const auto& pos = local_crop_pos(d, s);
        if (pos < DataType(0) || pos >= DataType(1)) {
          std::stringstream err;
          err << "crop position not in range [0,1) "
              << "(pos[" << d << "] = " << pos << ")";
          LBANN_ERROR(err.str());
        }
        const auto& num_offsets = input_dims[d] - output_dims[d] + 1;
        crop_offsets.push_back((int)(pos * num_offsets));
      }

      // Copy entries from input tensor to output tensor
      std::vector<int> output_pos(num_dims, 0);
      while (output_pos[0] < output_dims[0]) {

        // Copy entry from input tensor to output tensor
        int input_index = output_pos[0] + crop_offsets[0];
        int output_index = output_pos[0];
        for (int d = 0; d < num_dims-1; ++d) {
          input_index = (input_dims[d] * input_index
                         + output_pos[d+1] + crop_offsets[d+1]);
          output_index = output_dims[d] * output_index + output_pos[d+1];
        }
        local_output(output_index, s) = local_input(input_index, s);
          
        // Move to next entry
        ++output_pos.back();
        for (int d = num_dims-1; d >= 1; --d) {
          if (output_pos[d] >= output_dims[d]) {
            output_pos[d] = 0;
            ++output_pos[d-1];
          }
        }
        
      }

    }

  }

  void bp_compute_cpu() {

    // Tensor dimensions
    const auto& input_dims = get_input_dims(0);
    const auto& output_dims = get_output_dims();
    const IntType num_dims = output_dims.size();

    // Input and output tensors
    const auto& local_crop_pos = get_local_prev_activations(1);
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_gradient_wrt_input = get_local_error_signals(0);
    El::Zero(get_error_signals(1));

    // Crop each mini-batch sample
    #pragma omp parallel for
    for (IntType s = 0; s < local_gradient_wrt_output.Width(); ++s) {

      // Determine crop position
      std::vector<int> crop_offsets;
      for (IntType d = 0; d < num_dims; ++d) {
        const auto& pos = local_crop_pos(d, s);
        if (pos < DataType(0) || pos >= DataType(1)) {
          LBANN_ERROR("crop position not in range [0,1)");
        }
        const auto& num_offsets = input_dims[d] - output_dims[d] + 1;
        crop_offsets.push_back((int)(pos * num_offsets));
      }

      // Copy entries from input tensor to output tensor
      std::vector<int> output_pos(num_dims, 0);
      while (output_pos[0] < output_dims[0]) {

        // Copy entry from input tensor to output tensor
        int input_index = output_pos[0] + crop_offsets[0];
        int output_index = output_pos[0];
        for (int d = 0; d < num_dims-1; ++d) {
          input_index = (input_dims[d] * input_index
                         + output_pos[d+1] + crop_offsets[d+1]);
          output_index = output_dims[d] * output_index + output_pos[d+1];
        }
        local_gradient_wrt_input(input_index, s)
          = local_gradient_wrt_output(output_index, s);
          
        // Move to next entry
        ++output_pos.back();
        for (int d = num_dims-1; d >= 1; --d) {
          if (output_pos[d] >= output_dims[d]) {
            output_pos[d] = 0;
            ++output_pos[d-1];
          }
        }
        
      }

    }

  }

  void fp_compute_gpu() {
    /// @todo Implement
    LBANN_ERROR("not yet implemented");
  }

  void bp_compute_gpu() {
    /// @todo Implement
    LBANN_ERROR("not yet implemented");
  }

};

} // namespace lbann

#endif // LBANN_LAYER_CROP_HPP_INCLUDED
