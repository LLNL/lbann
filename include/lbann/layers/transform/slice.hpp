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
//
// slice.hpp - Slice layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_SLICE_HPP_INCLUDED
#define LBANN_LAYER_SLICE_HPP_INCLUDED

#include <utility>
#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/// Slice layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class slice_layer : public transform_layer {
 private:

  /** Tensor dimension to slice. */
  int m_slice_axis;
  /** Slice points for each child layer. */
  std::vector<int> m_slice_points;

 public:
  /// Constructor
  slice_layer(lbann_comm *comm,
              int slice_axis,
              std::vector<int> slice_points,
              cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm),
      m_slice_axis(slice_axis),
      m_slice_points(std::move(slice_points)) {

    // Slice layer has no limit on children
    m_expected_num_child_layers = -1;

  #ifdef __LIB_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN

  }

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
    return s.str();
  }

  slice_layer* copy() const override { return new slice_layer(*this); }

  std::string get_type() const override { return "slice"; }

  data_layout get_data_layout() const override { return T_layout; }

  void setup_dims() override {
    std::stringstream err;

    // Initialize previous neuron tensor dimensions
    transform_layer::setup_dims();

    // Set first and last slice points if needed
    if(m_slice_points.size() == m_child_layers.size() - 1) {
      m_slice_points.insert(m_slice_points.begin(), 0);
      m_slice_points.push_back(this->m_neuron_dims[m_slice_axis]);
    }

    // Check that slice points are valid
    if(m_slice_points.size() != m_child_layers.size() + 1
       || !std::is_sorted(m_slice_points.begin(), m_slice_points.end())) {
      err << __FILE__ << " " << __LINE__ << " :: slice_layer: ";
      if (!std::is_sorted(m_slice_points.begin(), m_slice_points.end())) {
        err << "slice points not sorted";
      } else {
        err << "number of slice points (" << m_slice_points.size()
            << ") != number of children (" << m_child_layers.size() << ") + 1"
            << " {" << get_layer_names(m_child_layers) << "}";
      }
      throw lbann_exception(err.str());
    }

  }

  protected:

  void fp_compute() override {
    if(this->m_using_gpus) {
      fp_compute_gpu();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
      bp_compute_gpu();
    } else {
      bp_compute_cpu();
    }
  }

  void fp_compute_cpu() {

    // Input tensor
    const auto& input = get_prev_activations();
    const int width = input.Width();

    // Get number of contiguous regions in a tensor slice of width 1
    const auto& input_dims = this->m_prev_neuron_dims;
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
      if (num_regions == 1) {
        El::LockedView(output,
                       input,
                       El::IR(input_region_start, input_region_end),
                       El::ALL);
      } else {
        for (int region = 0; region < num_regions; ++region) {
          const auto& input_region
            = El::LockedView(input,
                             El::IR(input_region_start + region * input_region_stride,
                                    input_region_end + region * input_region_stride),
                             El::ALL);
          auto&& output_region = El::View(output,
                                         El::IR(region * output_region_stride,
                                                (region+1) * output_region_stride),
                                         El::ALL);
          El::Copy(input_region, output_region);
        }
      }
    }

  }

  void bp_compute_cpu() {

    // Gradient w.r.t. input
    auto& gradient_wrt_input = get_error_signals();

    // Get number of contiguous regions in a tensor slice of width 1
    const auto& input_dims = this->m_prev_neuron_dims;
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
        auto&& gradient_wrt_input_region
          = El::View(gradient_wrt_input,
                     El::IR(input_region_start + region * input_region_stride,
                            input_region_end + region * input_region_stride),
                     El::ALL);
        const auto& gradient_wrt_output_region
          = El::View(gradient_wrt_output,
                     El::IR(region * output_region_stride,
                            (region+1) * output_region_stride),
                     El::ALL);
        El::Axpy(DataType(1),
                 gradient_wrt_output_region,
                 gradient_wrt_input_region);
      }

    }

  }

  void fp_compute_gpu() {
  #ifndef __LIB_CUDNN
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: slice_layer: cuDNN not detected";
    throw lbann_exception(err.str());
  #else

    // Input tensor
    const auto& input_d = m_prev_activations_d[0];
    const int width_per_gpu = input_d.get_width_per_gpu();

    // Get number of contiguous regions in a tensor slice of width 1
    const auto& input_dims = this->m_prev_neuron_dims;
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
    cudnn::matrix input_region_d(m_cudnn), output_region_d(m_cudnn);
    for (int i = 0; i < get_num_children(); ++i) {
      auto& output_d = m_activations_d[i];

      // Get stride between contiguous regions in output tensor slices
      const int output_slice_dim = m_slice_points[i+1] - m_slice_points[i];
      const int output_region_stride = output_slice_dim * unit_region_size;

      // Get position of first contiguous region in input tensor
      const int input_region_start = m_slice_points[i] * unit_region_size;
      const int input_region_end = m_slice_points[i+1] * unit_region_size;

      // Populate current output tensor
      output_d.resize(num_regions * output_region_stride, width_per_gpu);
      if (num_regions == 1) {
        auto input_ptrs = input_d.get_locked_data();
        for (auto& ptr : input_ptrs) { ptr += input_region_start; }
        output_d.locked_attach(input_ptrs,
                               output_d.get_height(),
                               width_per_gpu,
                               input_d.get_leading_dim());
      } else {
        for (int region = 0; region < num_regions; ++region) {
          auto input_ptrs = input_d.get_locked_data();
          auto output_ptrs = output_d.get_data();
          for (auto& ptr : input_ptrs) {
            ptr += input_region_start + region * input_region_stride;
          }
          for (auto& ptr : output_ptrs) {
            ptr += region * output_region_stride;
          }
          input_region_d.locked_attach(input_ptrs,
                                       output_region_stride,
                                       width_per_gpu,
                                       input_d.get_leading_dim());
          output_region_d.attach(output_ptrs,
                                 output_region_stride,
                                 width_per_gpu,
                                 output_d.get_leading_dim());
          output_region_d.copy(input_region_d);
        }
      }
    }

  #endif // __LIB_CUDNN
  }

  void bp_compute_gpu() {
  #ifndef __LIB_CUDNN
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: slice_layer: cuDNN not detected";
    throw lbann_exception(err.str());
  #else

    // Gradient w.r.t. input
    auto& gradient_wrt_input_d = m_error_signals_d[0];
    const int width_per_gpu = gradient_wrt_input_d.get_width_per_gpu();

    // Get number of contiguous regions in a tensor slice of width 1
    const auto& input_dims = this->m_prev_neuron_dims;
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
    cudnn::matrix gradient_wrt_input_region_d(m_cudnn);
    cudnn::matrix gradient_wrt_output_region_d(m_cudnn);
    for (int i = 0; i < get_num_children(); ++i) {
      const auto& gradient_wrt_output_d = m_prev_error_signals_d[i];

      // Get stride between contiguous regions in output tensor slices
      const int output_slice_dim = m_slice_points[i+1] - m_slice_points[i];
      const int output_region_stride = output_slice_dim * unit_region_size;

      // Get position of first contiguous region in input tensor
      const int input_region_start = m_slice_points[i] * unit_region_size;
      const int input_region_end = m_slice_points[i+1] * unit_region_size;

      // Populate slice of gradient w.r.t. input
      for (int region = 0; region < num_regions; ++region) {
        auto gradient_wrt_input_ptrs = gradient_wrt_input_d.get_data();
        auto gradient_wrt_output_ptrs = gradient_wrt_output_d.get_locked_data();
        for (auto& ptr : gradient_wrt_input_ptrs) {
          ptr += input_region_start + region * input_region_stride;
        }
        for (auto& ptr : gradient_wrt_output_ptrs) {
          ptr += region * output_region_stride;
        }
        gradient_wrt_input_region_d.attach(gradient_wrt_input_ptrs,
                                           output_region_stride,
                                           width_per_gpu,
                                           gradient_wrt_input_d.get_leading_dim());
        gradient_wrt_output_region_d.locked_attach(gradient_wrt_output_ptrs,
                                                   output_region_stride,
                                                   width_per_gpu,
                                                   gradient_wrt_output_d.get_leading_dim());
        const DataType one = DataType(1);
        const int num_gpus = m_cudnn->get_num_gpus();
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
          CHECK_CUBLAS(cublas::geam(this->m_cudnn->get_cublas_handle(gpu),
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    gradient_wrt_output_region_d.get_height(),
                                    gradient_wrt_output_region_d.get_width_per_gpu(),
                                    &one,
                                    gradient_wrt_output_region_d.get_locked_data(gpu),
                                    gradient_wrt_output_region_d.get_leading_dim(),
                                    &one,
                                    gradient_wrt_input_region_d.get_locked_data(gpu),
                                    gradient_wrt_input_region_d.get_leading_dim(),
                                    gradient_wrt_input_region_d.get_data(gpu),
                                    gradient_wrt_input_region_d.get_leading_dim()));
        }
      }

    }

  #endif // __LIB_CUDNN
  }

  const std::vector<int> fp_output_dims(const Layer* next_layer) const override {

    // Return all neurons if input is null
    if(next_layer == nullptr) {
      return m_neuron_dims;
    }

    // Check if input is in the list of child layers
    const int child_index = (std::find(this->m_child_layers.begin(),
                                       this->m_child_layers.end(),
                                       next_layer)
                             - this->m_child_layers.begin());
    if(child_index >= (int) this->m_child_layers.size()) {
      return m_neuron_dims;
    }

    // Return slice dimensions
    std::vector<int> neuron_dims = m_neuron_dims;
    neuron_dims[m_slice_axis] = m_slice_points[child_index+1] - m_slice_points[child_index];
    return neuron_dims;

  }

};

} // namespace lbann

#endif // LBANN_LAYER_SLICE_HPP_INCLUDED
