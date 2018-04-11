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
#include "lbann/utils/cublas_wrapper.hpp"

namespace lbann {

/** Crop layer. */
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class crop_layer : public transform_layer {
 private:

  std::vector<int> m_crop_dims;

  /** View into region of input tensor. */
  AbsDistMat* m_input_region_v;
  /** View into region of output tensor. */
  AbsDistMat* m_output_region_v;

 public:

  crop_layer(lbann_comm *comm,
             std::vector<int> dims,
             cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm),
      m_crop_dims(dims),
      m_input_region_v(nullptr),
      m_output_region_v(nullptr) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "crop layer currently only supports DATA_PARALLEL");
 
    // Parent layers for original tensor and crop position
    m_expected_num_parent_layers = 2;

  #if 0
  //  #ifdef LBANN_HAS_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      this->m_using_gpus = true;
       this->m_cudnn = cudnn;
    }
  #endif // LBANN_HAS_CUDNN

  }

  crop_layer(const crop_layer& other)
    : transform_layer(other),
      m_crop_dims(other.m_crop_dims),
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
    m_crop_dims = other.m_crop_dims;

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

  virtual ~crop_layer() override {
    if (m_input_region_v != nullptr)  { delete m_input_region_v; }
    if (m_output_region_v != nullptr) { delete m_output_region_v; }
  }

  crop_layer* copy() const override { return new crop_layer(*this); }
  std::string get_type() const override { return "crop"; }
  data_layout get_data_layout() const override { return T_layout; }

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
    this->m_neuron_dims = m_crop_dims;
    this->m_num_neuron_dims = m_neuron_dims.size();
    this->m_num_neurons = std::accumulate(m_neuron_dims.begin(),
                                          m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

    // Make sure crop position has correct dimensions
    if (this->m_parent_layers[1]->fp_output_dims(this)
        != get_prev_neuron_dims(1)) {
      LBANN_ERROR("crop position tensor input must match number of neuron dimensions");
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

    // Tensor dimensions
    const El::Int num_dims = get_num_neuron_dims(0);
    const auto& input_dims = get_prev_neuron_dims(0);
    const auto& output_dims = get_neuron_dims(0);

    // Input and output tensors
    const auto& local_crop_pos = get_local_prev_activations(1);
    const auto& local_input = get_local_prev_activations(0);
    auto& local_output = get_local_activations();

    // Crop each mini-batch sample
    #pragma omp parallel for
    for (El::Int s = 0; s < local_input.Width(); ++s) {

      // Determine crop position
      std::vector<int> crop_offsets;
      for (El::Int d = 0; d < num_dims; ++d) {
        const auto& pos = local_crop_pos(d, s);
        if (pos < DataType(0) || pos >= DataType(1)) {
          LBANN_ERROR("crop position not in range [0,1)");
        }
        const auto& num_offsets = m_neuron_dims[d] - m_crop_dims[d] + 1;
        crop_offsets.push_back((int)(pos * num_offsets));
      }

      // Copy entries from input tensor to output tensor
      std::vector<int> region_pos(num_dims, 0);
      while (region_pos[0] < m_crop_dims[0]) {

        // Copy entry from input tensor to output tensor
        int input_pos = region_pos[0] + crop_offsets[0];;
        int output_pos = region_pos[0];
        for (int d = 0; d < num_dims-1; ++d) {
          input_pos = (input_dims[d] * input_pos
                       + region_pos[d+1] + crop_offsets[d+1]);
          output_pos = output_dims[d] * output_pos + region_pos[d+1];
        }
        local_output(output_pos, s) = local_input(input_pos, s);
          
        // Move to next entry
        ++region_pos.back();
        for (int d = num_dims-1; d >= 1; --d) {
          if (region_pos[d] >= m_crop_dims[d]) {
            region_pos[d] = 0;
            ++region_pos[d-1];
          }
        }
        
      }

    }

  }

  void bp_compute_cpu() {

    // Tensor dimensions
    const El::Int num_dims = get_num_neuron_dims(0);
    const auto& input_dims = get_prev_neuron_dims(0);
    const auto& output_dims = get_neuron_dims(0);

    // Input and output tensors
    const auto& local_crop_pos = get_local_prev_activations(1);
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_gradient_wrt_input = get_local_error_signals(0);

    // Crop each mini-batch sample
    #pragma omp parallel for
    for (El::Int s = 0; s < local_gradient_wrt_output.Width(); ++s) {

      // Determine crop position
      std::vector<int> crop_offsets;
      for (El::Int d = 0; d < num_dims; ++d) {
        const auto& pos = local_crop_pos(d, s);
        if (pos < DataType(0) || pos >= DataType(1)) {
          LBANN_ERROR("crop position not in range [0,1)");
        }
        const auto& num_offsets = m_neuron_dims[d] - m_crop_dims[d] + 1;
        crop_offsets.push_back((int)(pos * num_offsets));
      }

      // Copy entries from input tensor to output tensor
      std::vector<int> region_pos(num_dims, 0);
      while (region_pos[0] < m_crop_dims[0]) {

        // Copy entry from input tensor to output tensor
        int input_pos = region_pos[0] + crop_offsets[0];;
        int output_pos = region_pos[0];
        for (int d = 0; d < num_dims-1; ++d) {
          input_pos = (input_dims[d] * input_pos
                       + region_pos[d+1] + crop_offsets[d+1]);
          output_pos = output_dims[d] * output_pos + region_pos[d+1];
        }
        local_gradient_wrt_input(input_pos, s)
          += local_gradient_wrt_output(output_pos, s);
          
        // Move to next entry
        ++region_pos.back();
        for (int d = num_dims-1; d >= 1; --d) {
          if (region_pos[d] >= m_crop_dims[d]) {
            region_pos[d] = 0;
            ++region_pos[d-1];
          }
        }
        
      }

    }

  }

  void fp_compute_gpu() {
    
    /// @todo Implement
    LBANN_ERROR("not yet implemented");

  #if 0
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else

    // Input tensor
    const auto& input_d = m_prev_activations_d[0];

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

      // Populate current output tensor
      if (num_regions == 1) {
        auto input_ptrs = input_d.get_locked_data();
        for (auto& ptr : input_ptrs) { ptr += input_region_start; }
        output_d.locked_attach(input_ptrs,
                               output_d.get_height(),
                               input_d.get_width_per_gpu(),
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
                                       this->m_mini_batch_size_per_gpu,
                                       input_d.get_leading_dim());
          output_region_d.attach(output_ptrs,
                                 output_region_stride,
                                 this->m_mini_batch_size_per_gpu,
                                 output_d.get_leading_dim());
          output_region_d.copy(input_region_d);
        }
      }
    }

  #endif // LBANN_HAS_CUDNN
  #endif // 0
  }

  void bp_compute_gpu() {
    /// @todo Implement
    LBANN_ERROR("not yet implemented");
  #if 0
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else

    // Gradient w.r.t. input
    auto& gradient_wrt_input_d = m_error_signals_d[0];

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
                                           this->m_mini_batch_size_per_gpu,
                                           gradient_wrt_input_d.get_leading_dim());
        gradient_wrt_output_region_d.locked_attach(gradient_wrt_output_ptrs,
                                                   output_region_stride,
                                                   this->m_mini_batch_size_per_gpu,
                                                   gradient_wrt_output_d.get_leading_dim());
        const int num_gpus = m_cudnn->get_num_gpus();
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
          CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(gpu)));
          cublas::geam(this->m_cudnn->get_cublas_handle(gpu),
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       gradient_wrt_output_region_d.get_height(),
                       this->m_mini_batch_size_per_gpu,
                       DataType(1),
                       gradient_wrt_output_region_d.get_locked_data(gpu),
                       gradient_wrt_output_region_d.get_leading_dim(),
                       DataType(1),
                       gradient_wrt_input_region_d.get_locked_data(gpu),
                       gradient_wrt_input_region_d.get_leading_dim(),
                       gradient_wrt_input_region_d.get_data(gpu),
                       gradient_wrt_input_region_d.get_leading_dim());
        }
      }

    }

  #endif // LBANN_HAS_CUDNN
  #endif // 0
  }

  std::vector<int> get_prev_neuron_dims(int parent_index = 0) const override {
    switch (parent_index) {
    case 0: return transform_layer::get_prev_neuron_dims(parent_index);
    case 1: return {get_num_neuron_dims()};
    default: LBANN_ERROR("attempted to access invalid parent of crop layer");
    }
  }

  int get_num_prev_neurons(int parent_index = 0) const override {
    const auto& prev_neuron_dims = get_prev_neuron_dims(parent_index);
    return std::accumulate(prev_neuron_dims.begin(),
                           prev_neuron_dims.end(),
                           1,
                           std::multiplies<int>());
  }

  int get_num_prev_neuron_dims(int parent_index = 0) const override {
    return get_prev_neuron_dims(parent_index).size();
  }

};

} // namespace lbann

#endif // LBANN_LAYER_CROP_HPP_INCLUDED
