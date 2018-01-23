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

#ifndef LBANN_LAYER_CONCATENATION_HPP_INCLUDED
#define LBANN_LAYER_CONCATENATION_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** Concatenation layer.
 *  This layer concatenates input tensors along a specified axis.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class concatenation_layer : public transform_layer {
 private:

  /** Tensor dimension to concatenation. */
  int m_concatenation_axis;
  /** Concatenation points for each child layer. */
  std::vector<int> m_concatenation_points;

  /** View into region of input tensor. */
  AbsDistMat *m_input_region_v;
  /** View into region of output tensor. */
  AbsDistMat *m_output_region_v;

 public:
  /// Constructor
  concatenation_layer(lbann_comm *comm,
                      int concatenation_axis,
                      cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm),
      m_concatenation_axis(concatenation_axis),
      m_input_region_v(nullptr),
      m_output_region_v(nullptr) {

    // Concatenation layer has no limit on parents
    m_expected_num_parent_layers = -1;

  #ifdef __LIB_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN

  }

  concatenation_layer(const concatenation_layer& other)
    : transform_layer(other),
      m_concatenation_axis(other.m_concatenation_axis),
      m_concatenation_points(other.m_concatenation_points),
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

  concatenation_layer& operator=(const concatenation_layer& other) {
    transform_layer::operator=(other);
    m_concatenation_axis = other.m_concatenation_axis;
    m_concatenation_points = other.m_concatenation_points;

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

  virtual ~concatenation_layer() override {
    if (m_input_region_v != nullptr)  { delete m_input_region_v; }
    if (m_output_region_v != nullptr) { delete m_output_region_v; }
  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
    s << " concatenation; concatenation_axis: "
      << m_concatenation_axis << " parents: ";
    for (size_t h=0; h<this->m_parent_layers.size(); h++) {
      s << this->m_parent_layers[h]->get_name() << " " << this->m_parent_layers[h]->get_type() << " ";
    }
    s << " concatenation_points: ";
    for (size_t h=0; h<this->m_concatenation_points.size(); h++) {
      s << this->m_concatenation_points[h] << " ";
    }
    s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    return s.str();
  }

  concatenation_layer* copy() const override { return new concatenation_layer(*this); }

  std::string get_type() const override { return "concatenation"; }

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
    std::stringstream err;

    // Initialize previous neuron tensor dimensions
    transform_layer::setup_dims();

    // Check that layer has at least one parent
    const int num_parents = get_num_parents();
    if (num_parents <= 0) {
      err << __FILE__ << " " << __LINE__ << " :: concatenation_layer: "
          << "concatenation layer has no parents";
      throw lbann_exception(err.str());
    }

    // Get concatenation axis indices corresponding to parent layers
    m_concatenation_points.empty();
    m_concatenation_points.push_back(0);
    for(int i = 0; i < num_parents; ++i) {
      const auto& parent_dims = get_prev_neuron_dims(i);
      
      // Check that parent layer has valid dimensions
      const auto& first_parent_dims = get_prev_neuron_dims(0);
      if (parent_dims.size() != first_parent_dims.size()) {
        err << __FILE__ << " " << __LINE__ << " :: concatenation_layer: "
            << "parent layer has invalid number of dimensions";
        throw lbann_exception(err.str());
      }
      for (size_t d = 0; d < parent_dims.size(); ++d) {
        if ((int) d != m_concatenation_axis
            && parent_dims[d] != first_parent_dims[d]) {
          err << __FILE__ << " " << __LINE__ << " :: concatenation_layer: "
              << "parent layer has invalid dimensions";
          throw lbann_exception(err.str());
        }
      }

      // Get concatentation axis upper bound for parent layer
      m_concatenation_points.push_back(m_concatenation_points.back()
                                       + parent_dims[m_concatenation_axis]);

    }    

    // Update neuron dimensions
    this->m_neuron_dims[m_concatenation_axis] = m_concatenation_points.back();
    this->m_num_neurons = std::accumulate(m_neuron_dims.begin(),
                                          m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

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

    // Gradient w.r.t. input
    auto& output = get_activations();

    // Get number of contiguous regions in a tensor slice of width 1
    const auto& output_dims = this->m_neuron_dims;
    const int num_regions = std::accumulate(output_dims.begin(),
                                            output_dims.begin() + m_concatenation_axis,
                                            1,
                                            std::multiplies<int>());
    const int unit_region_size = std::accumulate(output_dims.begin() + m_concatenation_axis + 1,
                                                 output_dims.end(),
                                                 1,
                                                 std::multiplies<int>());

    // Get stride between contiguous regions in output tensor slices
    const int output_slice_dim = output_dims[m_concatenation_axis];
    const int output_region_stride = output_slice_dim * unit_region_size;
    
    // Populate output with slices of inputs
    for (int i = 0; i < get_num_parents(); ++i) {
      const auto& input = get_prev_activations(i);

      // Get stride between contiguous regions in input tensor slices
      const int input_slice_dim = m_concatenation_points[i+1] - m_concatenation_points[i];
      const int input_region_stride = input_slice_dim * unit_region_size;

      // Get position of first contiguous region in output tensor
      const int output_region_start = m_concatenation_points[i] * unit_region_size;
      const int output_region_end = m_concatenation_points[i+1] * unit_region_size;

      // Populate concatenation of gradient w.r.t. input
      for (int region = 0; region < num_regions; ++region) {
        El::LockedView(*m_input_region_v,
                       input,
                       El::IR(region * input_region_stride,
                              (region+1) * input_region_stride),
                       El::ALL);
        El::View(*m_output_region_v,
                 output,
                 El::IR(output_region_start + region * output_region_stride,
                        output_region_end + region * output_region_stride),
                 El::ALL);
        El::Copy(*m_input_region_v, *m_output_region_v);
      }

    }

  }

  void bp_compute_cpu() {

    // Gradient w.r.t. output
    const auto& gradient_wrt_output = get_prev_error_signals();

    // Get number of contiguous regions in a tensor slice of width 1
    const auto& output_dims = this->m_neuron_dims;
    const int num_regions = std::accumulate(output_dims.begin(),
                                            output_dims.begin() + m_concatenation_axis,
                                            1,
                                            std::multiplies<int>());
    const int unit_region_size = std::accumulate(output_dims.begin() + m_concatenation_axis + 1,
                                                 output_dims.end(),
                                                 1,
                                                 std::multiplies<int>());

    // Get stride between contiguous regions in input tensor slice
    const int output_slice_dim = output_dims[m_concatenation_axis];
    const int output_region_stride = output_slice_dim * unit_region_size;
    
    // Populate output tensors with concatenations of input tensor
    for (int i = 0; i < get_num_parents(); ++i) {
      auto& gradient_wrt_input = get_error_signals(i);

      // Get stride between contiguous regions in output tensor concatenations
      const int input_slice_dim = m_concatenation_points[i+1] - m_concatenation_points[i];
      const int input_region_stride = input_slice_dim * unit_region_size;

      // Get position of first contiguous region in output tensor
      const int output_region_start = m_concatenation_points[i] * unit_region_size;
      const int output_region_end = m_concatenation_points[i+1] * unit_region_size;

      // Populate current output tensor
      for (int region = 0; region < num_regions; ++region) {
        El::LockedView(*m_output_region_v,
                       gradient_wrt_output,
                       El::IR(output_region_start + region * output_region_stride,
                              output_region_end + region * output_region_stride),
                       El::ALL);
        El::View(*m_input_region_v,
                 gradient_wrt_input,
                 El::IR(region * input_region_stride,
                        (region+1) * input_region_stride),
                 El::ALL);
        El::Axpy(DataType(1), *m_output_region_v, *m_input_region_v);
      }
    }

  }

  void fp_compute_gpu() {
  #ifndef __LIB_CUDNN
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: concatenation_layer: cuDNN not detected";
    throw lbann_exception(err.str());
  #else

    // Gradient w.r.t. input
    auto& output_d = m_activations_d[0];

    // Get number of contiguous regions in a tensor slice of width 1
    const auto& output_dims = this->m_neuron_dims;
    const int num_regions = std::accumulate(output_dims.begin(),
                                            output_dims.begin() + m_concatenation_axis,
                                            1,
                                            std::multiplies<int>());
    const int unit_region_size = std::accumulate(output_dims.begin() + m_concatenation_axis + 1,
                                                 output_dims.end(),
                                                 1,
                                                 std::multiplies<int>());

    // Get stride between contiguous regions in output tensor slices
    const int output_slice_dim = output_dims[m_concatenation_axis];
    const int output_region_stride = output_slice_dim * unit_region_size;
    
    // Populate output with slices of inputs
    cudnn::matrix input_region_d(m_cudnn), output_region_d(m_cudnn);
    for (int i = 0; i < get_num_parents(); ++i) {
      const auto& input_d = m_prev_activations_d[i];

      // Get stride between contiguous regions in input tensor slices
      const int input_slice_dim = m_concatenation_points[i+1] - m_concatenation_points[i];
      const int input_region_stride = input_slice_dim * unit_region_size;

      // Get position of first contiguous region in output tensor
      const int output_region_start = m_concatenation_points[i] * unit_region_size;

      // Populate slice of output
      for (int region = 0; region < num_regions; ++region) {
        auto input_ptrs = input_d.get_locked_data();
        auto output_ptrs = output_d.get_data();
        for (auto& ptr : input_ptrs) {
          ptr += region * input_region_stride;
        }
        for (auto& ptr : output_ptrs) {
          ptr += output_region_start + region * output_region_stride;
        }
        input_region_d.locked_attach(input_ptrs,
                                     input_region_stride,
                                     this->m_mini_batch_size_per_gpu,
                                     input_d.get_leading_dim());
        output_region_d.attach(output_ptrs,
                               input_region_stride,
                               this->m_mini_batch_size_per_gpu,
                               output_d.get_leading_dim());
        output_region_d.copy(input_region_d);
      }

    }

  #endif // __LIB_CUDNN
  }

  void bp_compute_gpu() {
  #ifndef __LIB_CUDNN
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: concatenation_layer: cuDNN not detected";
    throw lbann_exception(err.str());
  #else

    // Gradient w.r.t. output
    const auto& gradient_wrt_output_d = m_prev_error_signals_d[0];

    // Get number of contiguous regions in a tensor slice of width 1
    const auto& output_dims = this->m_neuron_dims;
    const int num_regions = std::accumulate(output_dims.begin(),
                                            output_dims.begin() + m_concatenation_axis,
                                            1,
                                            std::multiplies<int>());
    const int unit_region_size = std::accumulate(output_dims.begin() + m_concatenation_axis + 1,
                                                 output_dims.end(),
                                                 1,
                                                 std::multiplies<int>());

    // Get stride between contiguous regions in input tensor slice
    const int output_slice_dim = output_dims[m_concatenation_axis];
    const int output_region_stride = output_slice_dim * unit_region_size;
    
    // Populate output tensors with concatenations of input tensor
    cudnn::matrix gradient_wrt_input_region_d(m_cudnn);
    cudnn::matrix gradient_wrt_output_region_d(m_cudnn);
    for (int i = 0; i < get_num_parents(); ++i) {
      auto& gradient_wrt_input_d = m_error_signals_d[i];

      // Get stride between contiguous regions in output tensor concatenations
      const int input_slice_dim = m_concatenation_points[i+1] - m_concatenation_points[i];
      const int input_region_stride = input_slice_dim * unit_region_size;

      // Get position of first contiguous region in output tensor
      const int output_region_start = m_concatenation_points[i] * unit_region_size;

      // Populate current output tensor
      for (int region = 0; region < num_regions; ++region) {
        auto gradient_wrt_input_ptrs = gradient_wrt_input_d.get_data();
        auto gradient_wrt_output_ptrs = gradient_wrt_output_d.get_locked_data();
        for (auto& ptr : gradient_wrt_input_ptrs) {
          ptr += region * input_region_stride;
        }
        for (auto& ptr : gradient_wrt_output_ptrs) {
          ptr += output_region_start + region * output_region_stride;
        }
        gradient_wrt_input_region_d.attach(gradient_wrt_input_ptrs,
                                           input_region_stride,
                                           this->m_mini_batch_size_per_gpu,
                                           gradient_wrt_input_d.get_leading_dim());
        gradient_wrt_output_region_d.locked_attach(gradient_wrt_output_ptrs,
                                                   input_region_stride,
                                                   this->m_mini_batch_size_per_gpu,
                                                   gradient_wrt_output_d.get_leading_dim());
        const int num_gpus = m_cudnn->get_num_gpus();
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
          CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(gpu)));
          CHECK_CUBLAS(cublas::geam(this->m_cudnn->get_cublas_handle(gpu),
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    gradient_wrt_input_region_d.get_height(),
                                    this->m_mini_batch_size_per_gpu,
                                    DataType(1),
                                    gradient_wrt_output_region_d.get_locked_data(gpu),
                                    gradient_wrt_output_region_d.get_leading_dim(),
                                    DataType(1),
                                    gradient_wrt_input_region_d.get_locked_data(gpu),
                                    gradient_wrt_input_region_d.get_leading_dim(),
                                    gradient_wrt_input_region_d.get_data(gpu),
                                    gradient_wrt_input_region_d.get_leading_dim()));
        }
      }

    }

  #endif // __LIB_CUDNN
  }

  std::vector<int> get_prev_neuron_dims(int parent_index = 0) const override {
    std::vector<int> prev_neuron_dims = m_prev_neuron_dims;
    prev_neuron_dims[m_concatenation_axis]
      = (m_concatenation_points[parent_index+1] - m_concatenation_points[parent_index]);
    return prev_neuron_dims;
  }

  int get_num_prev_neurons(int parent_index = 0) const override {
    const auto& prev_neuron_dims = get_prev_neuron_dims(parent_index);
    return std::accumulate(prev_neuron_dims.begin(),
                           prev_neuron_dims.end(),
                           1,
                           std::multiplies<int>());
  }

};

} // namespace lbann

#endif // LBANN_LAYER_CONCATENATION_HPP_INCLUDED
