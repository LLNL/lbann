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

#ifndef LBANN_LAYER_HADAMARD_HPP_INCLUDED
#define LBANN_LAYER_HADAMARD_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

#ifdef LBANN_HAS_CUDNN
namespace hadamard_cuda {
void fp(cudnn::cudnn_manager& cudnn,
        int height,
        int width_per_gpu,
        const std::vector<std::vector<lbann::DataType*>>& inputs,
        const std::vector<int>& input_leading_dims,
        std::vector<lbann::DataType*>& output,
        int output_leading_dim);
void bp(cudnn::cudnn_manager& cudnn,
        int height,
        int width_per_gpu,
        const std::vector<std::vector<lbann::DataType*>>& inputs,
        const std::vector<int>& input_leading_dims,
        const std::vector<lbann::DataType*>& gradient_wrt_output,
        int gradient_wrt_output_leading_dim,
        std::vector<std::vector<lbann::DataType*>>& gradient_wrt_inputs,
        const std::vector<int>& gradient_wrt_input_leading_dims);
} // namespace hadamard_cuda
#endif // LBANN_HAS_CUDNN

/** Hadamard layer.
 *  This layer computes the entrywise product of the input tensors.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class hadamard_layer : public transform_layer {
 public:

  hadamard_layer(lbann_comm *comm,
                 cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm) {

    // Hadamard layer has no limit on parents
    m_expected_num_parent_layers = -1;

  #ifdef LBANN_HAS_CUDNN
    // Initialize GPU memory if using GPU
    if (cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // LBANN_HAS_CUDNN

  }

  hadamard_layer* copy() const override { return new hadamard_layer(*this); }
  std::string get_type() const override { return "Hadamard"; }
  data_layout get_data_layout() const override { return T_layout; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
    s << " Hadamard; parents: ";
    for (size_t i=0; i<this->m_parent_layers.size(); i++) {
      s << this->m_parent_layers[i]->get_name() << " " << this->m_parent_layers[i]->get_type() << " ";
    }
    s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    return s.str();
  }

  protected:

  void setup_pointers() override {
    transform_layer::setup_pointers();
    std::stringstream err;
    if (get_num_parents() <= 0) {
      err << __FILE__ << " " << __LINE__ << " :: hadamard_layer: "
          << "Hadamard layer has no parents";
      throw lbann_exception(err.str());
    }
  }

  void fp_compute() override {
    if (this->m_using_gpus) {
      fp_compute_gpu();
    } else { 
      fp_compute_cpu();
    }
  }

  void bp_compute() override {
    if (this->m_using_gpus) {
      bp_compute_gpu();
    } else { 
      bp_compute_cpu();
    }
  }

 private:

  void fp_compute_cpu() {

    // Return quickly if layer has fewer than two parents
    const int num_parents = get_num_parents();
    if (num_parents == 0) {
      El::Fill(get_activations(), DataType(1));
      return;
    }
    if (num_parents == 1) {
      El::Copy(get_prev_activations(), get_activations());
      return;
    }

    // Get local matrices
    auto& local_input0 = get_local_prev_activations(0);
    std::vector<const Mat*> local_inputs;
    for (int i = 1; i < num_parents; ++i) {
      local_inputs.push_back(&get_local_prev_activations(i));
    }
    auto& local_output = get_local_activations();

    // Compute entrywise product
    const int local_height = local_output.Height();
    const int local_width = local_output.Width();
    #pragma omp parallel for collapse(2)
    for (int col = 0; col < local_width; ++col) {
      for (int row = 0; row < local_height; ++row) {
        DataType y = local_input0(row, col);
        for (const auto& local_input : local_inputs) {
          y *= (*local_input)(row, col);
        }
        local_output(row, col) = y;
      }
    }
    
  }  

  void bp_compute_cpu() {

    // Return quickly if layer has fewer than two parents
    const int num_parents = get_num_parents();
    if (num_parents == 0) { return; }
    if (num_parents == 1) {
      El::Axpy(DataType(1), get_prev_error_signals(), get_error_signals());
      return;
    }

    // Get local matrices
    std::vector<const Mat*> local_inputs;
    for (const auto& input : this->m_prev_activations) {
      local_inputs.push_back(&input->LockedMatrix());
    }
    auto& local_gradient_wrt_output = get_local_prev_error_signals();
    std::vector<Mat*> local_gradient_wrt_inputs;
    for (auto& gradient_wrt_input : this->m_error_signals) {
      local_gradient_wrt_inputs.push_back(&gradient_wrt_input->Matrix());
    }

    // Compute derivative of entrywise product
    const int local_height = local_gradient_wrt_output.Height();
    const int local_width = local_gradient_wrt_output.Width();
    #pragma omp parallel for collapse(2)
    for (int col = 0; col < local_width; ++col) {
      for (int row = 0; row < local_height; ++row) {
        const DataType dy = local_gradient_wrt_output(row, col);
        for (int parent = 0; parent < num_parents; ++parent) {
          DataType dx = dy;
          for (int i = 0; i < num_parents; ++i) {
            if (i != parent) {
              dx *= (*local_inputs[i])(row, col);
            }
          }
          (*local_gradient_wrt_inputs[parent])(row, col) += dx;
        }
      }
    }

  }

  void fp_compute_gpu() {
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else
    std::vector<std::vector<lbann::DataType*>> inputs;
    std::vector<int> input_leading_dims;
    for (auto&& input : m_prev_activations_d) {
      inputs.push_back(input.get_locked_data());
      input_leading_dims.push_back(input.get_leading_dim());
    }
    hadamard_cuda::fp(*m_cudnn,
                      get_num_neurons(),
                      m_mini_batch_size_per_gpu,
                      inputs,
                      input_leading_dims,
                      m_activations_d[0].get_data(),
                      m_activations_d[0].get_leading_dim());
  #endif // LBANN_HAS_CUDNN
  }

  void bp_compute_gpu() {
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else
    std::vector<std::vector<lbann::DataType*>> inputs;
    std::vector<int> input_leading_dims;
    std::vector<std::vector<lbann::DataType*>> gradient_wrt_inputs;
    std::vector<int> gradient_wrt_input_leading_dims;
    for (auto&& input : m_prev_activations_d) {
      inputs.push_back(input.get_locked_data());
      input_leading_dims.push_back(input.get_leading_dim());
    }
    for (auto&& gradient_wrt_input : m_error_signals_d) {
      gradient_wrt_inputs.push_back(gradient_wrt_input.get_data());
      gradient_wrt_input_leading_dims.push_back(gradient_wrt_input.get_leading_dim());
    }
    hadamard_cuda::bp(*m_cudnn,
                      get_num_neurons(),
                      m_mini_batch_size_per_gpu,
                      inputs,
                      input_leading_dims,
                      m_prev_error_signals_d[0].get_locked_data(),
                      m_prev_error_signals_d[0].get_leading_dim(),
                      gradient_wrt_inputs,
                      gradient_wrt_input_leading_dims);
  #endif // LBANN_HAS_CUDNN
  }

};

} // namespace lbann

#endif // LBANN_LAYER_HADAMARD_HPP_INCLUDED
