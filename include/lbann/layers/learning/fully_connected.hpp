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
// lbann_layer_fully_connected .hpp .cpp - Dense, fully connected, layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
#define LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED

#include "lbann/layers/learning/learning.hpp"
#include "lbann/layers/activations/activation.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/models/model.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/fan_in_fan_out_initializers.hpp"
#ifdef __LIB_CUDNN
#include "lbann/layers/learning/fully_connected_cuda.hpp"
#include "lbann/utils/cublas_wrapper.hpp"
#endif // __LIB_CUDNN
#include <string>
#include <sstream>

namespace lbann {

enum class device {CPU, CUDA};

/** Fully-connected layer.
 *  This layer applies an affine transformation.
 */
template <data_layout T_layout>
class fully_connected_layer : public learning_layer {
 private:

  /** Scaling factor for bias term. 
   *  If the scaling factor is zero, bias is not applied.
   */
  DataType m_bias_scaling_factor;

  /** Linearity gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the linearity weights (i.e. its matrix weights).
   */
  AbsDistMat* m_linearity_gradient;
  /** Bias weights gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the bias weights.
   */
  AbsDistMat* m_bias_gradient;

#ifdef __LIB_CUDNN

  /** GPU memory for matrix weights gradient. */
  std::vector<DataType *> m_matrix_weights_gradient_d;
  /** GPU memory for bias weights gradient. */
  std::vector<DataType *> m_bias_weights_gradient_d;

  /** Bias tensor cuDNN descriptor. */
  cudnnTensorDescriptor_t m_bias_weights_desc;
  /** Activations matrix cuDNN descriptor*/
  cudnnTensorDescriptor_t m_activations_desc;

#endif // __LIB_CUNN

  void deallocate_matrices() {
    if (m_linearity_gradient != nullptr) delete m_linearity_gradient;
    if (m_bias_gradient != nullptr) delete m_bias_gradient;
    m_linearity_gradient = nullptr;
    m_bias_gradient = nullptr;
  }

 public:

  fully_connected_layer(lbann_comm *comm,
                        int num_neurons,  // TODO: accept a vector for neuron dims
                        weights* weight = nullptr,
                        bool has_bias = true,
                        cudnn::cudnn_manager *cudnn = nullptr)
    : learning_layer(comm),
      m_linearity_gradient(nullptr),
      m_bias_gradient(nullptr) {

    // Initialize neuron tensor dimensions
    this->m_num_neurons = num_neurons;
    this->m_num_neuron_dims = 1;
    this->m_neuron_dims.assign(1, this->m_num_neurons);

    // Initialize bias
    m_bias_scaling_factor = has_bias ? DataType(1) : DataType(0);

#ifdef __LIB_CUDNN
    if (cudnn && T_layout == data_layout::DATA_PARALLEL) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
#endif
  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    return std::string {} +
     " fully_connected; num_neurons: " 
     + std::to_string(this->m_num_neurons)
     + " has_bias: " + std::to_string(this->m_bias_scaling_factor)
     + " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  static void setup_gpu_activation_bias(std::vector<DataType*> &parameters,
                                        std::vector<DataType*> &activations,
                                        std::vector<DataType*> &bias,
                                        El::Int height, El::Int width) {
    activations = parameters;
    for (auto & parameter : parameters) {
      // point to the last column 
      bias.push_back(parameter + height * (width - 1));
    }
  }

  fully_connected_layer(const fully_connected_layer& other) :
    learning_layer(other),
    m_bias_scaling_factor(other.m_bias_scaling_factor) {
    
    // Deep matrix copies
    m_linearity_gradient = other.m_linearity_gradient;
    m_bias_gradient = other.m_bias_gradient;
    if (m_linearity_gradient != nullptr) {
      m_linearity_gradient = m_linearity_gradient->Copy();
    }
    if (m_bias_gradient != nullptr) {
      m_bias_gradient = m_bias_gradient->Copy();
    }

#ifdef __LIB_CUDNN
    if (m_cudnn != nullptr) {
      m_matrix_weights_gradient_d = m_cudnn->copy(other.m_matrix_weights_gradient_d,
                                                  other.m_matrix_weights_gradient->Height(),
                                                  other.m_matrix_weights_gradient->Width());
      m_bias_weights_gradient_d = m_cudnn->copy(other.m_bias_weights_gradient_d,
                                                other.m_bias_weights_gradient->Height(),
                                                other.m_bias_weights_gradient->Width());
      cudnn::copy_tensor_cudnn_desc(other.m_bias_weights_desc,
                                    m_bias_weights_desc);
      cudnn::copy_tensor_cudnn_desc(other.m_activations_desc,
                                    m_activations_desc);
    }
#endif // __LIB_CUDNN
    
  }

  fully_connected_layer& operator=(const fully_connected_layer& other) {
    learning_layer::operator=(other);
    m_bias_scaling_factor = other.m_bias_scaling_factor;

    // Deep matrix copies
    deallocate_matrices();
    m_linearity_gradient = other.m_linearity_gradient;
    m_bias_gradient = other.m_bias_gradient;
    if (m_linearity_gradient != nullptr) {
      m_linearity_gradient = m_linearity_gradient->Copy();
    }
    if (m_bias_gradient != nullptr) {
      m_bias_gradient = m_bias_gradient->Copy();
    }

  #ifdef __LIB_CUDNN
    if (m_cudnn != nullptr) {
      m_cudnn->deallocate_on_gpus(m_matrix_weights_gradient_d);
      m_cudnn->deallocate_on_gpus(m_bias_weights_gradient_d);
      m_matrix_weights_gradient_d = m_cudnn->copy(other.m_matrix_weights_gradient_d,
                                                  this->m_matrix_weights_gradient->Height(),
                                                  this->m_matrix_weights_gradient->Width());
      m_bias_weights_gradient_d = m_cudnn->copy(other.m_bias_weights_gradient_d,
                                                this->m_bias_weights_gradient->Height(),
                                                this->m_bias_weights_gradient->Width());
      cudnn::copy_tensor_cudnn_desc(other.m_bias_weights_desc,
                                    m_bias_weights_desc);
      cudnn::copy_tensor_cudnn_desc(other.m_activations_desc,
                                    m_activations_desc);
    }
  #endif // __LIB_CUDNN

    return *this;
  }

  ~fully_connected_layer() override {
    deallocate_matrices();

#ifdef __LIB_CUDNN
    if (m_cudnn != nullptr) {
      this->m_cudnn->deallocate_on_gpus(m_matrix_weights_gradient_d);
      this->m_cudnn->deallocate_on_gpus(m_bias_weights_gradient_d);
      if (m_bias_weights_desc != nullptr) {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_bias_weights_desc));
      }
      if (m_activations_desc != nullptr) {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_activations_desc));
      }
    }
#endif // __LIB_CUDNN
    
  }

  fully_connected_layer* copy() const override {
    return new fully_connected_layer(*this);
  }

  std::string get_type() const override { return "fully connected"; }

  data_layout get_data_layout() const override { return T_layout; }

  void setup_matrices(const El::Grid& grid) override {
    learning_layer::setup_matrices(grid);
    deallocate_matrices();
    switch (get_data_layout()) {
    case data_layout::MODEL_PARALLEL:
      m_linearity_gradient = new MCMRMat(grid);
      m_bias_gradient = new MCStarMat(grid);
      break;
    case data_layout::DATA_PARALLEL:
      m_linearity_gradient = new StarMat(grid);
      m_bias_gradient = new StarMat(grid);
      break;
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "invalid distributed matrix layout";
      throw lbann_exception(err.str());
    }
  }

  void setup_dims() override {
    // Store neuron tensor dimensions
    const int num_neurons = this->m_num_neurons;
    const int num_neuron_dims = this->m_num_neuron_dims;
    const std::vector<int> neuron_dims = this->m_neuron_dims;

    // Initialize previous neuron tensor dimensions
    learning_layer::setup_dims();

    // Initialize neuron tensor dimensions
    this->m_num_neurons = num_neurons;
    this->m_num_neuron_dims = num_neuron_dims;
    this->m_neuron_dims = neuron_dims;
  }

  void setup_data() override {
    learning_layer::setup_data();

    // Initialize default weights if none are provided
    if (this->m_weights.size() > 2) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "attempted to setup " << m_name << " with an invalid number of weights";
      throw lbann_exception(err.str());
    }
    this->m_weights.resize(2, nullptr);
    if (this->m_weights[0] == nullptr) {
      this->m_weights[0] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[0]->set_name(this->m_name + "_linearity_weights");
      this->m_weights[0]->set_initializer(new he_normal_initializer(this->m_comm));
      this->m_weights[0]->set_optimizer(m_model->create_optimizer());
      this->m_model->add_weights(this->m_weights[0]);
    }
    if (this->m_weights[1] == nullptr) {
      this->m_weights[1] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[1]->set_name(this->m_name + "_bias_weights");
      this->m_weights[1]->set_optimizer(m_model->create_optimizer());
      this->m_model->add_weights(this->m_weights[1]);
    }

    // Initialize Glorot or He weight initialization
    auto* cast_initializer
      = dynamic_cast<fan_in_fan_out_initializer*>(&this->m_weights[0]->get_initializer());
    if (cast_initializer != nullptr) {
      cast_initializer->set_fan_in(this->m_num_prev_neurons);
      cast_initializer->set_fan_out(this->m_num_neurons);
    }

    // Setup weights
    switch (get_data_layout()) {
    case data_layout::MODEL_PARALLEL:
      this->m_weights[0]->setup(this->m_num_neurons,
                                this->m_num_prev_neurons,
                                El::MC, El::MR);
      this->m_weights[1]->setup(this->m_num_neurons, 1,
                                El::MC, El::STAR);
      break;
    case data_layout::DATA_PARALLEL:
      this->m_weights[0]->setup(this->m_num_neurons,
                                this->m_num_prev_neurons,
                                El::STAR, El::STAR);
      this->m_weights[1]->setup(this->m_num_neurons, 1,
                                El::STAR, El::STAR);
      break;
    default:
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "attempted to setup " << m_name << " with an invalid data layout";
      throw lbann_exception(err.str());
    }

    // Setup weight gradients
    El::Zeros(*this->m_linearity_gradient,
              this->m_weights[0]->get_height(),
              this->m_weights[0]->get_width());
    El::Zeros(*this->m_bias_gradient,
              this->m_weights[1]->get_height(),
              this->m_weights[1]->get_width());

  }

  void setup_gpu() override {
    learning_layer::setup_gpu();
#ifndef __LIB_CUDNN
    throw lbann_exception("fully_connected_layer: CUDA not detected");
#else

    // Allocate GPU memory
    this->m_cudnn->allocate_on_gpus(this->m_matrix_weights_gradient_d,
                                    this->m_matrix_weights_gradient->Height(),
                                    this->m_matrix_weights_gradient->Width());
    if(m_bias_scaling_factor != DataType(0)) {
      this->m_cudnn->allocate_on_gpus(this->m_bias_weights_gradient_d,
                                      this->m_bias_weights_gradient->Height(),
                                      this->m_bias_weights_gradient->Width());
    }

    // CUDNN setup
    // NOTE: Setting tensor dimensions as (1, 1, X, Y), where X is the
    // mini batch size, and bias dimensions as (1, 1, 1, Y) does not
    // work. Calls to cudnnAddTensor return
    // CUDNN_STATUS_NOT_SUPPORTED. Setting X as the dimension of C
    // works, though they should be mathematically the same. 
    FORCE_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_bias_weights_desc));
    FORCE_CHECK_CUDNN(cudnnSetTensor4dDescriptor(m_bias_weights_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 cudnn::get_cudnn_data_type(),
                                                 1, 1, 1,
                                                 m_num_neurons));
    FORCE_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_activations_desc));
    FORCE_CHECK_CUDNN(cudnnSetTensor4dDescriptor(m_activations_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 cudnn::get_cudnn_data_type(),
                                                 1, m_mini_batch_size_per_gpu, 1,
                                                 m_num_neurons));

#endif // __LIB_CUDNN
  }

  void fp_compute() override {
    if(this->m_using_gpus) {
      fp_compute_cuda();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
      bp_compute_cuda();
    } else {
      bp_compute_cpu();
    }
  }

  void fp_compute_cpu() {

    // Matrices
    const auto& input = get_prev_activations();
    auto& output = get_activations();

    // Apply linearity
    const auto& linearity = m_weights[0]->get_values();
    if (linearity.DistSize() == 1) {
      El::Gemm(El::NORMAL, El::NORMAL,
               DataType(1), linearity.LockedMatrix(), input.LockedMatrix(),
               DataType(0), output.Matrix());
    } else {
      El::Gemm(El::NORMAL, El::NORMAL,
               DataType(1), linearity, input,
               DataType(0), output);
    }

    // Apply bias if needed
    if(m_bias_scaling_factor != DataType(0)) {
      const auto& local_bias = m_weights[1]->get_values().LockedMatrix();
      auto& local_output = output.Matrix();
      El::IndexDependentMap(local_output,
                            (std::function<DataType(El::Int,El::Int,const DataType&)>)
                            ([this,&local_bias](El::Int r, El::Int c,const DataType& z)
                             ->DataType {
                              return z + m_bias_scaling_factor * local_bias(r, 0);
                            }));
    }

  }

  void bp_compute_cpu() {

    // Effective mini-batch size
    const int mini_batch_size = this->m_model->get_current_mini_batch_size();

    // Matrices
    const auto& linearity = m_weights[0]->get_values();
    const auto& input = get_prev_activations();
    const auto& gradient_wrt_output = get_prev_error_signals();
    auto& gradient_wrt_input = get_error_signals();
    const auto& local_linearity = linearity.LockedMatrix();
    const auto& local_input = input.LockedMatrix();
    const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
    auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();

    // Compute gradient w.r.t. bias if needed
    optimizer* bias_optimizer = this->m_weights[1]->get_optimizer();
    if (m_bias_scaling_factor != DataType(0)
        && bias_optimizer != nullptr) {
      El::RowSum(local_gradient_wrt_output,
                 m_bias_gradient->Matrix());
      bias_optimizer->stage_gradient_for_accumulation(
        *m_bias_gradient,
        m_bias_scaling_factor / mini_batch_size);
    }

    // Compute gradient w.r.t. linearity if needed
    optimizer* linearity_optimizer = this->m_weights[0]->get_optimizer();
    if (linearity_optimizer != nullptr) {
      if (linearity.DistSize() == 1) {
        El::Gemm(El::NORMAL, El::TRANSPOSE,
                 DataType(1), local_gradient_wrt_output, local_input,
                 DataType(0), m_linearity_gradient->Matrix());
        linearity_optimizer->stage_gradient_for_accumulation(
          *m_linearity_gradient,
          DataType(1) / mini_batch_size);
      } else {
        El::Gemm(El::NORMAL, El::TRANSPOSE,
                 DataType(1), gradient_wrt_output, input,
                 DataType(0), *m_linearity_gradient);
        linearity_optimizer->add_to_gradient(
          *m_linearity_gradient,
          DataType(1) / mini_batch_size);
      }
    }

    // Compute gradient w.r.t. input
    if (linearity.DistSize() == 1) {
      El::Gemm(El::TRANSPOSE, El::NORMAL,
               DataType(1), local_linearity, local_gradient_wrt_output,
               DataType(1), local_gradient_wrt_input);
    } else {
      El::Gemm(El::TRANSPOSE, El::NORMAL,
               DataType(1), linearity, gradient_wrt_output,
               DataType(1), gradient_wrt_input);
    }

  }

  void fp_compute_cuda() {
#ifndef __LIB_CUDNN
    throw lbann_exception("fully_connected: CUDA not detected");
#else
    // Apply weight matrix
    fp_compute_weights<device::CUDA>();

    // Apply bias if needed
    if(m_bias_scaling_factor != DataType(0)) {
      std::vector<DataType*> bias_weights_d = m_weights[1]->get_values_gpu();
      const int num_gpus = this->m_cudnn->get_num_gpus();
      for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
#if 1
        const DataType one = 1;        
        CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                   this->m_cudnn->get_stream(i)));
        FORCE_CHECK_CUDNN(cudnnAddTensor(this->m_cudnn->get_handle(i),
                                         &m_bias_scaling_factor,
                                         m_bias_weights_desc,
                                         bias_weights_d[i],
                                         &one,
                                         m_activations_desc,
                                         m_activations_d[i]));
#else
        fully_connected_cuda::add_tensor(m_bias_scaling_factor,
                                         bias_weights_d[i],
                                         m_bias_weights_v->Height(), 1,
                                         DataType(1), m_activations_d[i],
                                         this->m_activations_v->Height(),
                                         this->m_mini_batch_size_per_gpu);
#endif
      }
    }
#ifdef LBANN_DEBUG
    this->m_cudnn->check_error();
#endif
#endif
  }

  void bp_compute_cuda() {
#ifndef __LIB_CUDNN
    throw lbann_exception("fully_connected: CUDA not detected");
#else
    // Compute the error signal and gradients.
    bp_compute_weights<device::CUDA>();

    // Compute bias update if needed
    optimizer* bias_optimizer = m_weights[1]->get_optimizer();
    if(bias_optimizer != nullptr && m_bias_scaling_factor != DataType(0)) {
      fully_connected_cuda::row_sum(*this->m_cudnn,
                                    m_prev_error_signal_dv,
                                    m_prev_error_signal_v->Height(),
                                    m_mini_batch_size_per_gpu,
                                    DataType(1),
                                    m_bias_weights_gradient_d);
      bias_optimizer->stage_gradient_for_accumulation_gpu(
        m_bias_weights_gradient_d,
        m_bias_scaling_factor / this->m_model->get_current_mini_batch_size());
    }

#ifdef LBANN_DEBUG
    this->m_cudnn->check_error();
#endif
#endif // __LIB_CUDNN
  }

};

#ifdef __LIB_CUDNN
template<> template<> inline void
fully_connected_layer<data_layout::DATA_PARALLEL>::fp_compute_weights<device::CUDA>() {
  std::vector<DataType*> matrix_weights_d = m_weights[0]->get_values_gpu();
  const int num_gpus = this->m_cudnn->get_num_gpus();
  for(int i=0; i<num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
    CHECK_CUBLAS(cublas::gemm(this->m_cudnn->get_cublas_handle(i),
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              m_weights[0]->get_height(),
                              m_mini_batch_size_per_gpu,
                              m_weights[0]->get_width(),
                              DataType(1),
                              matrix_weights_d[i],
                              m_weights[0]->get_height(),
                              this->m_prev_activations_dv[i],
                              this->m_prev_activations_v->Height(),
                              DataType(0),
                              this->m_activations_d[i],
                              this->m_activations_v->Height()));
  }
}
#endif // __LIB_CUDNN

#ifdef __LIB_CUDNN
template<> template<> inline void
fully_connected_layer<data_layout::DATA_PARALLEL>::bp_compute_weights<device::CUDA>() {

  // Compute objective function gradient w.r.t. input
  std::vector<DataType*> matrix_weights_d = m_weights[0]->get_values_gpu();
  const int num_gpus = this->m_cudnn->get_num_gpus();
  for(int i=0; i<num_gpus; ++i) {
    CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
    CHECK_CUBLAS(cublas::gemm(this->m_cudnn->get_cublas_handle(i),
                              CUBLAS_OP_T,
                              CUBLAS_OP_N,
                              m_weights[0]->get_width(),
                              m_mini_batch_size_per_gpu,
                              m_weights[0]->get_height(),
                              DataType(1),
                              matrix_weights_d[i],
                              m_weights[0]->get_height(),
                              this->m_prev_error_signal_dv[i],
                              this->m_prev_error_signal_v->Height(),
                              DataType(0),
                              this->m_error_signal_d[i],
                              this->m_error_signal_v->Height()));
  }

  // Compute objective function gradient w.r.t. matrix weights
  optimizer* matrix_optimizer = m_weights[0]->get_optimizer();
  if (matrix_optimizer != nullptr) {
    for(int i=0; i<num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUBLAS(cublas::gemm(this->m_cudnn->get_cublas_handle(i),
                                CUBLAS_OP_N,
                                CUBLAS_OP_T,
                                m_weights[0]->get_height(),
                                m_weights[0]->get_width(),
                                m_mini_batch_size_per_gpu,
                                DataType(1),
                                this->m_prev_error_signal_dv[i],
                                this->m_prev_error_signal_v->Height(),
                                this->m_prev_activations_dv[i],
                                this->m_prev_activations_v->Height(),
                                DataType(0),
                                this->m_matrix_weights_gradient_d[i],
                                this->m_matrix_weights_gradient->Height()));
    }
    matrix_optimizer->stage_gradient_for_accumulation_gpu(
      m_matrix_weights_gradient_d,
      DataType(1) / this->m_model->get_current_mini_batch_size());
  }
  
}
#endif // __LIB_CUDNN

}  // namespace lbann

#endif  // LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
