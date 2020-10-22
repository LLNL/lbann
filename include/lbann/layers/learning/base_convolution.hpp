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

#ifndef LBANN_LAYERS_LEARNING_BASE_CONVOLUTION_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_BASE_CONVOLUTION_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/memory.hpp"

#include <vector>

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
template <typename TensorDataType, El::Device Device>
class base_convolution_adapter: public data_type_distconv_adapter<TensorDataType> {
 public:
  using TensorDevType = typename data_type_distconv_adapter<TensorDataType>::TensorDevType;

  base_convolution_adapter(Layer& layer): data_type_distconv_adapter<TensorDataType>(layer) {}
  virtual ~base_convolution_adapter() = default;

  void setup_fp_tensors() override;
  void setup_bp_tensors() override;
  void setup_layer(size_t workspace_capacity) override;

  void fp_compute_convolution();
  void fp_apply_bias();

  void bp_compute_convolution_data();
  void bp_compute_convolution_filter();

  std::unique_ptr<dc::Convolution<TensorDataType>> m_conv;
  std::unique_ptr<TensorDevType> m_kernel;
  std::unique_ptr<TensorDevType> m_bias;
  std::unique_ptr<TensorDevType> m_kernel_gradient;
  std::unique_ptr<TensorDevType> m_bias_gradient;

  std::string m_fwd_algo;
  std::string m_bwd_data_algo;
  std::string m_bwd_filter_algo;
};
#endif // LBANN_HAS_DISTCONV

/** @brief Computation kernels for convolution and deconvolution layers.
 */
template <typename TensorDataType, El::Device Device>
class base_convolution_layer : public data_type_layer<TensorDataType> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  /** @brief The concrete optimizer type used by this object. */
  using OptimizerType = data_type_optimizer<TensorDataType>;

  template <El::Device D>
  using DMatDT = El::Matrix<TensorDataType, D>;

#ifdef LBANN_HAS_CUDNN
  using ScalingType = cudnn::ScalingParamType<TensorDataType>;
#else
  using ScalingType = TensorDataType;
#endif // LBANN_HAS_CUDNN

  ///@}

protected:

  int m_output_channels;
  /** @brief Spatial dimensions for convolution kernel.
   *  @details Excludes number of input and output channels.
   */
  std::vector<int> m_conv_dims;
  /** Convolution padding. */
  std::vector<int> m_pads;
  /** Convolution strides. */
  std::vector<int> m_strides;
  /** Convolution dilations. */
  std::vector<int> m_dilations;
  /** Convolution groups.
   *  The channels are split into this many independent groups when performing
   *  convolution. The default convolution operation has one group, and a
   *  depthwise convolution has as many groups as there are input channels.
   */
  int m_groups;

  /** Scaling factor for bias term.
   *  If the scaling factor is zero, bias is not applied.
   */
  ScalingType m_bias_scaling_factor;

#ifdef LBANN_HAS_CUDNN

  /** @brief Math type to use inside cuDNN.
   *  @details Must be cached since it isn't used until setup.
   */
  cudnnMathType_t m_convolution_math_type =
    cudnn::get_default_convolution_math_type();
  /** Convolution kernel cuDNN descriptor. */
  cudnn::FilterDescriptor m_kernel_cudnn_desc;
  /** Convolution cuDNN descriptor. */
  cudnn::ConvolutionDescriptor m_convolution_cudnn_desc;
  /** Bias tensor cuDNN descriptor. */
  cudnn::TensorDescriptor m_bias_cudnn_desc;
  /** Tensor cuDNN descriptors. */
  cudnn::data_parallel_layer_tensor_manager<TensorDataType> m_tensors_cudnn_desc;
  /** Forward algorithm cache (mini-batch size -> algo). */
  std::unordered_map<int, cudnnConvolutionFwdAlgo_t> m_fwd_cudnn_algos;
  /** Backward data algorithm cache (mini-batch size -> algo). */
  std::unordered_map<int, cudnnConvolutionBwdDataAlgo_t> m_bwd_data_cudnn_algos;
  /** Backward filter algorithm cache (mini-batch size -> algo). */
  std::unordered_map<int, cudnnConvolutionBwdFilterAlgo_t> m_bwd_filter_cudnn_algos;

#endif // LBANN_HAS_CUDNN

public:
  /** @todo Remove num_data_dims from arg list */
  base_convolution_layer(lbann_comm* comm,
                         int num_data_dims,
                         int output_channels,
                         std::vector<int> conv_dims,
                         std::vector<int> pads,
                         std::vector<int> strides,
                         std::vector<int> dilations,
                         int groups,
                         bool has_bias);

  base_convolution_layer(const base_convolution_layer& other);

  base_convolution_layer& operator=(const base_convolution_layer& other);

  ~base_convolution_layer();

#ifdef LBANN_HAS_CUDNN
  void set_cudnn_math_mode(cudnnMathType_t math_type) noexcept;
#endif // LBANN_HAS_CUDNN

  description get_description() const override;
  void setup_dims(DataReaderMetaData& dr_metadata) override;

  /** @brief Setup layer data.
   *  The kernel weights are setup in the convolution and
   *  deconvolution classes. */
  void setup_data(size_t max_mini_batch_size) override;

  /** @brief Initialize GPU objects */
  void setup_gpu() override;

protected:

  /** Dimensions of convolution kernel. */
  virtual std::vector<int> get_kernel_dims() const = 0;

  /** Convolution with cuDNN. */
  void apply_convolution_cudnn(bool during_forward_prop);

  /** Transposed convolution with cuDNN. */
  void apply_transposed_convolution_cudnn(bool during_forward_prop);

  void apply_bias_cudnn();
  void compute_gradients_cudnn(bool using_transposed_convolution);

  /** Convolution with im2col GEMM algorithm. */
  void apply_convolution_im2col(bool during_forward_prop);

  /** Transposed convolution with im2col GEMM algorithm. */
  void apply_transposed_convolution_im2col(bool during_forward_prop);

  void apply_bias_cpu();

  void compute_gradients_im2col(bool using_transposed_convolution);

private:

#ifdef LBANN_HAS_CUDNN

  /** Get the cuDNN algorithm to use for forward prop. */
  cudnnConvolutionFwdAlgo_t get_forward_algo_cudnn(
    const int local_mini_batch_size,
    const cudnn::TensorDescriptor& input_desc,
    const TensorDataType* input,
    const cudnn::FilterDescriptor& kernel_desc,
    const TensorDataType* kernel,
    const cudnn::ConvolutionDescriptor& conv_desc,
    const cudnn::TensorDescriptor& output_desc,
    TensorDataType* output,
    size_t ws_size,
    TensorDataType* ws);

  /** Get the cuDNN algorithm to use for backward-data. */
  cudnnConvolutionBwdDataAlgo_t get_backward_data_algo_cudnn(
    const int local_mini_batch_size,
    const cudnn::FilterDescriptor& kernel_desc,
    const TensorDataType* kernel,
    const cudnn::TensorDescriptor& prev_error_signal_desc,
    const TensorDataType* prev_error_signal,
    const cudnn::ConvolutionDescriptor& conv_desc,
    const cudnn::TensorDescriptor& error_signal_desc,
    TensorDataType* error_signal,
    size_t ws_size,
    TensorDataType* ws);

  /**
   * Get the cuDNN algorithm to use for backward-filter.
   * Buffer space for kernel_gradient is allocated via temporary workspace.
   */
  cudnnConvolutionBwdFilterAlgo_t get_backward_filter_algo_cudnn(
    const int local_mini_batch_size,
    const cudnn::TensorDescriptor& input_desc,
    const TensorDataType* input,
    const cudnn::TensorDescriptor& prev_error_signal_desc,
    const TensorDataType* prev_error_signal,
    const cudnn::ConvolutionDescriptor& conv_desc,
    const cudnn::FilterDescriptor& kernel_gradient_desc,
    size_t ws_size,
    TensorDataType* ws);
#endif // LBANN_HAS_CUDNN

#ifdef LBANN_HAS_DISTCONV
  friend class base_convolution_adapter<TensorDataType, Device>;
 protected:
  using BaseConvAdapterType = base_convolution_adapter<TensorDataType, Device>;
  void setup_distconv_adapter(const DataReaderMetaData& dr_metadata) override;
  BaseConvAdapterType& get_distconv_adapter() override;
  const BaseConvAdapterType& get_distconv_adapter() const override;
#endif // LBANN_HAS_DISTCONV
};

} // namespace lbann
#endif // LBANN_LAYERS_LEARNING_BASE_CONVOLUTION_HPP_INCLUDED
