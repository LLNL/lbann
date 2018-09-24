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

#include "math.h"
#include "lbann/layers/activations/sigmoid.hpp"
#include <limits>

namespace lbann {
namespace {

// Sigmoid function
#if __CUDA_ARCH__ >= 530
inline __device__ __half sigmoid(__half x) {
  static_cast<void>(static_cast<__half (*)(__half)>(sigmoid)); // Suppress "unused function" warning
  return __hdiv(__float2half(1.f),
                __hadd(__float2half(1.f), hexp(__hneg(x))));
}
#endif // __CUDA_ARCH__ >= 530
inline __device__ float sigmoid(float x) {
  static_cast<void>(static_cast<float (*)(float)>(sigmoid)); // Suppress "unused function" warning
  return 1 / (1.0f + expf(-x));
}
inline __device__ double sigmoid(double x) {
  static_cast<void>(static_cast<double (*)(double)>(sigmoid)); // Suppress "unused function" warning
  return 1 / (1.0 + exp(-x));
}

// Machine epsilon
#ifdef __CUDACC_RELAXED_CONSTEXPR__
template <typename T>
inline __device__ T epsilon() {
  return std::numeric_limits<T>::epsilon();
}
#else // __CUDACC_RELAXED_CONSTEXPR__
template <typename T>
inline __device__ T epsilon();
#if __CUDA_ARCH__ >= 530
template <>
inline __device__ __half epsilon<__half>() {
  static_cast<void>(static_cast<__half (*)()>(epsilon<__half>)); // Suppress "unused function" warning
  return __float2half(0.0009765625f);
}
#endif // __CUDA_ARCH__ >= 530
template <>
inline __device__ float epsilon<float>()   {
  static_cast<void>(static_cast<float (*)()>(epsilon<float>)); // Suppress "unused function" warning
  return FLT_EPSILON;
}
template <>
inline __device__ double epsilon<double>() {
  static_cast<void>(static_cast<double (*)()>(epsilon<double>)); // Suppress "unused function" warning
  return DBL_EPSILON;
}
#endif // __CUDACC_RELAXED_CONSTEXPR__
  
/** Entry-wise operator. */
struct op {
  inline __device__ DataType operator()(DataType x) const {
    const DataType y = sigmoid(x);
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    const DataType eps = epsilon<DataType>();
    if (y <= eps) { return eps; }
    else if (y >= DataType(1) - eps) { return DataType(1) - eps; }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return y;
  }
};
  
/** Entry-wise operator for backprop.
 *  If the forward propagation step computes \f$ y = f(x) \f$, the
 *  backward propagation step computes
 *  \f$ \frac{dL}{dx} = \frac{dL}{dy} f'(x) \f$.
 */
struct op_backprop {
  inline __device__  DataType operator()(DataType x, DataType dy) const {
    const auto& y = op()(x);
#ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    const DataType eps = epsilon<DataType>();
    if (y <= eps || y >= DataType(1) - eps) { return DataType(0); }
#endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return dy * y * (DataType(1) - y);
  }
};
  
} // namespace

// Template instantiation
template <>
void sigmoid_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::fp_compute() {
  cuda::apply_entrywise_unary_operator<op>(get_prev_activations(),
                                           get_activations());
}
template <>
void sigmoid_layer<data_layout::MODEL_PARALLEL, El::Device::GPU>::bp_compute() {
  cuda::apply_entrywise_binary_operator<op_backprop>(get_prev_activations(),
                                                     get_prev_error_signals(),
                                                     get_error_signals());
}
template <>
void sigmoid_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
  cuda::apply_entrywise_unary_operator<op>(get_prev_activations(),
                                           get_activations());
}
template <>
void sigmoid_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::bp_compute() {
  cuda::apply_entrywise_binary_operator<op_backprop>(get_prev_activations(),
                                                     get_prev_error_signals(),
                                                     get_error_signals());
}

} // namespace lbann
