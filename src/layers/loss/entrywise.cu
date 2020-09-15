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

#define LBANN_ENTRYWISE_LAYER_INSTANTIATE
#include "lbann/layers/loss/entrywise.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

/** CUDA kernel to apply an binary backprop operator. */
template <template <typename> class BinaryBackPropOperator,
          typename TensorDataType>
__global__
void binary_backprop_operator_kernel(El::Int height, El::Int width,
                                     const TensorDataType* __restrict__ x1,
                                     El::Int x1_ldim,
                                     const TensorDataType* __restrict__ x2,
                                     El::Int x2_ldim,
                                     const TensorDataType* __restrict__ dy,
                                     El::Int dy_ldim,
                                     TensorDataType* __restrict__ dx1,
                                     El::Int dx1_ldim,
                                     TensorDataType* __restrict__ dx2,
                                     El::Int dx2_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  BinaryBackPropOperator<TensorDataType> op;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    op(x1[row + col * x1_ldim],
       x2[row + col * x2_ldim],
       dy[row + col * dy_ldim],
       dx1[row + col * dx1_ldim],
       dx2[row + col * dx2_ldim]);
  }
}

/** Apply a binary backprop operator to CPU data.
 *  The input and output data must be on CPU and must have the same
 *  dimensions. Given a binary function \f$ y = f(x_1,x_2) \f$, the
 *  corresponding BinaryBackPropOperator is a 5-ary function with the
 *  arguments \f$ x_1 \f$, \f$ x_2 \f$, \f$ dL/dy \f$, \f$ dL/dx_1\f$,
 *  \f$ dL/dx_2 \f$. The last two arguments should be overwritten when
 *  the BinaryBackPropOperator is called.
 */
template <template <typename> class Op, typename TensorDataType>
void apply_binary_backprop_operator(
  const El::AbstractMatrix<TensorDataType>& x1,
  const El::AbstractMatrix<TensorDataType>& x2,
  const El::AbstractMatrix<TensorDataType>& dy,
  El::AbstractMatrix<TensorDataType>& dx1,
  El::AbstractMatrix<TensorDataType>& dx2) {

  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  const El::Int height = x1.Height();
  const El::Int width = x1.Width();
  const El::Int block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  //Launch CUDA kernel
  if (grid_dim > 0) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(dx2),
                                       gpu::get_sync_info(dx1),
                                       gpu::get_sync_info(dy),
                                       gpu::get_sync_info(x2),
                                       gpu::get_sync_info(x1));
    hydrogen::gpu::LaunchKernel(
      binary_backprop_operator_kernel<Op, TensorDataType>,
      grid_dim, block_dim, 0, multisync,
      height, width,
      x1.LockedBuffer(), x1.LDim(),
      x2.LockedBuffer(), x2.LDim(),
      dy.LockedBuffer(), dy.LDim(),
      dx1.Buffer(), dx1.LDim(),
      dx2.Buffer(), dx2.LDim());
  }

}

// =========================================================
// Operator objects for entry-wise binary layers
// =========================================================
// Note: Binary operator corresponds to forward prop step
// (\f$ y = f(x_1,x_2) \f$) and 5-ary operator corresponds
// to back prop step
// (\f$ \frac{dL}{dx_i} = \frac{dL}{dy} \frac{df}{dx_i}(x_1,x_2) \f$).

/** Binary cross entropy operator. */
template <typename TensorDataType>
struct binary_cross_entropy_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x1,
                                        const TensorDataType& x2) const {
    const TensorDataType zero = 0.;
    const TensorDataType one = 1.;
    TensorDataType y = zero;
    if (x2 > zero) { y += -x2 * cuda::log(x1); }
    if (x2 < one)  { y += -(one-x2) * cuda::log(one-x1); }
    return y;
  }
  inline __device__ void operator()(const TensorDataType& x1,
                                    const TensorDataType& x2,
                                    const TensorDataType& dy,
                                    TensorDataType& dx1,
                                    TensorDataType& dx2) const {
    const TensorDataType zero = 0.;
    const TensorDataType one = 1.;
    dx1 = zero;
    dx2 = zero;
    if (dy == zero) { return; }
    if (x2 > zero) {
      dx1 += -x2 / x1 * dy;
      dx2 += -cuda::log(x1) * dy;
    }
    if (x2 < one)  {
      dx1 += (one-x2) / (one-x1) * dy;
      dx2 += cuda::log(one-x1) * dy;
    }
  }
};

/** Sigmoid binary cross entropy operator.
 *  Equivalent to applying a sigmoid function to the first operand and
 *  then computing the binary cross entropy. Numerically stable
 *  implementation is taken from
 *  https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits.
 */
template <typename TensorDataType>
struct sigmoid_binary_cross_entropy_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x1,
                                        const TensorDataType& x2) const {
    const TensorDataType zero = 0.;
    const TensorDataType one = 1.;
    const auto& z = cuda::max(zero, cuda::min(x2, one));
    if (x1 > zero) {
      return (one - z) * x1 + cuda::log1p(cuda::exp(-x1));
    } else {
      return - x1 * z + cuda::log1p(cuda::exp(x1));
    }
  }
  inline __device__ void operator()(const TensorDataType& x1,
                                    const TensorDataType& x2,
                                    const TensorDataType& dy,
                                    TensorDataType& dx1,
                                    TensorDataType& dx2) const {
    const TensorDataType zero = 0.;
    const TensorDataType one = 1.;
    const auto& z = cuda::max(zero, cuda::min(x2, one));
    if (x1 > zero) {
      dx1 = -z + one / (one + cuda::exp(-x1));
    } else {
      dx1 = one - z - one / (one + cuda::exp(x1));
    }
    dx1 *= dy;
    dx2 = (x2 == z) ? -x1 * dy : zero;
  }
};

/** Boolean accuracy operator. */
template <typename TensorDataType>
struct boolean_accuracy_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x1,
                                        const TensorDataType& x2) const {
    const auto& b1 = x1 >= TensorDataType(0.5);
    const auto& b2 = x2 >= TensorDataType(0.5);
    return b1 == b2 ? TensorDataType(1.0) : TensorDataType(0.0);
  }
  inline __device__ void operator()(const TensorDataType& x1,
                                    const TensorDataType& x2,
                                    const TensorDataType& dy,
                                    TensorDataType& dx1,
                                    TensorDataType& dx2) const {
    dx1 = TensorDataType(0.0);
    dx2 = TensorDataType(0.0);
  }
};

/** Boolean false negative operator. */
template <typename TensorDataType>
struct boolean_false_negative_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x1,
                                        const TensorDataType& x2) const {
    const auto& b1 = x1 >= TensorDataType(0.5);
    const auto& b2 = x2 >= TensorDataType(0.5);
    return (!b1 && b2) ? TensorDataType(1.0) : TensorDataType(0.0);
  }
  inline __device__ void operator()(const TensorDataType& x1,
                                    const TensorDataType& x2,
                                    const TensorDataType& dy,
                                    TensorDataType& dx1,
                                    TensorDataType& dx2) const {
    dx1 = TensorDataType(0.0);
    dx2 = TensorDataType(0.0);
  }
};

/** Boolean false positive operator. */
template <typename TensorDataType>
struct boolean_false_positive_op {
  inline __device__ TensorDataType operator()(const TensorDataType& x1,
                                        const TensorDataType& x2) const {
    const auto& b1 = x1 >= TensorDataType(0.5);
    const auto& b2 = x2 >= TensorDataType(0.5);
    return (b1 && !b2) ? TensorDataType(1.0) : TensorDataType(0.0);
  }
  inline __device__ void operator()(const TensorDataType& x1,
                                    const TensorDataType& x2,
                                    const TensorDataType& dy,
                                    TensorDataType& dx1,
                                    TensorDataType& dx2) const {
    dx1 = TensorDataType(0.0);
    dx2 = TensorDataType(0.0);
  }
};

} // namespace

// Template instantiation
#define DEFINE_COMPUTE_OPS(layer, op)                                   \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  void layer<TensorDataType, Layout, Device>::fp_compute() {            \
    cuda::apply_entrywise_binary_operator<op>(                          \
      this->get_prev_activations(0),                                    \
      this->get_prev_activations(1),                                    \
      this->get_activations());                                         \
  }                                                                     \
  template <typename TensorDataType, data_layout Layout, El::Device Device> \
  void layer<TensorDataType, Layout, Device>::bp_compute() {            \
    apply_binary_backprop_operator<op>(                                 \
      this->get_local_prev_activations(0),                              \
      this->get_local_prev_activations(1),                              \
      this->get_local_prev_error_signals(),                             \
      this->get_local_error_signals(0),                                 \
      this->get_local_error_signals(1));                                \
  }                                                                     \

DEFINE_COMPUTE_OPS(binary_cross_entropy_layer, binary_cross_entropy_op)
DEFINE_COMPUTE_OPS(sigmoid_binary_cross_entropy_layer, sigmoid_binary_cross_entropy_op)
DEFINE_COMPUTE_OPS(boolean_accuracy_layer, boolean_accuracy_op)
DEFINE_COMPUTE_OPS(boolean_false_negative_layer, boolean_false_negative_op)
DEFINE_COMPUTE_OPS(boolean_false_positive_layer, boolean_false_positive_op)

#define PROTO(T) \
  BINARY_ETI_INST_MACRO_DEV_DT(binary_cross_entropy_layer, T, El::Device::GPU); \
  BINARY_ETI_INST_MACRO_DEV_DT(sigmoid_binary_cross_entropy_layer, T, El::Device::GPU); \
  BINARY_ETI_INST_MACRO_DEV_DT(boolean_accuracy_layer, T, El::Device::GPU); \
  BINARY_ETI_INST_MACRO_DEV_DT(boolean_false_negative_layer, T, El::Device::GPU); \
  BINARY_ETI_INST_MACRO_DEV_DT(boolean_false_positive_layer, T, El::Device::GPU)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
