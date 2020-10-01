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

#include "lbann/proto/factories.hpp"
#include "lbann/proto/helpers.hpp"
#include "lbann/utils/factory.hpp"
#include "lbann/utils/typename.hpp"

#include "lbann/layers/layer.hpp"
#include "lbann/layers/activations/activations.hpp"
#include "lbann/layers/activations/elu.hpp"
#include "lbann/layers/activations/identity.hpp"
#include "lbann/layers/activations/leaky_relu.hpp"
#include "lbann/layers/activations/relu.hpp"
#include "lbann/layers/activations/log_softmax.hpp"
#include "lbann/layers/activations/softmax.hpp"
#include "lbann/layers/image/bilinear_resize.hpp"
#include "lbann/layers/io/input/generic_input_layer.hpp"
#include "lbann/layers/io/input/input_layer.hpp"
#include "lbann/layers/io/io_layer.hpp"
#include "lbann/layers/learning/channelwise_fully_connected.hpp"
#include "lbann/layers/learning/channelwise_scale_bias.hpp"
#include "lbann/layers/learning/convolution.hpp"
#include "lbann/layers/learning/deconvolution.hpp"
#include "lbann/layers/learning/embedding.hpp"
#include "lbann/layers/learning/entrywise_scale_bias.hpp"
#include "lbann/layers/learning/fully_connected.hpp"
#include "lbann/layers/learning/gru.hpp"
#include "lbann/layers/learning/learning.hpp"
#include "lbann/layers/loss/categorical_accuracy.hpp"
#include "lbann/layers/loss/cross_entropy.hpp"
#include "lbann/layers/loss/entrywise.hpp"
#include "lbann/layers/loss/l1_norm.hpp"
#include "lbann/layers/loss/l2_norm2.hpp"
#include "lbann/layers/loss/mean_absolute_error.hpp"
#include "lbann/layers/loss/mean_squared_error.hpp"
#include "lbann/layers/loss/top_k_categorical_accuracy.hpp"
#include "lbann/layers/math/binary.hpp"
#include "lbann/layers/math/clamp.hpp"
#include "lbann/layers/math/matmul.hpp"
#include "lbann/layers/math/unary.hpp"
#include "lbann/layers/misc/channelwise_mean.hpp"
#include "lbann/layers/misc/channelwise_softmax.hpp"
#include "lbann/layers/misc/covariance.hpp"
#include "lbann/layers/misc/dft_abs_builder.hpp"
#include "lbann/layers/misc/mini_batch_index.hpp"
#include "lbann/layers/misc/mini_batch_size.hpp"
#include "lbann/layers/misc/variance.hpp"
#include "lbann/layers/misc/argmax.hpp"
#include "lbann/layers/misc/argmin.hpp"
#include "lbann/layers/misc/one_hot.hpp"
#include "lbann/layers/misc/dist_embedding.hpp"
#include "lbann/layers/regularizers/batch_normalization.hpp"
#include "lbann/layers/regularizers/dropout.hpp"
#include "lbann/layers/regularizers/local_response_normalization.hpp"
#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/layers/regularizers/selu_dropout.hpp"
#include "lbann/layers/regularizers/entrywise_batch_normalization.hpp"
#include "lbann/layers/regularizers/layer_norm.hpp"
#include "lbann/layers/regularizers/instance_norm.hpp"
#include "lbann/layers/transform/bernoulli.hpp"
#include "lbann/layers/transform/categorical_random.hpp"
#include "lbann/layers/transform/concatenate.hpp"
#include "lbann/layers/transform/constant.hpp"
#include "lbann/layers/transform/crop.hpp"
#include "lbann/layers/transform/discrete_random.hpp"
#include "lbann/layers/transform/dummy.hpp"
#include "lbann/layers/transform/evaluation.hpp"
#include "lbann/layers/transform/gaussian.hpp"
#include "lbann/layers/transform/hadamard.hpp"
#include "lbann/layers/transform/in_top_k.hpp"
#include "lbann/layers/transform/pooling.hpp"
#include "lbann/layers/transform/reduction.hpp"
#include "lbann/layers/transform/reshape.hpp"
#include "lbann/layers/transform/slice.hpp"
#include "lbann/layers/transform/sort.hpp"
#include "lbann/layers/transform/split.hpp"
#include "lbann/layers/transform/stop_gradient.hpp"
#include "lbann/layers/transform/sum.hpp"
#include "lbann/layers/transform/tessellate.hpp"
#include "lbann/layers/transform/transform.hpp"
#include "lbann/layers/transform/uniform.hpp"
#include "lbann/layers/transform/unpooling.hpp"
#include "lbann/layers/transform/weighted_sum.hpp"
#include "lbann/layers/transform/weights.hpp"

#include "lbann/data_coordinator/data_coordinator_metadata.hpp"
#include "lbann/utils/peek_map.hpp"

#include <layers.pb.h>

#ifdef LBANN_HAS_CUDNN
#include <cudnn.h>
#endif // LBANN_HAS_CUDNN

namespace lbann {
namespace proto {

namespace {

// Define the factory type.
using factory_type = lbann::generic_factory<
  lbann::Layer,
  std::string,
  generate_builder_type<lbann::Layer,
                        lbann_comm*,
                        const lbann_data::Layer&>,
  nullptr_key_error_policy>;

/** @brief Singleton holder for a factory.
 *
 *  @note This design requires that the builder function be valid for
 *  every combination of T, L, and D. That is, layer types for which a
 *  combination is invalid must handle that error inside their builder
 *  function.
 */
template <typename T, data_layout L, El::Device D>
class factory_manager
{
public:

  factory_manager() { register_default_builders(); }
  factory_type const& get() const noexcept { return factory_; }

private:

  // This macro simplifies the process of adding default builders
#define LBANN_REGISTER_BUILDER(KEY, LAYER_NAME)                         \
    factory_.register_builder(                                          \
      #KEY, build_##LAYER_NAME##_layer_from_pbuf<T,L,D>)
#define LBANN_REGISTER_DEFAULT_BUILDER(KEY, LAYER_NAME)                 \
    factory_.register_builder(                                          \
      #KEY,                                                             \
      [](lbann_comm* comm,                                              \
         lbann_data::Layer const&){                                     \
        return lbann::make_unique<LAYER_NAME##_layer<T,L,D>>(comm);     \
      })

  // Builder registration happens here
  void register_default_builders() {

    // Learning layers
    LBANN_REGISTER_BUILDER(Convolution, convolution);
    LBANN_REGISTER_BUILDER(ChannelwiseFullyConnected, channelwise_fully_connected);
    LBANN_REGISTER_BUILDER(ChannelwiseScaleBias, channelwise_scale_bias);
    LBANN_REGISTER_BUILDER(Embedding, embedding);
    LBANN_REGISTER_BUILDER(EntrywiseScaleBias, entrywise_scale_bias);
    LBANN_REGISTER_BUILDER(FullyConnected, fully_connected);
    LBANN_REGISTER_BUILDER(GRU, gru);

    // Math layers
    LBANN_REGISTER_DEFAULT_BUILDER(Abs, abs);
    LBANN_REGISTER_DEFAULT_BUILDER(Acos, acos);
    LBANN_REGISTER_DEFAULT_BUILDER(Acosh, acosh);
    LBANN_REGISTER_DEFAULT_BUILDER(Add, add);
    LBANN_REGISTER_DEFAULT_BUILDER(Asin, asin);
    LBANN_REGISTER_DEFAULT_BUILDER(Asinh, asinh);
    LBANN_REGISTER_DEFAULT_BUILDER(Atan, atan);
    LBANN_REGISTER_DEFAULT_BUILDER(Atanh, atanh);
    LBANN_REGISTER_DEFAULT_BUILDER(Ceil, ceil);
    LBANN_REGISTER_DEFAULT_BUILDER(Cos, cos);
    LBANN_REGISTER_DEFAULT_BUILDER(Cosh, cosh);
    LBANN_REGISTER_DEFAULT_BUILDER(Divide, divide);
    LBANN_REGISTER_DEFAULT_BUILDER(Equal, equal);
    LBANN_REGISTER_DEFAULT_BUILDER(Exp, exp);
    LBANN_REGISTER_DEFAULT_BUILDER(Expm1, expm1);
    LBANN_REGISTER_DEFAULT_BUILDER(Floor, floor);
    LBANN_REGISTER_DEFAULT_BUILDER(Greater, greater);
    LBANN_REGISTER_DEFAULT_BUILDER(GreaterEqual, greater_equal);
    LBANN_REGISTER_DEFAULT_BUILDER(Less, less);
    LBANN_REGISTER_DEFAULT_BUILDER(LessEqual, less_equal);
    LBANN_REGISTER_DEFAULT_BUILDER(Log, log);
    LBANN_REGISTER_DEFAULT_BUILDER(Log1p, log1p);
    LBANN_REGISTER_DEFAULT_BUILDER(LogicalAnd, logical_and);
    LBANN_REGISTER_DEFAULT_BUILDER(LogicalNot, logical_not);
    LBANN_REGISTER_DEFAULT_BUILDER(LogicalOr, logical_or);
    LBANN_REGISTER_DEFAULT_BUILDER(LogicalXor, logical_xor);
    LBANN_REGISTER_DEFAULT_BUILDER(Max, max);
    LBANN_REGISTER_DEFAULT_BUILDER(Min, min);
    LBANN_REGISTER_DEFAULT_BUILDER(Mod, mod);
    LBANN_REGISTER_DEFAULT_BUILDER(Multiply, multiply);
    LBANN_REGISTER_DEFAULT_BUILDER(Negative, negative);
    LBANN_REGISTER_DEFAULT_BUILDER(NotEqual, not_equal);
    LBANN_REGISTER_DEFAULT_BUILDER(Pow, pow);
    LBANN_REGISTER_DEFAULT_BUILDER(Reciprocal, reciprocal);
    LBANN_REGISTER_DEFAULT_BUILDER(Round, round);
    LBANN_REGISTER_DEFAULT_BUILDER(Rsqrt, rsqrt);
    LBANN_REGISTER_DEFAULT_BUILDER(SafeDivide, safe_divide);
    LBANN_REGISTER_DEFAULT_BUILDER(SafeReciprocal, safe_reciprocal);
    LBANN_REGISTER_DEFAULT_BUILDER(Sign, sign);
    LBANN_REGISTER_DEFAULT_BUILDER(Sin, sin);
    LBANN_REGISTER_DEFAULT_BUILDER(Sinh, sinh);
    LBANN_REGISTER_DEFAULT_BUILDER(Sqrt, sqrt);
    LBANN_REGISTER_DEFAULT_BUILDER(Square, square);
    LBANN_REGISTER_DEFAULT_BUILDER(SquaredDifference, squared_difference);
    LBANN_REGISTER_DEFAULT_BUILDER(Subtract, subtract);
    LBANN_REGISTER_DEFAULT_BUILDER(Tan, tan);
    LBANN_REGISTER_DEFAULT_BUILDER(Tanh, tanh);

    // Transform layers
    LBANN_REGISTER_BUILDER(Bernoulli, bernoulli);
    LBANN_REGISTER_BUILDER(CategoricalRandom, categorical_random);
    LBANN_REGISTER_BUILDER(Concatenation, concatenate);
    LBANN_REGISTER_BUILDER(Constant, constant);
    LBANN_REGISTER_BUILDER(Crop, crop);
    LBANN_REGISTER_BUILDER(Dummy, dummy);
    LBANN_REGISTER_BUILDER(Evaluation, evaluation);
    LBANN_REGISTER_BUILDER(Hadamard, hadamard);
    LBANN_REGISTER_BUILDER(Pooling, pooling);
    LBANN_REGISTER_BUILDER(Split, split);
    LBANN_REGISTER_BUILDER(StopGradient, stop_gradient);
    LBANN_REGISTER_BUILDER(Sum, sum);
    LBANN_REGISTER_BUILDER(WeightedSum, weighted_sum);
    LBANN_REGISTER_BUILDER(WeightsLayer, weights);

    // Activations
    LBANN_REGISTER_DEFAULT_BUILDER(Identity, identity);
    LBANN_REGISTER_DEFAULT_BUILDER(LogSigmoid, log_sigmoid);
    LBANN_REGISTER_DEFAULT_BUILDER(LogSoftmax, log_softmax);
    LBANN_REGISTER_DEFAULT_BUILDER(Relu, relu);
    LBANN_REGISTER_DEFAULT_BUILDER(Selu, selu);
    LBANN_REGISTER_DEFAULT_BUILDER(Sigmoid, sigmoid);
    LBANN_REGISTER_BUILDER(Softmax, softmax);
    LBANN_REGISTER_DEFAULT_BUILDER(Softplus, softplus);
    LBANN_REGISTER_DEFAULT_BUILDER(Softsign, softsign);

    // Loss Layers
    LBANN_REGISTER_DEFAULT_BUILDER(BinaryCrossEntropy, binary_cross_entropy);
    LBANN_REGISTER_DEFAULT_BUILDER(BooleanAccuracy, boolean_accuracy);
    LBANN_REGISTER_DEFAULT_BUILDER(BooleanFalseNegative, boolean_false_negative);
    LBANN_REGISTER_DEFAULT_BUILDER(BooleanFalsePositive, boolean_false_positive);
    LBANN_REGISTER_DEFAULT_BUILDER(CategoricalAccuracy, categorical_accuracy);
    LBANN_REGISTER_DEFAULT_BUILDER(L1Norm, l1_norm);
    LBANN_REGISTER_DEFAULT_BUILDER(L2Norm2, l2_norm2);
    LBANN_REGISTER_DEFAULT_BUILDER(MeanAbsoluteError, mean_absolute_error);
    LBANN_REGISTER_DEFAULT_BUILDER(MeanSquaredError, mean_squared_error);
    LBANN_REGISTER_DEFAULT_BUILDER(SigmoidBinaryCrossEntropy, sigmoid_binary_cross_entropy);

    // Regularizer layers
    LBANN_REGISTER_BUILDER(Dropout, dropout);
    LBANN_REGISTER_BUILDER(InstanceNorm, instance_norm);
    LBANN_REGISTER_BUILDER(LocalResponseNormalization,
                           local_response_normalization);
    // Miscellaneous layers
    LBANN_REGISTER_BUILDER(DFTAbs, dft_abs);
    LBANN_REGISTER_BUILDER(ChannelwiseSoftmax, channelwise_softmax);
    LBANN_REGISTER_DEFAULT_BUILDER(MiniBatchIndex, mini_batch_index);
    LBANN_REGISTER_DEFAULT_BUILDER(MiniBatchSize, mini_batch_size);
    LBANN_REGISTER_BUILDER(DistEmbedding, dist_embedding);

  }

  // Just to be clear/safe.
#undef LBANN_REGISTER_DEFAULT_BUILDER

private:
  factory_type factory_;
}; // class factory_manager

template <typename T, data_layout L, El::Device D>
factory_type const& get_layer_factory() noexcept
{
  static factory_manager<T,L,D> factory_mgr_;
  return factory_mgr_.get();
}

// Some cuDNN stuff -- copied from convolution.cpp. To what common
// location should this go?? The problem is it's the confluence of two
// evils: protobuf and cudnn. I'd rather they never meet, but whatdya
// gonna do.
#ifdef LBANN_HAS_CUDNN
using ProtoTensorOpEnumType = decltype(lbann_data::DEFAULT_TENSOR_OPS);
cudnnMathType_t convert_to_cudnn_math_type(ProtoTensorOpEnumType mt)
{
  switch (mt)
  {
  case lbann_data::DEFAULT_TENSOR_OPS:
    return cudnn::get_default_convolution_math_type();
  case lbann_data::NO_TENSOR_OPS:
    return CUDNN_DEFAULT_MATH;
  case lbann_data::USE_TENSOR_OPS:
    return CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
  default:
    LBANN_ERROR("Bad math type value.");
  }
  return CUDNN_DEFAULT_MATH;
}
#endif // LBANN_HAS_CUDNN
} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> construct_layer_legacy(
  lbann_comm* comm,
  int training_dr_linearized_data_size,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer) {
  std::stringstream err;

  // Input layers
  // Currently this cannot be suitably removed from this function
  // because it relies on "num_parallel_readers" and "data_readers"
  // arguments.
  if (proto_layer.has_input()) {
    const auto& params = proto_layer.input();
    const auto& io_buffer = params.io_buffer();
    const auto& mode_str = params.target_mode();
    data_reader_target_mode target_mode = data_reader_target_mode::CLASSIFICATION;
    if (mode_str.empty() || mode_str == "classification") { target_mode = data_reader_target_mode::CLASSIFICATION; }
    if (mode_str == "regression")                         { target_mode = data_reader_target_mode::REGRESSION; }
    if (mode_str == "reconstruction")                     { target_mode = data_reader_target_mode::RECONSTRUCTION; }
    if (mode_str == "label_reconstruction")               { target_mode = data_reader_target_mode::LABEL_RECONSTRUCTION; }
    if (mode_str == "na" || mode_str == "NA" || mode_str == "N/A") { target_mode = data_reader_target_mode::NA; }
    if (Layout != data_layout::DATA_PARALLEL) {
      LBANN_ERROR("input layer is only supported with "
                  "a data-parallel layout");
    }
    if (io_buffer == "partitioned" || io_buffer.empty()) {
      /// @todo Question for Tim Moon and Tom Benson, I had to change this line from Layout to
      /// data_layout::DATA_PARALLEL to make it compile with clang on OS X, but it seems like
      /// this is not related to this PR.
      if ((typeid(TensorDataType) == typeid(DataType))
          && (Layout == data_layout::DATA_PARALLEL)) {
        return lbann::make_unique<input_layer<DataType,
                                              partitioned_io_buffer<DataType>,
                                              data_layout::DATA_PARALLEL,
                                              Device>>(
                                                comm,
                                                num_parallel_readers,
                                                target_mode);
      }
      else {
        LBANN_ERROR("Input layers are only valid with "
                    "TensorDataType == DataType and Layout == DATA_PARALLEL");
      }
    } else {
      LBANN_ERROR("invalid IO buffer type (" + io_buffer + ")");
    }
  }

  // Currently this cannot be suitably removed from this function
  // because it relies on "num_parallel_readers" and "data_readers"
  // arguments.
  if (proto_layer.has_deconvolution()) {
    const auto& params = proto_layer.deconvolution();
    const auto& bias = params.has_bias();
    int num_output_channels = params.num_output_channels();
    int num_groups = params.num_groups();
    if (num_groups == 0) {
      num_groups = 1;
    }
    if (proto_layer.num_neurons_from_data_reader()) {
      if (training_dr_linearized_data_size == -1) {
        LBANN_ERROR("Training data reader does not exist!");
      }
      num_output_channels = training_dr_linearized_data_size;
    }
    if (Layout != data_layout::DATA_PARALLEL) {
      LBANN_ERROR("deconvolution layer is only supported with "
                  "a data-parallel layout");
    }
    if (params.has_vectors()) {
      const auto& dims = parse_list<int>(params.conv_dims());
      const auto& pads = parse_list<int>(params.conv_pads());
      const auto& strides = parse_list<int>(params.conv_strides());
      std::vector<int> dilations = parse_list<int>(params.conv_dilations());
      if (dilations.empty()) {
        dilations.resize(dims.size(), 1);
      }
#ifdef LBANN_HAS_CUDNN
      auto ret = lbann::make_unique<deconvolution_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
        comm, dims.size(), num_output_channels,
        dims, pads, strides, dilations, num_groups, bias);
      ret->set_cudnn_math_mode(
        convert_to_cudnn_math_type(params.conv_tensor_op_mode()));
      return ret;
#else
      return lbann::make_unique<deconvolution_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
               comm, dims.size(), num_output_channels,
               dims, pads, strides, dilations, num_groups, bias);
#endif // LBANN_HAS_CUDNN

    } else {
      const auto& num_dims = params.num_dims();
      const auto& dim = params.conv_dims_i();
      const auto& pad = params.conv_pads_i();
      const auto& stride = params.conv_strides_i();
      int dilation = params.conv_dilations_i();
      if (dilation == 0) {
        dilation = 1;
      }
#ifdef LBANN_HAS_CUDNN
      auto ret = lbann::make_unique<deconvolution_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
        comm, num_dims, num_output_channels,
        dim, pad, stride, dilation, num_groups, bias);
      ret->set_cudnn_math_mode(
        convert_to_cudnn_math_type(params.conv_tensor_op_mode()));
      return ret;
#else
      return lbann::make_unique<deconvolution_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
        comm, num_dims, num_output_channels,
        dim, pad, stride, dilation, num_groups, bias);
#endif // LBANN_HAS_CUDNN
    }
  }

  // Transform layers
  // Currently this cannot be suitably removed from this function
  // because it relies on "num_parallel_readers" and "data_readers"
  // arguments.
  if (proto_layer.has_reshape()) {
    const auto& params = proto_layer.reshape();
    std::vector<int> dims = parse_list<int>(params.dims());
    if (params.num_dims() != 0) {
      LBANN_WARNING("found unused and deprecated prototext field (Reshape.num_dims)");
    }
    if (proto_layer.num_neurons_from_data_reader()) {
      dims.clear();
      if (training_dr_linearized_data_size == -1) {
        LBANN_ERROR("Training data reader does not exist!");
      }
      dims.push_back(training_dr_linearized_data_size);
    }
    return lbann::make_unique<reshape_layer<TensorDataType, Layout, Device>>(comm, dims);
  }

  // Currently this cannot be suitably removed from this function
  // because it relies on "num_parallel_readers" and "data_readers"
  // arguments.
  if (proto_layer.has_slice()) {
    const auto& params = proto_layer.slice();
    std::vector<size_t> slice_points;

    auto layer = lbann::make_unique<slice_layer<TensorDataType, Layout, Device>>(comm);

    if (params.get_slice_points_from_reader() != "") {
      const slice_points_mode var = slice_points_mode_from_string(params.get_slice_points_from_reader());
      layer->setup_slice_points(params.axis(), true, var);
    } else {
      std::string slice_point_method_name = "'slice_points'";
      slice_points = parse_list<size_t>(params.slice_points());
      if (slice_points.size() < 2u) {
        err << "Failed to get slice points via " << slice_point_method_name << '.';
        LBANN_ERROR(err.str());
        return nullptr;
      }
      layer->setup_slice_points(params.axis(), slice_points);
    }
    return layer;
  }
  if (proto_layer.has_gaussian()) {
    const auto& params = proto_layer.gaussian();
    const auto& dims = parse_list<int>(params.neuron_dims());
    double mean = params.mean();
    double stdev = params.stdev();
    if (mean == 0.0 && stdev == 0.0) {
      mean = 0.0;
      stdev = 1.0;
    }
    return lbann::make_unique<gaussian_layer<TensorDataType,Layout,Device>>(
      comm,
      dims,
      mean,
      stdev,
      params.training_only());
  }
  if (proto_layer.has_uniform()) {
    const auto& params = proto_layer.uniform();
    const auto& dims = parse_list<int>(params.neuron_dims());
    double min = params.min();
    double max = params.max();
    if (min == 0.0 && max == 0.0) {
      min = 0.0;
      max = 1.0;
    }
    return lbann::make_unique<uniform_layer<TensorDataType,Layout,Device>>(
      comm,
      dims,
      min,
      max,
      params.training_only());
  }
  if (proto_layer.has_unpooling()) {
    if (Layout == data_layout::DATA_PARALLEL && Device == El::Device::CPU) {
      return lbann::make_unique<unpooling_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
    } else {
      LBANN_ERROR("unpooling layer is only supported with "
                  "a data-parallel layout and on CPU");
    }
  }
  if (proto_layer.has_reduction()) {
    const auto& params = proto_layer.reduction();
    const auto& mode_str = params.mode();
    reduction_mode mode = reduction_mode::INVALID;
    if (mode_str == "sum" || mode_str.empty()) { mode = reduction_mode::SUM; }
    if (mode_str == "average") { mode = reduction_mode::AVERAGE; }
    if (Layout == data_layout::DATA_PARALLEL) {
      return lbann::make_unique<reduction_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(comm, mode);
    } else {
      LBANN_ERROR("reduction layer is only supported with "
                  "a data-parallel layout");
    }
  }
  if (proto_layer.has_discrete_random()) {
    const auto& params = proto_layer.discrete_random();
    const auto& values = parse_list<DataType>(params.values());
    const auto& dims = parse_list<int>(params.dims());
    if (Layout == data_layout::DATA_PARALLEL
        && Device == El::Device::CPU) {
      return lbann::make_unique<discrete_random_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(
               comm, values, dims);
    } else {
      LBANN_ERROR("discrete random layer is only supported on CPU");
    }
  }
  if (proto_layer.has_in_top_k()) {
    const auto& params = proto_layer.in_top_k();
    return lbann::make_unique<in_top_k_layer<TensorDataType, Layout, Device>>(comm, params.k());
  }
  if (proto_layer.has_sort()) {
    const auto& params = proto_layer.sort();
    if (Layout == data_layout::DATA_PARALLEL) {
      return lbann::make_unique<sort_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(comm, params.descending());
    } else {
      LBANN_ERROR("sort layer is only supported with "
                  "a data-parallel layout");
    }
  }
  if (proto_layer.has_tessellate()) {
    const auto& params = proto_layer.tessellate();
    const auto& dims = parse_list<int>(params.dims());
    return lbann::make_unique<tessellate_layer<TensorDataType, Layout, Device>>(comm, dims);
  }

  // Regularizer layers
  if (proto_layer.has_batch_normalization()) {
    const auto& params = proto_layer.batch_normalization();
    if (Layout == data_layout::DATA_PARALLEL) {
      int statistics_group_size = params.statistics_group_size();
      if (statistics_group_size < 0) {
        statistics_group_size = 0;  // Global statistics.
      } else if (statistics_group_size == 0) {
        statistics_group_size = 1;  // Default to local.
      }
      const auto& aggr_str = params.stats_aggregation();
      if (!aggr_str.empty()) {
        LBANN_WARNING("stats_aggregation field for BatchNormalization is deprecated");
        if (aggr_str == "local") {
          statistics_group_size = 1;
        } else if (aggr_str == "node_local") {
          statistics_group_size = comm->get_procs_per_node();
        } else if (aggr_str == "global") {
          statistics_group_size = 0;
        } else {
          err << "Invalid batch normalization stats aggregation " << aggr_str;
          LBANN_ERROR(err.str());
          return nullptr;
        }
      }
      // Set defaults if not given.
      auto decay = params.decay();
      if (decay == 0.0) {
        decay = 0.9;
      }
      auto epsilon = params.epsilon();
      if (epsilon == 0.0) {
        epsilon = 1e-5;
      }
      return lbann::make_unique<batch_normalization_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
        comm,
        decay,
        epsilon,
        statistics_group_size);
    } else {
      LBANN_ERROR("batch normalization layer is only supported with "
                  "a data-parallel layout");
    }
  }
  if (proto_layer.has_selu_dropout()) {
    const auto& params = proto_layer.selu_dropout();
    const auto& keep_prob = params.keep_prob();
    const auto& alpha = params.alpha();
    const auto& scale = params.scale();
    if (alpha != 0.0 && scale != 0.0) {
      return lbann::make_unique<selu_dropout<TensorDataType, Layout, Device>>(comm, keep_prob, alpha, scale);
    } else {
      return lbann::make_unique<selu_dropout<TensorDataType, Layout, Device>>(comm, keep_prob);
    }
  }
  if (proto_layer.has_entrywise_batch_normalization()) {
    const auto& params = proto_layer.entrywise_batch_normalization();
    return lbann::make_unique<entrywise_batch_normalization_layer<TensorDataType, Layout, Device>>(comm, params.decay(), params.epsilon());
  }
  if (proto_layer.has_layer_norm()) {
    const auto& params = proto_layer.layer_norm();
    const double epsilon = (params.has_epsilon()
                            ? params.epsilon().value()
                            : 1e-5);
    return lbann::make_unique<layer_norm_layer<TensorDataType, Layout, Device>>(comm, epsilon);
  }

  if (proto_layer.has_clamp()) {
    const auto& params = proto_layer.clamp();
    return lbann::make_unique<clamp_layer<TensorDataType, Layout, Device>>(comm, params.min(), params.max());
  }
  if (proto_layer.has_matmul()) {
    if (Layout == data_layout::DATA_PARALLEL) {
      const auto& params = proto_layer.matmul();
      return lbann::make_unique<matmul_layer<TensorDataType, data_layout::DATA_PARALLEL,Device>>(
               comm,
               params.transpose_a(),
               params.transpose_b());
    } else {
      LBANN_ERROR("matrix multiply layer is only supported with "
                  "a data-parallel layout");
    }
  }

  // Activation layers
  if (proto_layer.has_elu()) {
    const auto& params = proto_layer.elu();
    const auto& alpha = params.alpha();
    if (alpha != 0) {
      return lbann::make_unique<elu_layer<TensorDataType, Layout, Device>>(comm, alpha);
    } else {
      return lbann::make_unique<elu_layer<TensorDataType, Layout, Device>>(comm);
    }
  }
  if (proto_layer.has_leaky_relu()) {
    const auto& params = proto_layer.leaky_relu();
    const auto& negative_slope = params.negative_slope();
    if (negative_slope != 0) {
      return lbann::make_unique<leaky_relu_layer<TensorDataType, Layout, Device>>(comm, negative_slope);
    } else {
      return lbann::make_unique<leaky_relu_layer<TensorDataType, Layout, Device>>(comm);
    }
  }

  // Loss layers
  if (proto_layer.has_cross_entropy()) {
    const auto& params = proto_layer.cross_entropy();
    return lbann::make_unique<cross_entropy_layer<TensorDataType, Layout, Device>>(comm, params.use_labels());
  }
  if (proto_layer.has_top_k_categorical_accuracy()) {
    const auto& params = proto_layer.top_k_categorical_accuracy();
    return lbann::make_unique<top_k_categorical_accuracy_layer<TensorDataType, Layout, Device>>(comm, params.k());
  }

  // Image layers
  if (proto_layer.has_bilinear_resize()) {
    const auto& params = proto_layer.bilinear_resize();
    if (Layout == data_layout::DATA_PARALLEL) {
      return lbann::make_unique<bilinear_resize_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(
               comm, params.height(), params.width());
    } else {
      LBANN_ERROR("bilinear resize layer is only supported with "
                  "a data-parallel layout");
    }
  }

  // Miscellaneous layers
  if (proto_layer.has_covariance()) {
    const auto& params = proto_layer.covariance();
    return lbann::make_unique<covariance_layer<TensorDataType, Layout, Device>>(comm, params.biased());
  }
  if (proto_layer.has_variance()) {
    const auto& params = proto_layer.variance();
    return lbann::make_unique<variance_layer<TensorDataType, Layout, Device>>(comm, params.biased());
  }
  if (proto_layer.has_channelwise_mean()) {
    if (Layout == data_layout::DATA_PARALLEL) {
      return lbann::make_unique<channelwise_mean_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(comm);
    } else {
      LBANN_ERROR("channel-wise mean layer is only supported with "
                  "a data-parallel layout");
    }
  }
  if (proto_layer.has_argmax()) {
    if (Layout == data_layout::DATA_PARALLEL && Device == El::Device::CPU) {
      return lbann::make_unique<argmax_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
    } else {
      LBANN_ERROR("argmax layer is only supported with "
                  "a data-parallel layout and on CPU");
    }
  }
  if (proto_layer.has_argmin()) {
    if (Layout == data_layout::DATA_PARALLEL && Device == El::Device::CPU) {
      return lbann::make_unique<argmin_layer<TensorDataType, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
    } else {
      LBANN_ERROR("argmin layer is only supported with "
                  "a data-parallel layout and on CPU");
    }
  }
  if (proto_layer.has_one_hot()) {
    if (Layout == data_layout::DATA_PARALLEL) {
      const auto& params = proto_layer.one_hot();
      return lbann::make_unique<one_hot_layer<TensorDataType, data_layout::DATA_PARALLEL, Device>>(comm, params.size());
    } else {
      LBANN_ERROR("one-hot layer is only supported with "
                  "a data-parallel layout");
    }
  }

  // Throw exception if layer has not been constructed
  err << "could not construct layer " << proto_layer.name();
  LBANN_ERROR(err.str());
  return nullptr;

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> construct_layer(
  lbann_comm* comm,
  int training_dr_linearized_data_size,
  int num_parallel_readers,
  const lbann_data::Layer& proto_layer) {

  auto const& factory = get_layer_factory<TensorDataType, Layout, Device>();
  auto const& msg =
    helpers::get_oneof_message(proto_layer, "layer_type");

  std::unique_ptr<Layer> l = factory.create_object(
    msg.GetDescriptor()->name(), comm, proto_layer);
  if(!l) {
    if (typeid(TensorDataType) == typeid(DataType))
      l = construct_layer_legacy<DataType, Layout, Device>(
            comm, training_dr_linearized_data_size, num_parallel_readers, proto_layer);
    else
      LBANN_ERROR("Currently, layers of type \"", msg.GetDescriptor()->name(),
                  "\" are not constructible with any type other than the "
                  "default DataType.");
  }
  return l;
}

// Template instantiation
#define PROTO_DEVICE(T, Device) \
  template std::unique_ptr<Layer> construct_layer<T, data_layout::DATA_PARALLEL, Device>(  \
    lbann_comm* comm,                                                                      \
    int training_dr_linearized_data_size,                                                  \
    int num_parallel_readers,                                                              \
    const lbann_data::Layer& proto_layer                                                   \
  );                                                                                       \
  template std::unique_ptr<Layer> construct_layer<T, data_layout::MODEL_PARALLEL, Device>( \
    lbann_comm* comm,                                                                      \
    int training_dr_linearized_data_size,                                                  \
    int num_parallel_readers,                                                              \
    const lbann_data::Layer& proto_layer                                                   \
  )

#include "lbann/macros/instantiate_device.hpp"

} // namespace proto
} // namespace lbann
