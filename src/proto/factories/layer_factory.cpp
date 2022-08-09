////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
#include "lbann/utils/factory.hpp"
#include "lbann/utils/protobuf.hpp"

#include "lbann/layers/layer.hpp"
#include "lbann/layers/operator_layer.hpp"

#include "lbann/layers/activations/activation_layer_builders.hpp"
#include "lbann/layers/image/image_layer_builders.hpp"
#include "lbann/layers/loss/loss_layer_builders.hpp"
#include "lbann/layers/math/math_builders.hpp"
#include "lbann/layers/misc/misc_builders.hpp"
#include "lbann/layers/transform/transform_builders.hpp"

// This is the one remaining "if" branch of the legacy factory function.
#include "lbann/layers/transform/reshape.hpp"

#include "lbann/data_coordinator/data_coordinator_metadata.hpp"

#include <layers.pb.h>

namespace lbann {

// Declarations of the builder functions.

// I/O
LBANN_DEFINE_LAYER_BUILDER(input);

// Learning
LBANN_DEFINE_LAYER_BUILDER(channelwise_fully_connected);
LBANN_DEFINE_LAYER_BUILDER(channelwise_scale_bias);
LBANN_DEFINE_LAYER_BUILDER(convolution);
LBANN_DEFINE_LAYER_BUILDER(deconvolution);
LBANN_DEFINE_LAYER_BUILDER(embedding);
LBANN_DEFINE_LAYER_BUILDER(entrywise_scale_bias);
LBANN_DEFINE_LAYER_BUILDER(fully_connected);
LBANN_DEFINE_LAYER_BUILDER(gru);

// Regularizers
LBANN_DEFINE_LAYER_BUILDER(batch_normalization);
LBANN_DEFINE_LAYER_BUILDER(dropout);
LBANN_DEFINE_LAYER_BUILDER(local_response_normalization);
LBANN_DEFINE_LAYER_BUILDER(selu_dropout);
LBANN_DEFINE_LAYER_BUILDER(entrywise_batch_normalization);
LBANN_DEFINE_LAYER_BUILDER(layer_norm);
LBANN_DEFINE_LAYER_BUILDER(instance_norm);

namespace proto {

namespace {

// Define the factory type.
using factory_type = lbann::generic_factory<
  lbann::Layer,
  std::string,
  generate_builder_type<lbann::Layer, lbann_comm*, const lbann_data::Layer&>,
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
#define LBANN_REGISTER_BUILDER(KEY, LAYER_NAME)                                \
  factory_.register_builder(#KEY, build_##LAYER_NAME##_layer_from_pbuf<T, L, D>)

  // Builder registration happens here
  void register_default_builders()
  {

    // For now, we add a custom builder that will use the same
    // input/output type for the multi-precision-capable
    // OperatorLayer. This is temporary, until more of the factory
    // infrastructure considers multiple in/out types.
    factory_.register_builder(
      "OperatorLayer",
      build_operator_layer_from_pbuf<T, T, L, D>);

    // Input layer
    LBANN_REGISTER_BUILDER(Input, input);

    // Learning layers
    LBANN_REGISTER_BUILDER(Convolution, convolution);
    LBANN_REGISTER_BUILDER(ChannelwiseFullyConnected,
                           channelwise_fully_connected);
    LBANN_REGISTER_BUILDER(ChannelwiseScaleBias, channelwise_scale_bias);
    LBANN_REGISTER_BUILDER(Deconvolution, deconvolution);
    LBANN_REGISTER_BUILDER(Embedding, embedding);
    LBANN_REGISTER_BUILDER(EntrywiseScaleBias, entrywise_scale_bias);
    LBANN_REGISTER_BUILDER(FullyConnected, fully_connected);
    LBANN_REGISTER_BUILDER(GRU, gru);

    // Math layers
    LBANN_REGISTER_BUILDER(MatMul, matmul);

    // Transform layers
    LBANN_REGISTER_BUILDER(BatchwiseReduceSum, batchwise_reduce_sum);
    LBANN_REGISTER_BUILDER(Bernoulli, bernoulli);
    LBANN_REGISTER_BUILDER(CategoricalRandom, categorical_random);
    LBANN_REGISTER_BUILDER(Concatenation, concatenate);
    LBANN_REGISTER_BUILDER(Constant, constant);
    LBANN_REGISTER_BUILDER(Crop, crop);
    LBANN_REGISTER_BUILDER(Cross_Grid_Sum_Slice, cross_grid_sum_slice);
    LBANN_REGISTER_BUILDER(Cross_Grid_Sum, cross_grid_sum);
    LBANN_REGISTER_BUILDER(DiscreteRandom, discrete_random);
    LBANN_REGISTER_BUILDER(Dummy, dummy);
    LBANN_REGISTER_BUILDER(Evaluation, evaluation);
    LBANN_REGISTER_BUILDER(Gather, gather);
    LBANN_REGISTER_BUILDER(Gaussian, gaussian);
    LBANN_REGISTER_BUILDER(Hadamard, hadamard);
    LBANN_REGISTER_BUILDER(InTopK, in_top_k);
    LBANN_REGISTER_BUILDER(Pooling, pooling);
    LBANN_REGISTER_BUILDER(Reduction, reduction);
    LBANN_REGISTER_BUILDER(Scatter, scatter);
    LBANN_REGISTER_BUILDER(Slice, slice);
    LBANN_REGISTER_BUILDER(Sort, sort);
    LBANN_REGISTER_BUILDER(Split, split);
    LBANN_REGISTER_BUILDER(StopGradient, stop_gradient);
    LBANN_REGISTER_BUILDER(Sum, sum);
    LBANN_REGISTER_BUILDER(TensorPermute, permute);
    LBANN_REGISTER_BUILDER(Tessellate, tessellate);
    LBANN_REGISTER_BUILDER(Uniform, uniform);
    LBANN_REGISTER_BUILDER(Unpooling, unpooling);
    LBANN_REGISTER_BUILDER(WeightedSum, weighted_sum);
    LBANN_REGISTER_BUILDER(WeightsLayer, weights);

    // Activations
    LBANN_REGISTER_BUILDER(Elu, elu);
    LBANN_REGISTER_BUILDER(Identity, identity);
    LBANN_REGISTER_BUILDER(LeakyRelu, leaky_relu);
    LBANN_REGISTER_BUILDER(LogSoftmax, log_softmax);
    LBANN_REGISTER_BUILDER(Relu, relu);
    LBANN_REGISTER_BUILDER(Softmax, softmax);

    // Loss Layers
    LBANN_REGISTER_BUILDER(CategoricalAccuracy, categorical_accuracy);
    LBANN_REGISTER_BUILDER(CrossEntropy, cross_entropy);
    LBANN_REGISTER_BUILDER(L1Norm, l1_norm);
    LBANN_REGISTER_BUILDER(L2Norm2, l2_norm2);
    LBANN_REGISTER_BUILDER(MeanAbsoluteError, mean_absolute_error);
    LBANN_REGISTER_BUILDER(MeanSquaredError, mean_squared_error);
    LBANN_REGISTER_BUILDER(TopKCategoricalAccuracy, top_k_categorical_accuracy);

    // Regularizer layers
    LBANN_REGISTER_BUILDER(BatchNormalization, batch_normalization);
    LBANN_REGISTER_BUILDER(Dropout, dropout);
    LBANN_REGISTER_BUILDER(EntrywiseBatchNormalization,
                           entrywise_batch_normalization);
    LBANN_REGISTER_BUILDER(InstanceNorm, instance_norm);
    LBANN_REGISTER_BUILDER(LayerNorm, layer_norm);
    LBANN_REGISTER_BUILDER(LocalResponseNormalization,
                           local_response_normalization);
    LBANN_REGISTER_BUILDER(SeluDropout, selu_dropout);

    // Image layers
    LBANN_REGISTER_BUILDER(BilinearResize, bilinear_resize);
    LBANN_REGISTER_BUILDER(CompositeImageTransformation,
                           composite_image_transformation);
    LBANN_REGISTER_BUILDER(Rotation, rotation);
    LBANN_REGISTER_BUILDER(Cutout, cutout);

    // Miscellaneous layers
    LBANN_REGISTER_BUILDER(Argmax, argmax);
    LBANN_REGISTER_BUILDER(Argmin, argmin);
    LBANN_REGISTER_BUILDER(ChannelwiseMean, channelwise_mean);
    LBANN_REGISTER_BUILDER(ChannelwiseSoftmax, channelwise_softmax);
    LBANN_REGISTER_BUILDER(Covariance, covariance);
    LBANN_REGISTER_BUILDER(DFTAbs, dft_abs);
    LBANN_REGISTER_BUILDER(DistEmbedding, dist_embedding);
    LBANN_REGISTER_BUILDER(MiniBatchIndex, mini_batch_index);
    LBANN_REGISTER_BUILDER(MiniBatchSize, mini_batch_size);
    LBANN_REGISTER_BUILDER(OneHot, one_hot);
    LBANN_REGISTER_BUILDER(RowwiseWeightsNorms, rowwise_weights_norms);
    LBANN_REGISTER_BUILDER(UniformHash, uniform_hash);
    LBANN_REGISTER_BUILDER(Variance, variance);
  }
#undef LBANN_REGISTER_BUILDER

private:
  factory_type factory_;
}; // class factory_manager

template <typename T, data_layout L, El::Device D>
factory_type const& get_layer_factory() noexcept
{
  static factory_manager<T, L, D> factory_mgr_;
  return factory_mgr_.get();
}

} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer>
construct_layer_legacy(lbann_comm* comm,
                       int training_dr_linearized_data_size,
                       int num_parallel_readers,
                       const lbann_data::Layer& proto_layer)
{

  // Transform layers
  // Currently this cannot be suitably removed from this function
  // because it relies on LBANN_OPTION_NUM_PARALLEL_READERS and "data_readers"
  // arguments.
  if (proto_layer.has_reshape()) {
    const auto& params = proto_layer.reshape();
    if (proto_layer.num_neurons_from_data_reader()) {
      if (training_dr_linearized_data_size == -1) {
        LBANN_ERROR("Training data reader does not exist!");
      }
      return std::make_unique<reshape_layer<TensorDataType, Layout, Device>>(
        comm,
        std::vector<int>{training_dr_linearized_data_size});
    }
    return std::make_unique<reshape_layer<TensorDataType, Layout, Device>>(
      comm,
      protobuf::to_vector<int>(params.dims()));
  }

  // Throw exception if layer has not been constructed
  LBANN_ERROR("could not construct layer ", proto_layer.name());
  return nullptr;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> construct_layer(lbann_comm* comm,
                                       int training_dr_linearized_data_size,
                                       int num_parallel_readers,
                                       const lbann_data::Layer& proto_layer)
{

  // Construct layer
  auto const& factory = get_layer_factory<TensorDataType, Layout, Device>();
  auto const& msg = protobuf::get_oneof_message(proto_layer, "layer_type");
  auto l =
    factory.create_object(msg.GetDescriptor()->name(), comm, proto_layer);
  if (!l) {
    if constexpr (std::is_same_v<TensorDataType, DataType>)
      l = construct_layer_legacy<DataType, Layout, Device>(
        comm,
        training_dr_linearized_data_size,
        num_parallel_readers,
        proto_layer);
    else {
      LBANN_ERROR("Currently, layers of type \"",
                  msg.GetDescriptor()->name(),
                  "\" are not constructible with any type other than the "
                  "default DataType.");
      return nullptr;
    }
  }

  // Set additional parameters
  if (proto_layer.parallel_strategy().has_grid_tag()) {
    l->set_grid_tag(proto_layer.parallel_strategy().grid_tag().value());
  }

  return l;
}

// Template instantiation
#define PROTO_DEVICE(T, Device)                                                \
  template std::unique_ptr<Layer>                                              \
  construct_layer<T, data_layout::DATA_PARALLEL, Device>(                      \
    lbann_comm * comm,                                                         \
    int training_dr_linearized_data_size,                                      \
    int num_parallel_readers,                                                  \
    const lbann_data::Layer& proto_layer);                                     \
  template std::unique_ptr<Layer>                                              \
  construct_layer<T, data_layout::MODEL_PARALLEL, Device>(                     \
    lbann_comm * comm,                                                         \
    int training_dr_linearized_data_size,                                      \
    int num_parallel_readers,                                                  \
    const lbann_data::Layer& proto_layer)

#include "lbann/macros/instantiate_device.hpp"

} // namespace proto
} // namespace lbann
