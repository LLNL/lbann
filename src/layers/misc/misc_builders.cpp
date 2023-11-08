////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/layers/misc/misc_builders.hpp"

#include "lbann/layers/misc/argmax.hpp"
#include "lbann/layers/misc/argmin.hpp"
#include "lbann/layers/misc/channelwise_mean.hpp"
#include "lbann/layers/misc/channelwise_softmax.hpp"
#include "lbann/layers/misc/covariance.hpp"
#include "lbann/layers/misc/dft_abs.hpp"
#include "lbann/layers/misc/external.hpp"
#include "lbann/layers/misc/mini_batch_index.hpp"
#include "lbann/layers/misc/mini_batch_size.hpp"
#include "lbann/layers/misc/one_hot.hpp"
#include "lbann/layers/misc/rowwise_weights_norms.hpp"
#include "lbann/layers/misc/uniform_hash.hpp"
#include "lbann/layers/misc/variance.hpp"

#include "lbann/proto/datatype_helpers.hpp"

#include "lbann/utils/protobuf.hpp"

#ifdef LBANN_HAS_FFTW
#include "lbann/layers/misc/dft_abs.hpp"
#endif // LBANN_HAS_FFTW

#include "lbann/proto/layers.pb.h"

#include <memory>
#include <type_traits>

namespace lbann {
namespace {

template <typename T, El::Device D>
struct DFTTypeSupported : std::false_type
{
};

#ifdef LBANN_HAS_FFTW
#ifdef LBANN_HAS_FFTW_FLOAT
template <>
struct DFTTypeSupported<float, El::Device::CPU> : std::true_type
{
};
#endif // LBANN_HAS_FFTW_FLOAT
#ifdef LBANN_HAS_FFTW_DOUBLE
template <>
struct DFTTypeSupported<double, El::Device::CPU> : std::true_type
{
};
#endif // LBANN_HAS_FFTW_DOUBLE

#ifdef LBANN_HAS_GPU
template <>
struct DFTTypeSupported<float, El::Device::GPU> : std::true_type
{
};
template <>
struct DFTTypeSupported<double, El::Device::GPU> : std::true_type
{
};
#endif // LBANN_HAS_GPU

template <typename T,
          El::Device D,
          typename = El::EnableWhen<DFTTypeSupported<T, D>>>
std::unique_ptr<Layer> build_dft_layer(lbann_comm* comm)
{
  return std::make_unique<dft_abs_layer<T, D>>(comm);
}
#endif // LBANN_HAS_FFTW

template <typename T,
          El::Device D,
          typename = El::EnableUnless<DFTTypeSupported<T, D>>,
          typename = void>
std::unique_ptr<Layer> build_dft_layer(lbann_comm const* const)
{
  LBANN_ERROR("No FFT support for ",
              El::TypeName<T>(),
              " on device ",
              El::DeviceName<D>());
  return nullptr;
}

template <typename T, data_layout L, El::Device D>
struct UniformHashBuilder
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&...)
  {
    LBANN_ERROR(
      "Attempted to construct uniform_hash_layer with invalid parameters: "
      "(T=",
      El::TypeName<T>(),
      ", L=",
      to_string(L),
      ", D=",
      to_string(D),
      ")");
    return nullptr;
  }
};

#ifdef LBANN_HAS_GPU
template <data_layout L>
struct UniformHashBuilder<float, L, El::Device::GPU>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    using LayerType = uniform_hash_layer<float, L, El::Device::GPU>;
    return std::make_unique<LayerType>(std::forward<Args>(args)...);
  }
};
#endif // LBANN_HAS_GPU

#ifdef LBANN_HAS_GPU
template <data_layout L>
struct UniformHashBuilder<double, L, El::Device::GPU>
{
  template <typename... Args>
  static std::unique_ptr<Layer> Build(Args&&... args)
  {
    using LayerType = uniform_hash_layer<double, L, El::Device::GPU>;
    return std::make_unique<LayerType>(std::forward<Args>(args)...);
  }
};
#endif // LBANN_HAS_GPU

} // namespace
} // namespace lbann

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_argmax_layer_from_pbuf(lbann_comm* comm,
                                    lbann_data::Layer const& /*proto_layer*/)
{
  if constexpr (L == data_layout::DATA_PARALLEL && D == El::Device::CPU)
    return std::make_unique<
      argmax_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
  else {
    (void)comm;
    LBANN_ERROR(
      "argmax layer is only supported with a data-parallel layout and on CPU");
    return nullptr;
  }
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_argmin_layer_from_pbuf(lbann_comm* comm,
                                    lbann_data::Layer const& /*proto_layer*/)
{
  if constexpr (L == data_layout::DATA_PARALLEL && D == El::Device::CPU)
    return std::make_unique<
      argmin_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
  else {
    (void)comm;
    LBANN_ERROR(
      "argmin layer is only supported with a data-parallel layout and on CPU");
    return nullptr;
  }
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer> lbann::build_channelwise_mean_layer_from_pbuf(
  lbann_comm* comm,
  lbann_data::Layer const& proto_layer)
{
  if constexpr (L == data_layout::DATA_PARALLEL) {
    return std::make_unique<
      channelwise_mean_layer<T, data_layout::DATA_PARALLEL, D>>(comm);
  }
  else {
    (void)comm;
    LBANN_ERROR("channel-wise mean layer is only supported with "
                "a data-parallel layout");
    return nullptr;
  }
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer> lbann::build_channelwise_softmax_layer_from_pbuf(
  lbann_comm* comm,
  lbann_data::Layer const& proto_layer)
{
  auto const& layer = proto_layer.channelwise_softmax();
  if constexpr (L == data_layout::DATA_PARALLEL) {
    if constexpr (std::is_same_v<T, float>)
      return std::make_unique<
        channelwise_softmax_layer<float, data_layout::DATA_PARALLEL, D>>(
        comm,
        layer.dim(),
        layer.single_dim_mode());
    else if constexpr (std::is_same_v<T, double>)
      return std::make_unique<
        channelwise_softmax_layer<double, data_layout::DATA_PARALLEL, D>>(
        comm,
        layer.dim(),
        layer.single_dim_mode());
  }
  (void)comm;
  LBANN_ERROR("Attempted to construct channelwise_softmax_layer ",
              "with invalid parameters ",
              "(TensorDataType=",
              TypeName<T>(),
              ", ",
              "Layout=",
              to_string(L),
              ", ",
              "Device=",
              to_string(D),
              ")");
  return nullptr;
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_covariance_layer_from_pbuf(lbann_comm* comm,
                                        lbann_data::Layer const& proto_layer)
{
  return std::make_unique<covariance_layer<T, L, D>>(
    comm,
    proto_layer.covariance().biased());
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_dft_abs_layer_from_pbuf(lbann_comm* comm, lbann_data::Layer const&)
{
  if constexpr (L == data_layout::DATA_PARALLEL)
    return build_dft_layer<T, D>(comm);
  else {
    (void)comm;
    LBANN_ERROR("dft_abs layers are only supported in DATA_PARALLEL. "
                "Requested layout: ",
                to_string(L));
    return nullptr;
  }
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_external_layer_from_pbuf(lbann_comm* comm,
                                      lbann_data::Layer const& proto_layer)
{
  lbann::external_layer_setup_t setupfunc =
    load_external_library(proto_layer.external().filename(),
                          proto_layer.external().layer_name());

  lbann::Layer* layer =
    setupfunc(lbann::proto::TypeToProtoDataType<T>::value, L, D, comm);
  if (!layer) {
    LBANN_ERROR("External layer \"",
                proto_layer.external().filename(),
                "\" could not be initialized with the chosen configuration.");
    return nullptr;
  }

  return std::unique_ptr<lbann::Layer>(layer);
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_mini_batch_index_layer_from_pbuf(lbann_comm* comm,
                                              lbann_data::Layer const&)
{
  return std::make_unique<lbann::mini_batch_index_layer<T, L, D>>(comm);
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_mini_batch_size_layer_from_pbuf(lbann_comm* comm,
                                             lbann_data::Layer const&)
{
  return std::make_unique<lbann::mini_batch_size_layer<T, L, D>>(comm);
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_one_hot_layer_from_pbuf(lbann_comm*,
                                     lbann_data::Layer const& proto_layer)
{
  return std::make_unique<lbann::one_hot_layer<T, L, D>>(
    proto_layer.one_hot().size());
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_rowwise_weights_norms_layer_from_pbuf(lbann_comm*,
                                                   lbann_data::Layer const&)
{
  return std::make_unique<lbann::rowwise_weights_norms_layer<T, L, D>>();
}

template <typename TensorDataType, lbann::data_layout Layout, El::Device Device>
std::unique_ptr<lbann::Layer>
lbann::build_uniform_hash_layer_from_pbuf(lbann_comm* comm,
                                          lbann_data::Layer const&)
{
  using BuilderType = UniformHashBuilder<TensorDataType, Layout, Device>;
  return BuilderType::Build(comm);
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_variance_layer_from_pbuf(lbann_comm* comm,
                                      lbann_data::Layer const& proto_layer)
{
  return std::make_unique<variance_layer<T, L, D>>(
    comm,
    proto_layer.variance().biased());
}

namespace lbann {

#define PROTO_DEVICE(T, Device)                                                \
  LBANN_LAYER_BUILDER_ETI(argmax, T, Device);                                  \
  LBANN_LAYER_BUILDER_ETI(argmin, T, Device);                                  \
  LBANN_LAYER_BUILDER_ETI(channelwise_mean, T, Device);                        \
  LBANN_LAYER_BUILDER_ETI(channelwise_softmax, T, Device);                     \
  LBANN_LAYER_BUILDER_ETI(covariance, T, Device);                              \
  LBANN_LAYER_BUILDER_ETI(dft_abs, T, Device);                                 \
  LBANN_LAYER_BUILDER_ETI(external, T, Device);                                \
  LBANN_LAYER_BUILDER_ETI(mini_batch_index, T, Device);                        \
  LBANN_LAYER_BUILDER_ETI(mini_batch_size, T, Device);                         \
  LBANN_LAYER_BUILDER_ETI(one_hot, T, Device);                                 \
  LBANN_LAYER_BUILDER_ETI(rowwise_weights_norms, T, Device);                   \
  LBANN_LAYER_BUILDER_ETI(uniform_hash, T, Device);                            \
  LBANN_LAYER_BUILDER_ETI(variance, T, Device)
#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
