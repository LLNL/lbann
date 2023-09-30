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

#include <lbann/lbann.hpp>

#ifdef LBANN_HAS_GPU
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
typedef hipStream_t gpuStream_t;
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#else
#include <cuda_runtime.h>
typedef cudaStream_t gpuStream_t;
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#endif

/*
Helper functions to extract POD type streams out of CPU and GPU sync infos
*/
template <El::Device D>
gpuStream_t to_native_stream_(El::SyncInfo<D> const& m) noexcept;

template <>
gpuStream_t to_native_stream_(El::SyncInfo<El::Device::GPU> const& m) noexcept
{
  El::SyncInfo<El::Device::GPU> const& si = m;
  return si.Stream();
}

template <>
gpuStream_t to_native_stream_(El::SyncInfo<El::Device::CPU> const& m) noexcept
{
  return nullptr;
}

template <El::Device D, El::Device... args>
gpuStream_t to_native_stream(El::MultiSync<D, args...> const& m) noexcept
{
  El::SyncInfo<D> const& si = m;
  return to_native_stream_(si);
}
#endif // LBANN_HAS_GPU

/**
 * Sample external layer that performs the identity function with (gpu)memcpy.
 **/
template <typename TensorDataType, lbann::data_layout Layout, El::Device Device>
class my_identity_layer final : public lbann::data_type_layer<TensorDataType>
{
  using MatType = El::Matrix<TensorDataType, Device>;

public:
  my_identity_layer(lbann::lbann_comm* comm)
    : lbann::data_type_layer<TensorDataType>(comm)
  {}
  my_identity_layer* copy() const final { return new my_identity_layer(*this); }

  std::string get_type() const final { return "my_identity"; }
  lbann::data_layout get_data_layout() const final { return Layout; }
  El::Device get_device_allocation() const final { return Device; }
  void write_specific_proto(lbann_data::Layer& proto) const final
  {
    proto.set_datatype(lbann::proto::ProtoDataType<TensorDataType>);
    proto.mutable_external();
  }

private:
  friend class cereal::access;
  my_identity_layer() : my_identity_layer(nullptr) {}

  void setup_dims() final
  {
    lbann::data_type_layer<TensorDataType>::setup_dims();
    this->set_output_dims(this->get_input_dims());
  }

  void fp_compute() final
  {
    auto& local_input =
      dynamic_cast<const MatType&>(this->get_local_prev_activations());
    auto& local_output = dynamic_cast<MatType&>(this->get_local_activations());

    auto multisync = El::MakeMultiSync(El::SyncInfoFromMatrix(local_input),
                                       El::SyncInfoFromMatrix(local_output));

    if constexpr (Device == El::Device::CPU)
      memcpy(local_output.Buffer(),
             local_input.LockedBuffer(),
             sizeof(TensorDataType) * local_input.Width() *
               local_input.Height());
#ifdef LBANN_HAS_GPU
    else if constexpr (Device == El::Device::GPU)
      static_cast<void>(gpuMemcpyAsync(
        local_output.Buffer(),
        local_input.LockedBuffer(),
        sizeof(TensorDataType) * local_input.Width() * local_input.Height(),
        gpuMemcpyDeviceToDevice,
        to_native_stream(multisync)));
#endif // LBANN_HAS_GPU
  }

  void bp_compute() final
  {
    auto& local_input =
      dynamic_cast<const MatType&>(this->get_local_prev_error_signals());
    auto& local_output =
      dynamic_cast<MatType&>(this->get_local_error_signals());

    auto multisync = El::MakeMultiSync(El::SyncInfoFromMatrix(local_input),
                                       El::SyncInfoFromMatrix(local_output));

    if constexpr (Device == El::Device::CPU)
      memcpy(local_output.Buffer(),
             local_input.LockedBuffer(),
             sizeof(TensorDataType) * local_input.Width() *
               local_input.Height());
#ifdef LBANN_HAS_GPU
    else if constexpr (Device == El::Device::GPU)
      static_cast<void>(gpuMemcpyAsync(
        local_output.Buffer(),
        local_input.LockedBuffer(),
        sizeof(TensorDataType) * local_input.Width() * local_input.Height(),
        gpuMemcpyDeviceToDevice,
        to_native_stream(multisync)));
#endif // LBANN_HAS_GPU
  }
};

/**
 * Create the layer from the given type/layout/device configuration.
 * In the below sample, only float is supported.
 **/
extern "C" lbann::Layer* setup_layer(int datatype,
                                     lbann::data_layout layout,
                                     El::Device device,
                                     lbann::lbann_comm* comm)
{
  if (datatype == lbann_data::FLOAT &&
      layout == lbann::data_layout::DATA_PARALLEL && device == El::Device::CPU)
    return new my_identity_layer<float,
                                 lbann::data_layout::DATA_PARALLEL,
                                 El::Device::CPU>(comm);

#ifdef LBANN_HAS_GPU
  if (datatype == lbann_data::FLOAT &&
      layout == lbann::data_layout::DATA_PARALLEL && device == El::Device::GPU)
    return new my_identity_layer<float,
                                 lbann::data_layout::DATA_PARALLEL,
                                 El::Device::GPU>(comm);
#endif // LBANN_HAS_GPU

  // Unsupported configurations return nullptr
  return nullptr;
}
