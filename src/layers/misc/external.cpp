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

#define LBANN_EXTERNAL_LAYER_INSTANTIATE
#include "lbann/layers/misc/external.hpp"

#include <algorithm>
#include <cstdio>
#include <dlfcn.h>

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
external_layer<TensorDataType, Layout, Device>::external_layer(
  lbann_comm* comm,
  const std::string& fp_name,
  const std::string& bp_name,
  std::string layer_name)
  : data_type_layer<TensorDataType>(comm)
{
  ////////////////////////////////////////////////////
  // Load the library handles dynamically using dlopen
  this->fp_handle = dlopen(fp_name.c_str(), RTLD_LAZY);
  if (!this->fp_handle) {
    LBANN_ERROR(
      "Cannot load library for external layer forward pass (filename: \"",
      fp_name,
      "\"). Reason: ",
      dlerror());
    return;
  }

  std::string bp_filename = bp_name;
  if (bp_name.length() > 0) { // Explicit backprop filename defined
    this->bp_handle = dlopen(bp_name.c_str(), RTLD_LAZY);
    if (!this->bp_handle) {
      LBANN_ERROR(
        "Cannot load library for external layer backprop (filename: \"",
        bp_name,
        "\"). Reason: ",
        dlerror());
      return;
    }
  }
  else {
    // Reuse fprop library
    this->bp_handle = this->fp_handle;
    bp_filename = fp_name;
  }

  ////////////////////////////////////////////////////
  // Collect functions from libraries based on device
  if (layer_name.length() == 0)
    layer_name = "layer";

  std::string fp_funcname =
    std::string(Device == El::Device::CPU ? "cpu_fprop_compute_"
                                          : "gpu_fprop_compute_") +
    layer_name;
  this->fp_compute_ptr = (fprop_t)dlsym(this->fp_handle, fp_funcname.c_str());
  if (!this->fp_compute_ptr) {
    LBANN_ERROR("Malformed external library (filename: \"",
                fp_name,
                "\"). Reason: Missing function \"",
                fp_funcname,
                "\"");
    return;
  }
  std::string bp_funcname =
    std::string(Device == El::Device::CPU ? "cpu_bprop_compute_"
                                          : "gpu_bprop_compute_") +
    layer_name;
  this->bp_compute_ptr = (bprop_t)dlsym(this->bp_handle, bp_funcname.c_str());
  if (!this->bp_compute_ptr) {
    LBANN_ERROR("Malformed external library (filename: \"",
                bp_filename,
                "\"). Reason: Missing function \"",
                bp_funcname,
                "\"");
    return;
  }

  ////////////////////////////////////////////////////
  // TODO: Setup/initalize library (comm, layout, config, accelerator library
  // handles etc.)
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
external_layer<TensorDataType, Layout, Device>::~external_layer()
{
  // Close loaded libraries
  if (this->bp_handle && this->bp_handle != this->fp_handle)
    dlclose(this->bp_handle);
  this->bp_handle = nullptr;

  if (this->fp_handle)
    dlclose(this->fp_handle);
  this->fp_handle = nullptr;
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void external_layer<TensorDataType, Layout, Device>::fp_compute()
{
  using MatType = El::Matrix<TensorDataType, Device>;

  // Obtain layer metadata
  auto ninputs = this->get_num_parents();
  auto noutputs = this->get_num_children();
  const auto& weight_ptrs = this->get_weights_pointers();
  auto nweights = weight_ptrs.size();

  // Create vectors for call. TODO: Can be cached on object.
  std::vector<void*> inputs(ninputs, nullptr), weights(nweights, nullptr),
    outputs(noutputs, nullptr);

  // Set arguments
  for (auto i = 0; i < ninputs; ++i) {
    const auto& local_input =
      dynamic_cast<const MatType&>(this->get_local_prev_activations(i));
    inputs[i] = (void *)local_input.LockedBuffer();
  }
  /*for (auto i = 0U; i < nweights; ++i) {
    // TODO: this can likely be cached at setup
    weights[i] = (void *)weight_ptrs[i].lock().get_values().Buffer();
  }*/
  for (auto i = 0; i < noutputs; ++i) {
    auto& local_output = dynamic_cast<MatType&>(this->get_local_activations(i));
    outputs[i] = (void *)local_output.Buffer();
  }

  // Invoke computation in external library
  this->fp_compute_ptr(inputs, weights, outputs);
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void external_layer<TensorDataType, Layout, Device>::bp_compute()
{
  using MatType = El::Matrix<TensorDataType, Device>;

  // Obtain layer metadata
  auto ninputs = this->get_num_parents();
  auto noutputs = this->get_num_children();
  const auto& weight_ptrs = this->get_weights_pointers();
  auto nweights = weight_ptrs.size();

  // Create vectors for call. TODO: Can be cached on object.
  std::vector<void*> inputs(ninputs, nullptr), prev_error_signals(noutputs, nullptr),
    output_error_signals(ninputs, nullptr), weight_grads(nweights, nullptr);

  // Set arguments
  for (auto i = 0; i < ninputs; ++i) {
    const auto& local_input =
      dynamic_cast<const MatType&>(this->get_local_prev_activations(i));
    auto& local_error =
      dynamic_cast<MatType&>(this->get_local_error_signals(i));
    inputs[i] = (void *)local_input.LockedBuffer();
    output_error_signals[i] = (void *)local_error.Buffer();
  }
  for (auto i = 0; i < noutputs; ++i) {
    const auto& local_prev_error = dynamic_cast<const MatType&>(this->get_local_prev_error_signals(i));
    prev_error_signals[i] = (void *)local_prev_error.LockedBuffer();
  }
  // TODO: Gradients w.r.t. weights

  // Invoke computation in external library
  this->bp_compute_ptr(inputs, prev_error_signals, output_error_signals, weight_grads);
}

#define PROTO_DEVICE(T, Device)                                                \
  template class external_layer<T, data_layout::DATA_PARALLEL, Device>;        \
  template class external_layer<T, data_layout::MODEL_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
