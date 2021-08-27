////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

#include "lbann/execution_algorithms/ltfb/mutation_strategy.hpp"

#include "lbann/comm_impl.hpp"

#include "lbann/layers/activations/elu.hpp"
#include "lbann/layers/activations/leaky_relu.hpp"
#include "lbann/layers/activations/log_softmax.hpp"
#include "lbann/layers/activations/relu.hpp"
#include "lbann/layers/activations/softmax.hpp"
#include "lbann/layers/operator_layer.hpp"

#include "lbann/operators/activations/activations.hpp" // SigmoidOperator
#include "lbann/operators/math/unary.hpp"              // TanhOperator

#include "lbann/layers/learning/convolution.hpp"

#include "lbann/models/model.hpp"
#include "lbann/operators/math/unary.hpp"
#include "lbann/utils/random.hpp"
#include "lbann_config.hpp"
#include <memory>

#ifdef LBANN_HAS_GPU
constexpr El::Device Dev = El::Device::GPU;
#else
constexpr El::Device Dev = El::Device::CPU;
#endif

namespace lbann {
namespace ltfb {
namespace {

/** @brief Construct a new activation layer.
 *  @param[in] comm The current communicator
 *  @param[in] new_type The type of the new activation layer (ReLU or tanh
 * etc)
 *  @param[in] new_name The name of the new activation layer
 */
std::unique_ptr<lbann::Layer>
make_new_activation_layer(lbann_comm& comm,
                          std::string const& new_type,
                          std::string const& new_name)
{
  std::unique_ptr<Layer> layer;

  if (new_type == "relu") {
    layer =
      std::make_unique<relu_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(
        &comm);
  }
  else if (new_type == "tanh") {
    layer = std::make_unique<
      OperatorLayer<DataType, DataType, data_layout::DATA_PARALLEL, Dev>>(
      comm,
      std::make_unique<TanhOperator<DataType, Dev>>());
  }
  else if (new_type == "softmax") {
    layer = std::make_unique<
      softmax_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(
      &comm,
      softmax_mode::INSTANCE);
  }
  else if (new_type == "elu") {
    layer =
      std::make_unique<elu_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(
        &comm,
        1);
  }
  else if (new_type == "leaky relu") {
    layer = std::make_unique<
      leaky_relu_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(&comm, 0.01);
  }
  else if (new_type == "log softmax") {
    layer = std::make_unique<
      log_softmax_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(&comm);
  }
  else if (new_type == "sigmoid") {
    layer = std::make_unique<
      OperatorLayer<DataType, DataType, data_layout::DATA_PARALLEL, Dev>>(
      comm,
      std::make_unique<SigmoidOperator<DataType, Dev>>());
  }
  else {
    LBANN_ERROR("Unknown new layer type: ", new_type);
  }
  layer->set_name(new_name);
  return layer;
}

/** @brief Construct a new convolution layer with padding adjusted
 *         such that the output height and width does not change.
 *  @param[in] old_kernel The old kernel size
 *  @param[in] old_pad The old padding
 *  @param[in] new_kernel The new kernel size
 *  @param[in] new_channels The new number of channels
 *  @param[in] stride Strides
 *  @param[in] dilation Dilations
 *  @param[in] new_name The name of the new activation layer
 */
std::unique_ptr<lbann::Layer>
make_new_convolution_layer(int const& old_kernel,
                           int const& old_pad,
                           int const& new_kernel,
                           int const& new_channels,
                           int const& stride,
                           int const& dilation,
                           std::string const& new_name)
{
  const int new_pad = old_pad + (new_kernel - old_kernel) / 2;
  std::vector<int> layer_kernel{new_kernel, new_kernel},
    layer_pads{new_pad, new_pad}, layer_strides{stride, stride},
    layer_dilations{dilation, dilation};
  auto layer = std::make_unique<
    lbann::convolution_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(
    2,
    new_channels,
    layer_kernel,
    layer_pads,
    layer_strides,
    layer_dilations,
    1,
    true);
  layer->set_name(new_name);
  return layer;
}

} // namespace

void ReplaceActivation::mutate(model& m, const int& step)
{
  static std::vector<std::string> const activation_types = {"relu",
                                                            "tanh",
                                                            "elu",
                                                            "sigmoid",
                                                            "leaky relu"};
  std::vector<std::string> activation_layer_names;

  auto& comm = *m.get_comm();

  for (int i = 0; i < m.get_num_layers(); ++i) {
    // Creating a temp string with lower case representation
    std::string temp_type = m.get_layer(i).get_type();
    std::transform(begin(temp_type),
                   end(temp_type),
                   begin(temp_type),
                   [](unsigned char c) { return ::tolower(c); });

    if (std::find(activation_types.cbegin(),
                  activation_types.cend(),
                  temp_type) != activation_types.cend()) {
      activation_layer_names.push_back(m.get_layer(i).get_name());
    }
  }

  // Generate two random numbers - one for new layer type and one for old layer
  // name
  int type_index;
  int name_index;

  // Generate the random numbers only in the master proc of trainer and
  // broadcast them so that all procs in a trainer are undergoing the same
  // mutation
  if (m.get_comm()->am_trainer_master()) {
    type_index = fast_rand_int(get_fast_generator(), activation_types.size());
    name_index =
      fast_rand_int(get_fast_generator(), activation_layer_names.size());
  }
  m.get_comm()->trainer_broadcast(m.get_comm()->get_trainer_master(),
                                  type_index);
  m.get_comm()->trainer_broadcast(m.get_comm()->get_trainer_master(),
                                  name_index);

  // Print mutation
  std::cout << "Changing type of activation layer "
            << activation_layer_names[name_index] << " to "
            << activation_types[type_index] << std::endl;

  // Replace layer
  m.replace_layer(make_new_activation_layer(comm,
                                            activation_types[type_index],
                                            activation_layer_names[name_index]),
                  activation_layer_names[name_index]);
}

void ReplaceConvolution::mutate(model& m, const int& step)
{
  static std::vector<int> const kernels = {3, 5, 7, 9};
  static std::vector<int> const channels = {6, 16, 32};

  std::vector<int> conv_layer_indices; // Indices of all convolution layers

  for (int i = 0; i < m.get_num_layers(); ++i) {
    // Creating a temp string with lower case representation
    std::string temp_type = m.get_layer(i).get_type();
    std::transform(begin(temp_type),
                   end(temp_type),
                   begin(temp_type),
                   [](unsigned char c) { return ::tolower(c); });

    std::string temp_name = m.get_layer(i).get_name();
    std::transform(begin(temp_name),
                   end(temp_name),
                   begin(temp_name),
                   [](unsigned char c) { return ::tolower(c); });

    // Find the indices of all convolution layers.
    // Ensure that convolution shim layers are not counted here
    if (temp_type == "convolution" &&
        temp_name.find("shim") == std::string::npos) {
      conv_layer_indices.push_back(i);
    }
  }

  // Generate three random numbers - one for new kernel size, one for new
  // channels and one for conv layer to choose
  int kernel_index;
  int channel_index;
  int conv_index;

  // Generate the random numbers only in the master proc of trainer and
  // broadcast them so that all procs in a trainer are undergoing the same
  // mutation
  if (m.get_comm()->am_trainer_master()) {
    kernel_index = fast_rand_int(get_fast_generator(), kernels.size());
    channel_index = fast_rand_int(get_fast_generator(), channels.size());
    conv_index = fast_rand_int(get_fast_generator(), conv_layer_indices.size());
  }
  m.get_comm()->trainer_broadcast(m.get_comm()->get_trainer_master(),
                                  kernel_index);
  m.get_comm()->trainer_broadcast(m.get_comm()->get_trainer_master(),
                                  channel_index);
  m.get_comm()->trainer_broadcast(m.get_comm()->get_trainer_master(),
                                  conv_index);

  // Print mutation
  std::cout << "Changing convolution layer "
            << m.get_layer(conv_layer_indices[conv_index]).get_name()
            << ": kernel -  " << kernels[kernel_index] << ": channels - "
            << channels[channel_index] << std::endl;

  // Get old specifics of layer
  auto& layer = m.get_layer(conv_layer_indices[conv_index]);
  using base_conv =
    lbann::convolution_layer<DataType, data_layout::DATA_PARALLEL, Dev>;
  auto& cast_layer = dynamic_cast<base_conv&>(layer);
  const int old_kernel = cast_layer.get_conv_dims()[0];
  const int old_pad = cast_layer.get_pads()[0];
  const int old_strides = cast_layer.get_strides()[0];
  const int old_dilations = cast_layer.get_dilations()[0];
  const int old_channels = cast_layer.get_output_dims(0)[0];
  const std::string name = cast_layer.get_name();

  // Get the child of the convolution layer before replacing it
  auto& child = layer.get_child_layer(0);

  // Replace the convolution layer
  m.replace_layer(make_new_convolution_layer(old_kernel,
                                             old_pad,
                                             kernels[kernel_index],
                                             channels[channel_index],
                                             old_strides,
                                             old_dilations,
                                             name),
                  name);

  // Find out if there is a shim layer.
  // If there, replace it to match channels appropriately
  // If not, insert a 1x1 conv shim layer to match channels
  if (child.get_name().find("shim") != std::string::npos) {
    std::cout << "Replacing shim of layer " << name << std::endl;
    const std::string shim_layer_name = child.get_name();

    // Get out channels of shim
    auto shim_channels = child.get_output_dims(0)[0];

    // Replace shim layer: O/P channels stay the same but I/P channels change
    m.replace_layer(
      make_new_convolution_layer(1, 0, 1, shim_channels, 1, 1, shim_layer_name),
      shim_layer_name);
  }
  else {
    std::cout << "Creating shim layer for layer " << name << std::endl;

    const std::string shim_layer_name = name + "_shim";

    // old_channels should be equal to shim_channels since
    // this block should get executed only once
    m.insert_layer(
      make_new_convolution_layer(1, 0, 1, old_channels, 1, 1, shim_layer_name),
      name);
  }
}

void HybridMutation::mutate(model& m, const int& step)
{
  // Generate a random number to alternate between ReplaceActivation and ReplaceConvolution
  int mutation_choice; // 0 - ReplaceActivation, 1 - ReplaceConvolution

  if (m.get_comm()->am_trainer_master()) {
    mutation_choice = fast_rand_int(get_fast_generator(), 2); // either 0 or 1
  }
  m.get_comm()->trainer_broadcast(m.get_comm()->get_trainer_master(),
                                  mutation_choice);

  ReplaceActivation *RA = new ReplaceActivation();
  ReplaceConvolution *RC = new ReplaceConvolution();

  if (mutation_choice == 0) {
    RA->mutate(m, step);
  }
  else {
    RC->mutate(m, step);
  }    
}

} // namespace ltfb
} // namespace lbann
