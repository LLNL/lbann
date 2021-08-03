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

#include "lbann/layers/activations/activations.hpp"
#include "lbann/layers/activations/elu.hpp"
#include "lbann/layers/activations/leaky_relu.hpp"
#include "lbann/layers/activations/log_softmax.hpp"
#include "lbann/layers/activations/relu.hpp"
#include "lbann/layers/activations/softmax.hpp"
#include "lbann/layers/math/unary.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/random.hpp"

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
#ifdef LBANN_HAS_GPU
  constexpr El::Device Dev = El::Device::GPU;
#else
  constexpr El::Device Dev = El::Device::CPU;
#endif

  std::unique_ptr<Layer> layer;

  if (new_type == "relu") {
    layer =
      std::make_unique<relu_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(
        &comm);
  }
  else if (new_type == "tanh") {
    layer =
      std::make_unique<tanh_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(
        &comm);
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
      sigmoid_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(&comm);
  }
  else {
    LBANN_ERROR("Unknown new layer type: ", new_type);
  }
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
  int const type_index =
    fast_rand_int(get_fast_generator(), activation_types.size());
  int const name_index =
    fast_rand_int(get_fast_generator(), activation_layer_names.size());

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

} // namespace ltfb
} // namespace lbann
