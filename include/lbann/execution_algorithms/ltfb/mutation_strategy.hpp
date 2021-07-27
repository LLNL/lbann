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
#ifndef LBANN_EXECUTION_ALGORITHMS_LTFB_MUTATION_STRATEGY_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_LTFB_MUTATION_STRATEGY_HPP_INCLUDED

#include "lbann/utils/random.hpp"
#include "lbann/models/model.hpp"
#include "lbann/layers/math/unary.hpp"
#include "lbann/layers/activations/relu.hpp"
#include "lbann/layers/activations/softmax.hpp"
#include "lbann/layers/activations/elu.hpp"
#include "lbann/layers/activations/leaky_relu.hpp"
#include "lbann/layers/activations/log_softmax.hpp"

namespace lbann {
namespace ltfb {

class MutationStrategy : public Cloneable<HasAbstractFunction<MutationStrategy>>
{
public:
  MutationStrategy() {};
  virtual ~MutationStrategy() = default;

public:
  /** @brief Apply a change to the model.
   *  @param[in,out] m The model to change.
   *  @param[in] step The current execution step in LTFB
   */
  virtual void mutate(model& m, const int& step) = 0;
};

// No Mutation
class NullMutation : public Cloneable<NullMutation, MutationStrategy>
{

public:
  NullMutation() = default; 
  void mutate(model&, const int&) override {}
};

// Replace activation layers
class ReplaceActivation : public Cloneable<ReplaceActivation, MutationStrategy>
{
private:
  std::vector<std::string> m_activation_types = {"relu", "tanh", "softmax", "elu", "leaky relu", "log softmax"};

public:
  ReplaceActivation() = default;

  std::unique_ptr<lbann::Layer> make_new_activation_layer(lbann::lbann_comm& comm,
                                                          std::string const& new_type,
                                                          std::string const& new_name)
  {
    using DataType = float;
    #ifdef LBANN_HAS_GPU
      constexpr El::Device Dev = El::Device::GPU;
    #else
      constexpr El::Device Dev = El::Device::CPU;
    #endif

    std::unique_ptr<lbann::Layer> layer;

    if(new_type == "relu") {
       layer = std::make_unique<
           lbann::relu_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(&comm); 
    } else if (new_type == "tanh") {
       layer = std::make_unique<
            lbann::tanh_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(&comm);
    } else if (new_type == "softmax") {
       layer = std::make_unique<
            lbann::softmax_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(&comm, softmax_mode::INSTANCE);
    } else if (new_type == "elu") {
       layer = std::make_unique<
            lbann::elu_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(&comm, 1);
    } else if (new_type == "leaky relu") {
       layer = std::make_unique<
            lbann::leaky_relu_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(&comm, 0.01);
    } else if (new_type == "log softmax") {
       layer = std::make_unique<
            lbann::log_softmax_layer<DataType, data_layout::DATA_PARALLEL, Dev>>(&comm);
    } else {
       LBANN_ERROR("Unknown new layer type: ", new_type);
    }
    layer->set_name(new_name);
    return layer;
  }

  void mutate(model& m, const int& step)
  {
    std::vector<std::string> activation_layer_names;

    auto& comm = *m.get_comm();
    
    for (int i = 0; i < m.get_num_layers(); ++i)
    {
       // Creating a temp string with lower case representation
       std::string temp_type = m.get_layer(i).get_type();
       std::transform(temp_type.begin(), temp_type.end(),
                                         temp_type.begin(), ::tolower);    

       if (std::find(m_activation_types.cbegin(), m_activation_types.cend(), temp_type)
                                                              != m_activation_types.cend()) {
         activation_layer_names.push_back(m.get_layer(i).get_name());
       }
    }

    // Generate two random numbers - one for new layer type and one for old layer name
    int type_index = fast_rand_int(get_fast_generator(), m_activation_types.size());
    int name_index = fast_rand_int(get_fast_generator(), activation_layer_names.size());

    // Name of new layer
    std::string new_layer_name = "new_" + m_activation_types[type_index] 
                                        + std::to_string(step) + std::to_string(name_index);
      
    // Print mutation
    std::cout << "Replacing " << activation_layer_names[name_index] << " with " << new_layer_name << std::endl;
 
    // Replace layer
    m.replace_layer(make_new_activation_layer(comm, m_activation_types[type_index], new_layer_name), 
                    activation_layer_names[name_index]);

    // Erase old layer from activation layer list
    activation_layer_names.erase(activation_layer_names.cbegin() + name_index);

    // Add new layer to activation layer list
    activation_layer_names.push_back(new_layer_name);  
  }
};

} // namespace ltfb

} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_LTFB_MUTATION_STRATEGY_HPP_INCLUDED
