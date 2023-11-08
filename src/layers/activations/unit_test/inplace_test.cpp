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

#include "Catch2BasicSupport.hpp"

#include "MPITestHelpers.hpp"
#include "TestHelpers.hpp"

#include <lbann/base.hpp>
#include <lbann/layers/activations/relu.hpp>
#include <lbann/layers/data_type_layer.hpp>
#include <lbann/layers/transform/dummy.hpp>
#include <lbann/metrics/metric.hpp>
#include <lbann/objective_functions/objective_function.hpp>
#include <lbann/proto/lbann.pb.h>
#include <lbann/proto/proto_common.hpp>

#include <h2/patterns/multimethods/SwitchDispatcher.hpp>
#include <lbann/utils/lbann_library.hpp>
#include <lbann/utils/memory.hpp>
#include <lbann/utils/serialize.hpp>

// Prototext graphs
const std::string boilerplate_header = R"""(
model {
  layer {
    name: "inp"
    children: "layer1"
    weights: "dummy_inputs"
    weights_layer {
      dims: 3
    }
  }
)""";

const std::string boilerplate_footer = R"""(
  weights {
    name: "dummy_inputs"
    initializer {
      value_initializer {
        values: -1.2
        values: 3.4
        values: -5.67
      }
    }
  }
}
)""";

std::unique_ptr<lbann::model> setup_model(const std::string& model_contents)
{
  auto& world_comm = unit_test::utilities::current_world_comm();
  auto& g = world_comm.get_trainer_grid();
  lbann::utils::grid_manager mgr(g);

  lbann_data::LbannPB pb;
  REQUIRE_NOTHROW(lbann::read_prototext_string(model_contents, pb, true));

  // Construct a trainer so that the model can register the input layer
  lbann::construct_trainer(&world_comm, pb.mutable_trainer(), pb);
  auto my_model = lbann::proto::construct_model(&world_comm,
                                                pb.optimizer(),
                                                pb.trainer(),
                                                pb.model());
  // Setup the model with a minibatch size of 1
  my_model->setup(1UL, {&g});

  return my_model;
}

const std::string one_layer = boilerplate_header + R"""(
  layer {
    name: "layer1"
    parents: "inp"
    relu {
    }
  }
)""" + boilerplate_footer;

TEST_CASE("Simple in-place test", "[layer][inplace]")
{
#ifdef LBANN_HAS_GPU
  using reluT = lbann::
    relu_layer<float, lbann::data_layout::DATA_PARALLEL, El::Device::GPU>;
  using dummyT = lbann::
    dummy_layer<float, lbann::data_layout::DATA_PARALLEL, El::Device::GPU>;
  using MatrixT =
    El::DistMatrix<float, El::STAR, El::STAR, El::ELEMENT, El::Device::GPU>;
#else
  using reluT = lbann::
    relu_layer<float, lbann::data_layout::DATA_PARALLEL, El::Device::CPU>;
  using dummyT = lbann::
    dummy_layer<float, lbann::data_layout::DATA_PARALLEL, El::Device::CPU>;
  using MatrixT =
    El::DistMatrix<float, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU>;
#endif

  auto my_model = setup_model(one_layer);

  // Get the ReLU layer and ensure it is in-place
  auto& layer = my_model->get_layer(1);
  REQUIRE(layer.get_type() == "ReLU");
  REQUIRE(layer.runs_inplace());

  // Run through the model and ensure the activations
  // are propagated correctly
  REQUIRE_NOTHROW(my_model->forward_prop(lbann::execution_mode::training));

  reluT* relu = dynamic_cast<reluT*>(&layer);
  REQUIRE(relu != nullptr);

  // Check activations
  auto const& act = relu->get_activations();
  CHECK(act.Get(0, 0) == 0.0f);
  CHECK(fabs(act.Get(1, 0) - 3.4f) <= 1e-6);
  CHECK(act.Get(2, 0) == 0.0f);

  // Run backpropagation and check gradients:
  // Create a sample error signal
  auto& world_comm = unit_test::utilities::current_world_comm();
  auto& g = world_comm.get_trainer_grid();
  auto error_signal = std::make_unique<MatrixT>(3, 1, g);
  El::Fill(*error_signal, 9.0f);

  // Set error signal on dummy layer and keep error signal for verification
  auto& layer2 = my_model->get_layer(2);
  dummyT* dummy = dynamic_cast<dummyT*>(&layer2);
  REQUIRE(dummy != nullptr);
  dummy->set_error_signal(std::move(error_signal));
  relu->set_keep_error_signals(true);

  // Run backpropagation
  REQUIRE_NOTHROW(my_model->backward_prop(false));

  // Check gradients for correctness
  auto const& grads = relu->get_error_signals();
  CHECK(grads.Get(0, 0) == 0.0f);
  CHECK(fabs(grads.Get(1, 0) - 9.0f) <= 1e-6);
  CHECK(grads.Get(2, 0) == 0.0f);
}

const std::string cannot_run_inplace = boilerplate_header + R"""(
layer {
  name: "layer1"
  parents: "inp"
  children: "relu"
  pooling {
    pool_mode: "max"
    num_dims: 2
    has_vectors: false
    pool_dims_i: 2
    pool_pads_i: 0
    pool_strides_i: 2
  }
}
layer {
  name: "relu"
  parents: "layer1"
  relu {
  }
}
)""" + boilerplate_footer;

TEST_CASE("In-place gradient-activation dependency", "[layer][inplace]")
{
  auto my_model = setup_model(cannot_run_inplace);

  // Get the ReLU layer and ensure it is not in-place
  const auto& relu_layer = my_model->get_layer(2);
  REQUIRE(relu_layer.get_type() == "ReLU");
  REQUIRE(!relu_layer.runs_inplace());
}

const std::string chain = boilerplate_header + R"""(
layer {
  name: "layer1"
  parents: "inp"
  relu {
  }
}
layer {
  name: "addconst1"
  parents: "layer1"
  children: "addconst2"
  operator_layer {
    ops {
      parameters {
        type_url: "type.googleapis.com/lbann_data.AddConstantOperator"
        value: "\t\000\000\000\000\000\000\360?"
      }
    }
  }
}
layer {
  name: "addconst2"
  parents: "addconst1"
  children: "relu"
  operator_layer {
    ops {
      parameters {
        type_url: "type.googleapis.com/lbann_data.AddConstantOperator"
        value: "\t\000\000\000\000\000\000\360?"
      }
    }
  }
}
layer {
  name: "relu"
  parents: "addconst2"
  relu {
  }
}
)""" + boilerplate_footer;

TEST_CASE("Chained in-place layers", "[layer][inplace]")
{
  auto my_model = setup_model(chain);

  // Get the intermediate layers and ensure they are correctly in-place

  // ReLU followed by an in-place layer means that the succeeding layer
  // must not be in-place
  auto& layer1 = my_model->get_layer(1);
  REQUIRE(layer1.get_type() == "ReLU");
  REQUIRE(layer1.runs_inplace());
  auto& layer2 = my_model->get_layer(2);
  REQUIRE(layer2.get_type() == "operator");
  REQUIRE(!layer2.runs_inplace());

  // Two chained in-place operators (addconstant, relu) are ok to run in-place
  auto& layer3 = my_model->get_layer(3);
  REQUIRE(layer3.get_type() == "operator");
  REQUIRE(layer3.runs_inplace());
  auto& layer4 = my_model->get_layer(4);
  REQUIRE(layer4.get_type() == "ReLU");
  REQUIRE(layer4.runs_inplace());
}
