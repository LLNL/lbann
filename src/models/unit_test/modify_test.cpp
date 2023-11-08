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

#include "lbann/layers/activations/relu.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include <lbann/base.hpp>
#include <lbann/models/model.hpp>
#include <lbann/utils/lbann_library.hpp>

#include "lbann/proto/lbann.pb.h"
#include <google/protobuf/text_format.h>

using namespace lbann;

namespace pb = ::google::protobuf;

namespace {
// model_prototext string is defined here as a "const std::string".
#include "lenet.prototext.inc"

auto make_model(lbann::lbann_comm& comm)
{
  lbann_data::LbannPB my_proto;
  if (!pb::TextFormat::ParseFromString(model_prototext, &my_proto))
    throw "Parsing protobuf failed.";
  // Construct a trainer so that the model can register the input layer
  auto& trainer =
    lbann::construct_trainer(&comm, my_proto.mutable_trainer(), my_proto);
  unit_test::utilities::mock_data_reader(trainer, {1, 28, 28}, 10);
  auto my_model = lbann::proto::construct_model(&comm,
                                                my_proto.optimizer(),
                                                my_proto.trainer(),
                                                my_proto.model());
  my_model->setup(1UL, {&comm.get_trainer_grid()});
  return my_model;
}

std::unique_ptr<lbann::Layer> make_new_relu_layer(lbann::lbann_comm& comm)
{
  auto layer = std::make_unique<
    lbann::relu_layer<float, data_layout::DATA_PARALLEL, El::Device::CPU>>(
    &comm);
  layer->set_name("new_relu");
  return layer;
}

bool has_layer(model const& m, std::string const& layer_name)
{
  auto layers = m.get_layers();
  auto iter =
    std::find_if(cbegin(layers), cend(layers), [&layer_name](auto const& l) {
      return l->get_name() == layer_name;
    });
  return (iter != cend(layers));
}

bool is_child(model const& m,
              std::string const& parent_layer_name,
              std::string const& child_layer_name)
{
  auto layers = m.get_layers();
  auto iter =
    std::find_if(cbegin(layers),
                 cend(layers),
                 [&parent_layer_name, &child_layer_name](auto const& l) {
                   return l->get_name() == parent_layer_name &&
                          l->get_num_children() > 0 &&
                          l->get_child_layer(0).get_name() == child_layer_name;
                 });
  return (iter != cend(layers));
}

} // namespace

TEST_CASE("Layer Insertion", "[mpi][model][dag]")
{
  auto& comm = unit_test::utilities::current_world_comm();
  std::unique_ptr<lbann::model> m = make_model(comm);

  SECTION("Attempting insertion after non-existent layer")
  {
    REQUIRE_THROWS(
      m->insert_layer(make_new_relu_layer(comm), "this layer doesn't exist"));

    // Verify that the new layer was not added
    CHECK_FALSE(has_layer(*m, "new_relu"));
  }

  SECTION("Inserting a valid layer")
  {
    REQUIRE_NOTHROW(m->insert_layer(make_new_relu_layer(comm), "layer5"));
    REQUIRE_NOTHROW(m->setup(1UL, {&comm.get_trainer_grid()}));

    // Verify the new layer is there
    CHECK(has_layer(*m, "new_relu"));

    // Verify the new layer is child of layer after which it is inserted
    CHECK(is_child(*m, "layer5", "new_relu"));
  }
}

TEST_CASE("Layer Removal", "[mpi][model][dag]")
{
  auto& comm = unit_test::utilities::current_world_comm();
  std::unique_ptr<lbann::model> m = make_model(comm);

  SECTION("Attempting removal of non-existent layer")
  {
    REQUIRE_THROWS(m->remove_layer("this layer doesn't exist"));

    // Verify the layer is still there
    CHECK(has_layer(*m, "layer5"));
  }

  SECTION("Removing a valid layer")
  {
    REQUIRE_NOTHROW(m->remove_layer("layer5"));
    REQUIRE_NOTHROW(m->setup(1UL, {&comm.get_trainer_grid()}));

    // Verify the layer is removed
    CHECK_FALSE(has_layer(*m, "layer5"));
  }
}

TEST_CASE("Layer Replacement", "[mpi][model][dag]")
{
  auto& comm = unit_test::utilities::current_world_comm();
  std::unique_ptr<lbann::model> m = make_model(comm);

  SECTION("Attempting replacement of non-existent layer")
  {
    REQUIRE_THROWS(
      m->replace_layer(make_new_relu_layer(comm), "this layer doesn't exist"));

    // Verify that the new layer was not added
    CHECK_FALSE(has_layer(*m, "new_relu"));
  }

  SECTION("Replacing a valid layer")
  {
    REQUIRE_NOTHROW(m->replace_layer(make_new_relu_layer(comm), "layer5"));
    REQUIRE_NOTHROW(m->setup(1UL, {&comm.get_trainer_grid()}));

    // Verify the new layer is there
    CHECK(has_layer(*m, "new_relu"));

    // Verify the old layer is gone
    CHECK_FALSE(has_layer(*m, "layer5"));
  }
}
