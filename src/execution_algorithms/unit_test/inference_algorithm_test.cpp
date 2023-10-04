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

#include "lbann/objective_functions/objective_function.hpp"
#include <lbann/base.hpp>
#include <lbann/execution_algorithms/batch_functional_inference_algorithm.hpp>
#include <lbann/models/model.hpp>
#include <lbann/utils/lbann_library.hpp>

#include "lbann/proto/lbann.pb.h"
#include <google/protobuf/text_format.h>

namespace pb = ::google::protobuf;

namespace {
// This model is just an input layer into a softmax layer, so we can verify the
// output is correct for a simple input (e.g., a matrix filled with 1.0)
std::string const model_prototext = R"ptext(
model {
  layer {
    name: "layer1"
    children: "layer2"
    input {
      data_field: "samples"
    }
  }
  layer {
    name: "layer2"
    parents: "layer1"
    softmax {
    }
  }
}
)ptext";

template <typename T>
auto make_model(lbann::lbann_comm& comm, int class_n)
{
  lbann_data::LbannPB my_proto;
  if (!pb::TextFormat::ParseFromString(model_prototext, &my_proto))
    throw "Parsing protobuf failed.";
  // Construct a trainer so that the model can register the input layer
  auto& trainer =
    lbann::construct_trainer(&comm, my_proto.mutable_trainer(), my_proto);
  unit_test::utilities::mock_data_reader(trainer, {1, 1, class_n}, class_n);
  auto my_model = lbann::proto::construct_model(&comm,
                                                my_proto.optimizer(),
                                                my_proto.trainer(),
                                                my_proto.model());
  my_model->setup(1UL, {&comm.get_trainer_grid()});
  return my_model;
}

} // namespace

TEST_CASE("Test batch_function_inference_algorithm", "[inference]")
{
  using DataType = float;
  DataType one = 1.;
  DataType zero = 0.;
  int mbs_class_n = 4;

  auto& comm = unit_test::utilities::current_world_comm();
  std::unique_ptr<lbann::model> model = make_model<DataType>(comm, mbs_class_n);
  auto const& g = comm.get_trainer_grid();
  El::DistMatrix<DataType, El::STAR, El::STAR, El::ELEMENT, El::Device::CPU>
    data(mbs_class_n, mbs_class_n, g);
  auto inf_alg = lbann::batch_functional_inference_algorithm();

  SECTION("Model data insert and forward prop")
  {
    El::Fill(data, one);

    inf_alg.infer(model.get(), data, mbs_class_n);
    const auto* l = model->get_layers()[1];
    auto const& dtl = dynamic_cast<lbann::data_type_layer<float> const&>(*l);
    const auto& output = dtl.get_activations();

    for (int i = 0; i < output.Height(); i++) {
      for (int j = 0; j < output.Width(); j++) {
        REQUIRE(output.Get(i, j) == Approx(1.0 / mbs_class_n));
      }
    }
  }

  SECTION("Verify inference label correctness")
  {
    El::Fill(data, zero);
    El::FillDiagonal(data, one);

    auto labels = inf_alg.infer(model.get(), data, mbs_class_n);

    for (int i = 0; i < labels.Height(); i++) {
      REQUIRE(labels(i) == i);
    }
  }
}
