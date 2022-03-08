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
#include <catch2/catch.hpp>

#include "lbann/proto/helpers.hpp"

#include <callbacks.pb.h>
#include <model.pb.h>
#include <training_algorithm.pb.h>

using namespace lbann::proto::helpers;

TEST_CASE("Oneof utilities", "[utils][proto]")
{
  lbann_data::Callback callback;

  SECTION("Valid oneof")
  {
    REQUIRE_FALSE(has_oneof(callback, "callback_type"));

    callback.mutable_hang()->set_rank(123);
    REQUIRE(has_oneof(callback, "callback_type"));

    auto const& hang_msg = get_oneof_message(callback, "callback_type");
    REQUIRE_NOTHROW(
      dynamic_cast<lbann_data::Callback::CallbackHang const&>(hang_msg));
  }
  SECTION("Invalid oneof")
  {
    REQUIRE_FALSE(has_oneof(callback, "potato"));
    REQUIRE_THROWS(get_oneof_message(callback, "broccoli"));
  }
}

TEST_CASE("Message type names", "[utils][proto]")
{
  lbann_data::TrainingAlgorithm algo;
  lbann_data::Callback callback;
  lbann_data::Model model;

  SECTION("Plain ol' messages")
  {
    REQUIRE(message_type(algo) == "TrainingAlgorithm");
    REQUIRE(message_type(callback) == "Callback");
    REQUIRE(message_type(model) == "Model");
  }

  SECTION("\'Any\' messages")
  {
    google::protobuf::Any algo_any, callback_any, model_any;
    algo_any.PackFrom(algo);
    callback_any.PackFrom(callback);
    model_any.PackFrom(model);

    REQUIRE(message_type(algo_any) == "TrainingAlgorithm");
    REQUIRE(message_type(callback_any) == "Callback");
    REQUIRE(message_type(model_any) == "Model");
  }
}
