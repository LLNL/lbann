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

#include "lbann/execution_algorithms/sgd_training_algorithm.hpp"
#include "lbann/execution_algorithms/training_algorithm.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/make_abstract.hpp"

#include <exception>
// #include <google/protobuf/stubs/logging.h>
#include <lbann/execution_algorithms/factory.hpp>

#include "lbann/proto/training_algorithm.pb.h"

#include <google/protobuf/text_format.h>

namespace pb = ::google::protobuf;

#ifdef LBANN_USE_CATCH2_V3
static Catch::Matchers::StringContainsMatcher Contains(std::string const& str)
{
  return Catch::Matchers::ContainsSubstring(str, Catch::CaseSensitive::Yes);
}
#endif // LBANN_USE_CATCH2_V3

TEST_CASE("Parsing training algorithm prototext", "[factory][algorithm][proto]")
{
  SECTION("Parsing any messages from prototext works")
  {
    std::string const valid_prototext = R"proto(
name: "local sgd"
parameters {
  [type.googleapis.com/lbann_data.SGD] {
    stopping_criteria {
      max_batches: 5
    }
  }
})proto";

    lbann_data::TrainingAlgorithm algo_msg;
    REQUIRE(pb::TextFormat::ParseFromString(valid_prototext, &algo_msg));
    REQUIRE(algo_msg.has_parameters());
    REQUIRE(algo_msg.parameters().Is<lbann_data::SGD>());
    REQUIRE(algo_msg.name() == "local sgd");

    lbann_data::SGD sgd_msg;
    REQUIRE(algo_msg.parameters().UnpackTo(&sgd_msg));
    REQUIRE(sgd_msg.stopping_criteria().max_batches() == 5);
  }

  SECTION("Type URL error -- bad domain")
  {
    std::string const bad_domain_prototext = R"proto(
name: "my domain name is gibberish"
parameters {
  [apples.oranges.com/lbann_data.SGD] {
    stopping_criteria {
      max_batches: 5
    }
  }
})proto";

    // Protobuf will log stuff to stderr when errors occur. We know
    // these errors will occur, but they'll add potentially confusing
    // output to the catch run, so we silence them in this section
    // google::protobuf::LogSilencer tmp_silence_pb_logs;

    lbann_data::TrainingAlgorithm algo_msg;
    REQUIRE_FALSE(
      pb::TextFormat::ParseFromString(bad_domain_prototext, &algo_msg));
  }
  SECTION("Type URL error -- unknown class")
  {
    std::string const unknown_class_prototext = R"proto(
name: "my class name does not exist"
parameters {
  [type.googleapis.com/lbann_data.UnknownSolverName] {
    stopping_criteria {
      max_batches: 5
    }
  }
})proto";

    // google::protobuf::LogSilencer tmp_silence_pb_logs;
    lbann_data::TrainingAlgorithm algo_msg;
    REQUIRE_FALSE(
      pb::TextFormat::ParseFromString(unknown_class_prototext, &algo_msg));
  }

  SECTION("Type URL error -- bad class parameters")
  {
    std::string const bad_class_prototext = R"proto(
name: "my class name does not match my parameters"
parameters {
  [type.googleapis.com/lbann_data.LTFB] {
    stopping_criteria {
      max_batches: 5
    }
  }
})proto";

    // google::protobuf::LogSilencer tmp_silence_pb_logs;
    lbann_data::TrainingAlgorithm algo_msg;
    REQUIRE_FALSE(
      pb::TextFormat::ParseFromString(bad_class_prototext, &algo_msg));
  }
}

TEST_CASE("Building training algorithm from the factory",
          "[factory][algorithm][proto]")
{
  SECTION("Building SGD works fine.")
  {
    lbann_data::SGD sgd_msg;
    sgd_msg.mutable_stopping_criteria()->set_max_batches(5);

    lbann_data::TrainingAlgorithm algo_msg;
    algo_msg.set_name("my sgd algo");
    algo_msg.mutable_parameters()->PackFrom(sgd_msg);

    auto sgd = lbann::make_abstract<lbann::TrainingAlgorithm>(algo_msg);

    REQUIRE_NOTHROW(dynamic_cast<lbann::SGDTrainingAlgorithm const&>(*sgd));

    REQUIRE(sgd->get_type() == "sgd");
    REQUIRE(sgd->get_name() == "my sgd algo");
  }

  SECTION("Building with an invalid message type fails")
  {
    lbann_data::SGD::TerminationCriteria wrong_msg_type;
    wrong_msg_type.set_max_batches(13);

    lbann_data::TrainingAlgorithm algo_msg;
    algo_msg.set_name("my bad sgd algo");
    algo_msg.mutable_parameters()->PackFrom(wrong_msg_type);

    REQUIRE_THROWS_WITH(
      lbann::make_abstract<lbann::TrainingAlgorithm>(algo_msg),
      Contains("Unknown id \"TerminationCriteria\" detected"));
  }
}
