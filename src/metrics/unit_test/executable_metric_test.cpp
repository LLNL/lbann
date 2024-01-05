////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
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

#include "lbann/metrics/executable_metric.hpp"
#include "lbann/models/model.hpp"
#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/proto/lbann.pb.h"
#include "lbann/proto/proto_common.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/lbann_library.hpp"
#include "lbann/utils/serialize.hpp"

const std::string model_header = R"""(
model {
  layer {
    name: "layer1"
    constant {
      value: 0.0
      num_neurons: 1
    }
  }
)""";

const std::string model_footer = R"""(
}
)""";

std::unique_ptr<lbann::model>
setup_metric_model(const std::string& metric_contents)
{
  std::string model_contents = model_header + metric_contents + model_footer;
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

TEST_CASE("Executable metric", "[metrics][executable]")
{
  // In the following tests, we must set up a model rather than instantiating
  // the lbann::executable_metric object directly because the metric forwards
  // the current trainer and model name. Since ``metric::setup`` is protected,
  // we also cannot instantiate one model and create the metrics separately.
  SECTION("normal")
  {
    // Set up an executable metric without any arguments
    auto m = setup_metric_model(R"""(
      metric {
        executable_metric {
          name: "metric"
          filename: "src/metrics/unit_test/metric-tester"
        }
      }
    )""");
    auto* metric = m->get_metrics()[0];

    // Check result
    CHECK(metric->evaluate(lbann::execution_mode::testing, 1) == 1.4);
  }
  SECTION("arguments")
  {
    auto m = setup_metric_model(R"""(
      metric {
        executable_metric {
          name: "metric"
          filename: "src/metrics/unit_test/metric-tester"
          other_args: "arg"
        }
      }
    )""");
    auto* metric = m->get_metrics()[0];

    // Adding the extra argument prints out a different result
    CHECK(metric->evaluate(lbann::execution_mode::testing, 1) == -2.8);
  }
  SECTION("nonnumeric")
  {
    auto m = setup_metric_model(R"""(
      metric {
        executable_metric {
          name: "metric"
          filename: "src/metrics/unit_test/metric-tester"
          other_args: "fail"
        }
      }
    )""");
    auto* metric = m->get_metrics()[0];

    // Application should print a non-numeric number
    REQUIRE_THROWS(metric->evaluate(lbann::execution_mode::testing, 1));
  }
  SECTION("returncode")
  {
    auto m = setup_metric_model(R"""(
      metric {
        executable_metric {
          name: "metric"
          filename: "src/metrics/unit_test/metric-tester"
          other_args: "retcode"
          }
      }
    )""");
    auto* metric = m->get_metrics()[0];

    // Application should return error code 2
    REQUIRE_THROWS(metric->evaluate(lbann::execution_mode::testing, 1));
  }
}
