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

#include "lbann/proto/factories.hpp"

#include "lbann/metrics/executable_metric.hpp"
#include "lbann/metrics/layer_metric.hpp"
#include "lbann/metrics/metric.hpp"
#include "lbann/metrics/python_metric.hpp"

#include "lbann/proto/metrics.pb.h"

#include "lbann/utils/protobuf/decl.hpp"

namespace lbann {
namespace proto {

std::unique_ptr<metric> construct_metric(lbann_comm* comm,
                                         const lbann_data::Metric& proto_metric)
{
  auto metric_type = lbann::protobuf::which_oneof(proto_metric, "metric_type");
  if (metric_type == "layer_metric") {
    auto const& lm = proto_metric.layer_metric();
    return std::make_unique<layer_metric>(comm, lm.name(), lm.unit());
  }
  else if (metric_type == "executable_metric") {
    auto const& xm = proto_metric.executable_metric();
    return std::make_unique<executable_metric>(comm,
                                               xm.name(),
                                               xm.filename(),
                                               xm.other_args());
  }
  else if (metric_type == "python_metric") {
    auto const& pym = proto_metric.python_metric();
    return std::make_unique<python_metric>(comm,
                                           pym.name(),
                                           pym.module(),
                                           pym.module_dir(),
                                           pym.function());
  }
  else {
    LBANN_ERROR("Unsupported metric type \"", metric_type, "\"");
  }
}

} // namespace proto
} // namespace lbann
