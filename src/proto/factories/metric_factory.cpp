////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

namespace lbann {
namespace proto {

metric* construct_metric(lbann_comm* comm,
                         const lbann_data::Metric& proto_metric) {

  // Construct metric
  if (proto_metric.has_categorical_accuracy()) {
    return new categorical_accuracy_metric(comm);
  }
  if (proto_metric.has_top_k_categorical_accuracy()) {
    const auto& params = proto_metric.top_k_categorical_accuracy();
    return new top_k_categorical_accuracy_metric(params.top_k(), comm);
  }
  if (proto_metric.has_mean_squared_error()) {
    return new mean_squared_error_metric(comm);
  }
  if (proto_metric.has_mean_absolute_deviation()) {
    return new mean_absolute_deviation_metric(comm);
  }
  if (proto_metric.has_pearson_correlation()) {
    return new pearson_correlation_metric(comm);
  }
  if (proto_metric.has_r2()) {
    return new r2_metric(comm);
  }

  // Return null pointer if no optimizer is specified
  return nullptr;

}

} // namespace proto
} // namespace lbann
