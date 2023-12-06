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

#ifndef LBANN_PROTO_FACTORIES_HPP_INCLUDED
#define LBANN_PROTO_FACTORIES_HPP_INCLUDED

#include "lbann/data_ingestion/data_reader.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/operators/operator.hpp"
#include "lbann/proto/proto_common.hpp"
#include "lbann/transforms/transform.hpp"
#include "lbann/transforms/transform_pipeline.hpp"

#include <google/protobuf/message.h>

#include <map>
#include <memory>

namespace lbann_data {
class Layer;
class Model;
class ObjectiveFunction;
class Optimizer;
class Operator;
class Reader;
class Transform;
class Weights;
} // namespace lbann_data

namespace lbann {

// Forward declarations
class callback_base;
class Layer;
class lbann_summary;
class model;
class objective_function;
class optimizer;
class trainer;
class weights;

namespace proto {

/** Construct a trainer specified with a prototext. */
std::unique_ptr<trainer>
construct_trainer(lbann_comm* comm, const lbann_data::Trainer& proto_trainer);

/** Construct a model specified with a prototext. */
std::unique_ptr<model> construct_model(lbann_comm* comm,
                                       const lbann_data::Optimizer& proto_opt,
                                       const lbann_data::Trainer& proto_trainer,
                                       const lbann_data::Model& proto_model);

/** Construct a layer graph specified with a prototext. */
std::vector<OwningLayerPtr>
construct_layer_graph(lbann_comm* comm,
                      const lbann_data::Trainer& proto_trainer,
                      const lbann_data::Model& proto_model);

/** Construct a layer specified with prototext. */
template <typename TensorDataType, data_layout layout, El::Device Dev>
std::unique_ptr<Layer> construct_layer(lbann_comm* comm,
                                       const lbann_data::Layer& proto_layer);

/** Construct an operator specified with prototext. */
template <typename InputT, typename OutputT, El::Device D>
auto construct_operator(const lbann_data::Operator& proto_operator)
  -> std::unique_ptr<Operator<InputT, OutputT, D>>;

/** Construct weights specified with prototext. */
std::unique_ptr<weights>
construct_weights(lbann_comm* comm,
                  const lbann_data::Optimizer& proto_opt,
                  const lbann_data::Weights& proto_weights);

/** Construct a callback specified with prototext. */
std::unique_ptr<callback_base>
construct_callback(const google::protobuf::Message& proto_cb);

/** Construct a callback specified with prototext. */
std::unique_ptr<callback_base>
construct_callback(const google::protobuf::Message& proto_cb,
                   std::shared_ptr<lbann_summary> const& summarizer);

/** Construct a summarizer specified with prototext.
 *  The summarizer is only constructed if the summarizer callback is
 *  enabled.
 */
std::unique_ptr<lbann_summary> construct_summarizer(lbann_comm* comm,
                                                    const lbann_data::Model& m);

/** Construct an optimizer specified with prototext. */
template <typename T>
std::unique_ptr<optimizer>
construct_optimizer(const lbann_data::Optimizer& proto_opt);

/** Construct an objective function specified with prototext. */
std::unique_ptr<objective_function>
construct_objective_function(const lbann_data::ObjectiveFunction& proto_obj);

/** Construct a transform given a prototext. */
std::unique_ptr<transform::transform>
construct_transform(const lbann_data::Transform& trans);
/** Construct a transform pipeline given a data reader prototext. */
transform::transform_pipeline
construct_transform_pipeline(const lbann_data::Reader& data_reader);

} // namespace proto
} // namespace lbann

#endif // LBANN_PROTO_FACTORIES_HPP_INCLUDED
