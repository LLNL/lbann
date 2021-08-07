////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/operators/math/clamp.hpp"
#include "lbann/operators/math/math_builders.hpp"
#include "lbann/operators/operator.hpp"
#include "lbann/proto/datatype_helpers.hpp"

#include <memory>

#include <operators.pb.h>

template <typename DataT, El::Device D>
std::unique_ptr<lbann::Operator<DataT, DataT, D>>
lbann::build_clamp_operator(lbann_data::Operator const& op)
{
  LBANN_ASSERT(proto::ProtoDataType<DataT> == op.input_datatype());
  LBANN_ASSERT(proto::ProtoDataType<DataT> == op.output_datatype());
  LBANN_ASSERT(proto::ProtoDevice<D> ==
               proto::resolve_default_device(op.device_allocation()));

  lbann_data::ClampOperator params;
  LBANN_ASSERT(op.parameters().UnpackTo(&params));
  return std::make_unique<ClampOperator<DataT, D>>(params.min(), params.max());
}
