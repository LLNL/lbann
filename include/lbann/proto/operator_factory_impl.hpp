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

#include "lbann/proto/operator_factory.hpp"

#include "lbann/operators/math/math_builders.hpp"
#include "lbann/operators/operator.hpp"

namespace lbann {
namespace proto {
namespace details {

template <typename InT, typename OutT, El::Device D>
OperatorFactory<InT, OutT, D> build_default_factory()
{
  OperatorFactory<InT, OutT, D> factory;

  if constexpr (std::is_same_v<InT, OutT>) {
    factory.register_builder("ClampOperator", build_clamp_operator<InT, D>);
  }
  return factory;
}

} // namespace details
} // namespace proto
} // namespace lbann

template <typename InT, typename OutT, El::Device D>
lbann::proto::OperatorFactory<InT, OutT, D>&
lbann::proto::get_operator_factory()
{
  static auto factory = details::build_default_factory<InT, OutT, D>();
  return factory;
}
