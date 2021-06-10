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

#include "lbann/proto/factories.hpp"
#include "lbann/proto/helpers.hpp"
#include "lbann/utils/factory.hpp"
#include "lbann/utils/typename.hpp"

#include "lbann/operators/operator.hpp"
#include "lbann/operators/math/math_builders.hpp"

#include "lbann/utils/peek_map.hpp"

#include <operators.pb.h>

#ifdef LBANN_HAS_DNN_LIB
#include "lbann/utils/dnn_lib/helpers.hpp"
#endif // LBANN_HAS_DNN_LIB

namespace lbann {
namespace proto {

namespace {

// Define the factory type.
using factory_type = lbann::generic_factory<
  lbann::Operator,
  std::string,
  generate_builder_type<lbann::Operator,
                        const lbann_data::Operator&>,
  nullptr_key_error_policy>;

/** @brief Singleton holder for a factory.
 *
 *  @note This design requires that the builder function be valid for
 *  every combination of T. That is, operator types for which a
 *  combination is invalid must handle that error inside their builder
 *  function.
 */
template <typename T>
class factory_manager
{
public:

  factory_manager() { register_default_builders(); }
  factory_type const& get() const noexcept { return factory_; }

private:

  // This macro simplifies the process of adding default builders
#define LBANN_REGISTER_BUILDER(KEY, OPERATOR_NAME)                         \
    factory_.register_builder(                                          \
      #KEY, build_##OPERATOR_NAME##_operator_from_pbuf<T>)
#define LBANN_REGISTER_DEFAULT_BUILDER(KEY, OPERATOR_NAME)                 \
    factory_.register_builder(                                          \
      #KEY,                                                             \
      [](lbann_data::Operator const&){                                     \
        return lbann::make_unique<OPERATOR_NAME##_operator<T>>();         \
      })

  // Builder registration happens here
  void register_default_builders() {

    // Math operators
    LBANN_REGISTER_BUILDER(Clamp, clamp);

  }

  // Just to be clear/safe.
#undef LBANN_REGISTER_DEFAULT_BUILDER

private:
  factory_type factory_;
}; // class factory_manager

template <typename T>
factory_type const& get_operator_factory() noexcept
{
  static factory_manager<T> factory_mgr_;
  return factory_mgr_.get();
}

} // namespace

template <typename TensorDataType>
std::unique_ptr<Operator> construct_operator(
  const lbann_data::Operator& proto_operator) {

  auto const& factory = get_operator_factory<TensorDataType>();
  auto const& msg =
    helpers::get_oneof_message(proto_operator, "operator_type");

  std::unique_ptr<Operator> o = factory.create_object(
    msg.GetDescriptor()->name(), proto_operator);
  return o;
}

// Template instantiation
#define PROTO(T) \
  template std::unique_ptr<Operator> construct_operator<T>( \
    const lbann_data::Operator& proto_operator              \
  )

#include "lbann/macros/instantiate.hpp"
#undef PROTO

} // namespace proto
} // namespace lbann
