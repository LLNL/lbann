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
#pragma once
#ifndef LBANN_UTILS_FACTORY_HPP_INCLUDED
#define LBANN_UTILS_FACTORY_HPP_INCLUDED

#include <lbann/utils/factory_error_policies.hpp>
#include <lbann_config.hpp>

#include <h2/patterns/factory/ObjectFactory.hpp>

namespace lbann {

/** @brief Generic factory template.
 *
 *  This is a generic factory that should be suitable for constructing
 *  objects of a particular base type. The goal is maximum reuse:
 *
 *      using layer_factory
 *        = generic_factory<layer, string, layer_builder_type>;
 *      using callback_factory
 *        = generic_factory<lbann_callback, string, callback_builder_type>;
 *
 *  The default behavior for id errors is to throw an exception.
 *
 *  @tparam BaseT        The base class of the types being constructed.
 *  @tparam IdT         The index type used to differentiate concrete types.
 *  @tparam BuilderT     The functor type that builds concrete types.
 *  @tparam ErrorPolicy  The policy for handling id errors.
 */
template <class BaseT,
          typename KeyT,
          typename BuilderT = std::function<std::unique_ptr<BaseT>()>,
          template <typename, class> class KeyErrorPolicy =
            default_key_error_policy>
using generic_factory =
  h2::factory::ObjectFactory<BaseT, KeyT, BuilderT, KeyErrorPolicy>;

/** @brief A helper struct for creating builder signatures.
 *  @details This struct, when used with the helper typedef, is used
 *           to hide some of the "std::function" stuff that is needed
 *           to make the factory work. It does nothing except
 *           short-hand some type information. As the factory pattern
 *           is primarily useful when dealing with class hierarchies,
 *           this struct wraps OutT in std::unique_ptr.
 *  @tparam OutT The type of unique_ptr that is returned by the builder.
 *  @tparam Args The types of the arguments passed to the builder.
 */
template <typename OutT, typename... Args>
struct GenerateBuilderType_struct
{
  using type = std::function<std::unique_ptr<OutT>(Args...)>;
};

/** @brief A helper typedef for wrapping builder signatures. */
template <typename OutT, typename... Args>
using generate_builder_type =
  typename GenerateBuilderType_struct<OutT, Args...>::type;

} // namespace lbann
#endif // LBANN_UTILS_FACTORY_HPP_INCLUDED
