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
#pragma once
#ifndef LBANN_UTILS_FACTORY_HPP_
#define LBANN_UTILS_FACTORY_HPP_

#include <lbann_config.hpp>
#include <lbann/utils/factory_error_policies.hpp>

#ifdef LBANN_HAS_DIHYDROGEN

#include <h2/patterns/factory/ObjectFactory.hpp>

namespace lbann
{

template <class BaseT, typename KeyT,
          typename BuilderT = std::function<std::unique_ptr<BaseT>()>,
          template <typename, class> class KeyErrorPolicy
          = default_key_error_policy>
using generic_factory =
  h2::factory::ObjectFactory<BaseT, KeyT, BuilderT, KeyErrorPolicy>;

} // namespace lbann

#else // !LBANN_HAS_DIHYDROGEN

// WARNING: This code is deprecated and will be removed when
// DiHydrogen becomes a required dependency of LBANN.

#include <algorithm>
#include <forward_list>
#include <functional>
#include <memory>
#include <unordered_map>

namespace lbann
{

/** @class generic_factory
 *  @brief Generic factory template.
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
template <class BaseT, typename IdT,
          typename BuilderT = std::function<std::unique_ptr<BaseT>()>,
          template <typename, class> class IdErrorPolicy
          = default_key_error_policy>
class generic_factory : private IdErrorPolicy<IdT,BaseT>
{
public:
  using base_type = BaseT;
  using id_type = IdT;
  using builder_type = BuilderT;

private:
  // This could be any of std::unordered_map, std::map, and something
  // even more bland like std::list<std::pair<id_type, builder_type>>
  // depending on the properties of "id_type". My initial assumption
  // is that ids will be hashable types...
  using map_type = std::unordered_map<id_type,builder_type>;

public:
  using size_type = typename map_type::size_type;

public:
  /** @name Builder registration */
  ///@{

  /** @brief Register a new builder for id @c id.
   *
   *  @param id     An identifier for a concrete type to be constructed.
   *  @param builder An @c Invokable object that builds concrete objects.
   *
   *  @return @c true if the builder was registered successfully; @c
   *      false otherise.
   */
  bool register_builder(id_type id, builder_type builder)
  {
    return m_registered_builders.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(std::move(id)),
      std::forward_as_tuple(std::move(builder))).second;
  }

  /** @brief Unregister the current builder for id @c id.
   *
   *  @param id The id for the builder to be removed from the factory.
   *
   *  @return @c true if a builder was unregistered; @c false
   *      otherwise.
   */
  bool unregister(id_type const& id)
  {
    return m_registered_builders.erase(id);
  }

  ///@}
  /** @brief Object construction */
  ///@{

  /** @brief Construct a new object.
   *
   *  @param id  The id for the object to be created.
   *  @param args Extra arguments for the builder.
   *
   *  @return A newly-built object managed by an @c std::unique_ptr.
   */
  template <typename... Ts>
  std::unique_ptr<base_type> create_object(
    id_type const& id, Ts&&... args) const
  {
    auto it = m_registered_builders.find(id);
    if (it != m_registered_builders.end())
      return (it->second)(std::forward<Ts>(args)...);

    return this->handle_unknown_id(id);
  }

  ///@}
  /** @name Queries */
  ///@{

  /** @brief Get the number of registered builders. */
  size_type size() const noexcept
  {
    return m_registered_builders.size();
  }

  /** @brief Get the names of all builders known to the factory.
   *
   *  @return A list of the known ids.
   */
  std::forward_list<id_type> registered_ids() const
  {
    std::forward_list<id_type> names;
    std::transform(
      m_registered_builders.cbegin(), m_registered_builders.cend(),
      std::front_inserter(names),
#if __cplusplus < 201402L
      [](typename map_type::value_type const& x)
#else
      [](auto const& x)
#endif
      { return x.first; });

    return names;
  }

private:
  /** @brief An associative list of ids and builders. */
  map_type m_registered_builders;
};// class generic_factory

}// namespace lbann
#endif // LBANN_HAS_DIHYDROGEN
#endif // LBANN_UTILS_FACTORY_HPP_
