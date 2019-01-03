#pragma once
#ifndef LBANN_UTILS_FACTORY_HPP_
#define LBANN_UTILS_FACTORY_HPP_

#include <algorithm>
#include <forward_list>
#include <functional>
#include <memory>
#include <unordered_map>

#include <lbann/utils/factory_error_policies.hpp>

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
 *  The default behavior for key errors is to throw an exception.
 *
 *  @tparam BaseT        The base class of the types being constructed.
 *  @tparam KeyT         The index type used to differentiate concrete types.
 *  @tparam BuilderT     The functor type that builds concrete types.
 *  @tparam ErrorPolicy  The policy for handling key errors.
 */
template <class BaseT, typename KeyT,
          typename BuilderType = std::function<std::unique_ptr<BaseT>()>,
          template <typename, class> class KeyErrorPolicy
          = default_key_error_policy>
class generic_factory : private KeyErrorPolicy<KeyT,BaseT>
{
public:
  using base_type = BaseT;
  using key_type = KeyT;
  using builder_type = BuilderT;

private:
  // This could be any of std::unordered_map, std::map, and something
  // even more bland like std::list<std::pair<key_type, builder_type>>
  // depending on the properties of "key_type". My initial assumption
  // is that keys will be hashable types...
  using map_type = std::unordered_map<key_type,builder_type>;

public:
  /** @brief Register a new builder for key @c key.
   *
   *  @param key     An identifier for a concrete type to be constructed.
   *  @param builder An @c Invokable object that builds concrete objects.
   *
   *  @return @c true if the builder was registered successfully; @c
   *      false otherise.
   */
  bool register_builder(key_type key, builder_type builder)
  {
    return m_registered_builders.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(std::move(key)),
      std::forward_as_tuple(std::move(builder))).second;
  }

  /** @brief Unregister the current builder for key @c key.
   *
   *  @param key The key for the builder to be removed from the factory.
   *
   *  @return @c true if a builder was unregistered; @c false
   *      otherwise.
   */
  bool unregister(key_type const& key)
  {
    return m_registered_builders.erase(key);
  }

  /** @brief Construct a new object.
   *
   *  @param key  The key for the object to be created.
   *  @param args Extra arguments for the builder.
   *
   *  @return A newly-built object managed by an @c std::unique_ptr.
   */
  template <typename... Ts>
  std::unique_ptr<base_type> create_object(
    key_type const& key, Ts&&... Args) const
  {
    auto it = m_registered_builders.find(key);
    if (it != m_registered_builders.end())
      return (it->second)(std::forward<Ts>(Args)...);

    return this->handle_unknown_key(key);
  }

  /** @brief Get the names of all builders known to the factory.
   *
   *  @return A list of the known keys.
   */
  std::forward_list<key_type> get_registered_keys() const
  {
    std::forward_list<key_type> names;
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
  /** @brief An associative list of keys and builders. */
  map_type m_registered_builders;
};// class generic_factory

}// namespace lbann
#endif /* LBANN_UTILS_FACTORY_HPP_ */
