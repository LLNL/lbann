#pragma once
#ifndef LBANN_UTILS_FACTORY_ERROR_POLICIES_HPP_
#define LBANN_UTILS_FACTORY_ERROR_POLICIES_HPP_

#include <memory>

#include <lbann/utils/exception.hpp>

namespace lbann
{

/** @class default_key_error_policy
 *  @brief Default policy describing how to handle unknown keys.
 *
 *  The policy must define "handle_unknown_key(KeyT const&)".
 *
 *  The default behavior is to throw an exception.
 *
 *  @tparam KeyT The type of key.
 *  @tparam ObjectT The type of the object being constructed by the factory.
 */
template <typename KeyT, class ObjectT>
struct default_key_error_policy
{
  std::unique_ptr<ObjectT> handle_unknown_key(KeyT const& key) const
  {
    // This could be expanded to print the key, but that would
    // assume that either the key can be inserted into a stream or
    // that the key can be converted to a string, which isn't
    // necessarily the case.
    LBANN_ERROR("Unknown key \"", key, "\" detected.");
  }
};// class default_key_error_policy

/** @class nullptr_key_error_policy
 *  @brief Policy returning a nullptr if the key is unknown
 *
 *  This class just returns "nullptr". Use of this class is not
 *  recommended as it probably indicates bad design that would better
 *  utilize exception handling. But it felt awkward to not at least
 *  provide it.
 *
 *  @tparam KeyT The type of key.
 *  @tparam ObjectT The type of the object being constructed by the factory.
 */
template <typename KeyT, class ObjectT>
struct nullptr_key_error_policy
{
  std::unique_ptr<ObjectT> handle_unknown_key(KeyT const&) const noexcept
  {
    return nullptr;
  }
};// class nullptr_key_error_policy

}// namespace lbann
#endif /* LBANN_UTILS_FACTORY_ERROR_POLICIES_HPP_ */
