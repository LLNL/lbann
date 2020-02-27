#pragma once
#ifndef LBANN_UTILS_FACTORY_ERROR_POLICIES_HPP_
#define LBANN_UTILS_FACTORY_ERROR_POLICIES_HPP_

#include <memory>

#include <lbann/utils/exception.hpp>

namespace lbann
{

/** @class default_id_error_policy
 *  @brief Default policy describing how to handle unknown ids.
 *
 *  The policy must define "handle_unknown_id(IdT const&)".
 *
 *  The default behavior is to throw an exception.
 *
 *  @tparam IdT The type of id.
 *  @tparam ObjectT The type of the object being constructed by the factory.
 */
template <typename IdT, class ObjectT>
struct default_key_error_policy
{
  std::unique_ptr<ObjectT> handle_unknown_id(IdT const& id) const
  {
    // This could be expanded to print the id, but that would
    // assume that either the id can be inserted into a stream or
    // that the id can be converted to a string, which isn't
    // necessarily the case.
    LBANN_ERROR("Unknown id \"", id, "\" detected.");
  }
};// class default_key_error_policy

/** @class nullptr_key_error_policy
 *  @brief Policy returning a nullptr if the id is unknown
 *
 *  This class just returns "nullptr". Use of this class is not
 *  recommended as it probably indicates bad design that would better
 *  utilize exception handling. But it felt awkward to not at least
 *  provide it.
 *
 *  @tparam IdT The type of id.
 *  @tparam ObjectT The type of the object being constructed by the factory.
 */
template <typename IdT, class ObjectT>
struct nullptr_key_error_policy
{
  std::unique_ptr<ObjectT> handle_unknown_id(IdT const&) const noexcept
  {
    return nullptr;
  }
};// class nullptr_key_error_policy

}// namespace lbann
#endif /* LBANN_UTILS_FACTORY_ERROR_POLICIES_HPP_ */
