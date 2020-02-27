#ifndef LBANN_MEMORY_HPP_
#define LBANN_MEMORY_HPP_

#include <lbann_config.hpp>
#include <memory>

namespace lbann {

#ifdef LBANN_HAS_STD_MAKE_UNIQUE

using std::make_unique;

#else

/** @brief Local definition of make_unique for non-C++14 compilers.
 *  @ingroup stl_wrappers
 */
template <typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts&&... params)
{
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}

#endif

/** @brief Convert the raw pointer to a unique_ptr. */
template <typename T>
std::unique_ptr<T> to_unique_ptr(T* ptr)
{
  return std::unique_ptr<T>(ptr);
}

}// namespace lbann

#endif /* LBANN_MEMORY_HPP_ */
