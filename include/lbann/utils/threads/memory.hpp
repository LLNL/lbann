#ifndef LBANN_MEMORY_HPP_
#define LBANN_MEMORY_HPP_

#include <memory>

namespace lbann {

#if __cplusplus < 201402L

/** \brief Local definition of make_unique for non-C++14 compilers */
template <typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts&&... params)
{
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}

#else

using std::make_unique;

#endif

}// namespace lbann

#endif /* LBANN_MEMORY_HPP_ */
