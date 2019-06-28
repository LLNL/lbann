#ifndef LBANN_UTILS_ANY_HPP_INCLUDED
#define LBANN_UTILS_ANY_HPP_INCLUDED

#include <lbann_config.hpp>

#ifdef LBANN_HAS_STD_ANY

#include <any>

#else
#include <lbann/utils/memory.hpp>// non-C++14 make_unique

#include <iostream>
#include <memory>
#include <stdexcept>
#include <typeinfo>
#endif // LBANN_HAS_STD_ANY

namespace lbann
{
namespace utils
{

#ifdef LBANN_HAS_STD_ANY
// This case is simple symbol injection; don't feel great about this,
// but it's not my fault they couldn't get this into C++11...

using any = std::any;
using bad_any_cast = std::bad_any_cast;
using std::any_cast;
using std::make_any;

#else

/** @defgroup stl_wrappers C++ STL Wrappers
 *
 *  The `std::any` interface was not added to ISO C++ until the 2017
 *  standard. However, it provides useful features and is
 *  implementable in C++11. This fallback implementation should only
 *  be used if C++17 is not available.
 */

/** @class any
 *  @brief Type-erasure class to store any object of copy-constructible type.
 *
 *  This class is (mostly) API-compatible with std::any. The
 *  most notable omission is the std::in_place_type_t overloads of the
 *  constructor (std::in_place_type_t is also C++17, and I don't want
 *  to implement the whole standard). For best results, do not attempt
 *  to use those in this code. For even better results (yes, better
 *  than best. English is overrated), incessently remind your friends,
 *  colleagues, and, most importantly, vendors that it's 2019, that
 *  2019 > 2017, and that there are excellent free compilers in the
 *  world until they concede to updating to a modern compiler and this
 *  implementation can be banished to the depths.
 *
 *  @ingroup stl_wrappers
 */
class any
{
public:

  /** @name Constructors and destructor */
  ///@{

  /** @brief Default construct an empty "any" */
  any() noexcept {}

  /** @brief Construct an object holding a T */
  template <typename T>
  any(T&& obj);

  /** @brief Copy construct from another container.
   *
   *  Makes a copy of the held object.
   */
  any(any const& other);

  /** @brief Move construct from another container */
  any(any&& other) noexcept = default;

  /** @brief Default destructor */
  ~any() = default;

  ///@}
  /** @name Assignment operator */
  ///@{

  /** @brief Copy assign from another container
   *
   *  Makes a deep copy of the held object.
   */
  any& operator=(any const& other);

  /** @brief Move assign from another container */
  any& operator=(any&& other) noexcept = default;

  ///@}
  /** @name Modifiers */
  ///@{

  /** @brief Change the contained object to one of type T
   *
   *  Any held object is destroyed and the new object is
   *  emplace-constructed from the arguments given.
   *
   *  @tparam T The type of the new held object
   *  @tparam Args (Deduced) types of arguments to the T constructor
   *
   *  @param args The arguments to the T constructor
   *
   *  @return A reference to the newly constructed object
   */
  template <typename T, typename... Args>
  auto emplace(Args&&... args) -> typename std::decay<T>::type&;

  /** @brief Reset the container to an empty state, destroying the
   *         held object.
   */
  void reset() noexcept;

  /** @brief Swap the contents of this container with another */
  void swap(any& other) noexcept;

  ///@}
  /** @name Observers */
  ///@{

  /** @brief Test whether the container holds a value */
  bool has_value() const noexcept;

  /** @brief Get the type_info object for the held type */
  std::type_info const& type() const noexcept;

  ///@}

private:

  /** @class holder_base
   *  @brief Abstract base class for storing the object
   */
  struct holder_base
  {
    /** @brief Destructor */
    virtual ~holder_base() = default;

    /** @brief Clone function */
    virtual std::unique_ptr<holder_base> clone() const = 0;

    /** @brief Get the type_info for the underlying object */
    virtual std::type_info const& type() const = 0;
  }; // class holder_base

  /** @class holder<T>
   *  @brief Class to hold a copy-constructible object of type T
   */
  template <typename T>
  struct holder : holder_base
  {
    /** @brief Construct by copying data */
    holder(T const& data) : m_data{data} {}

    /** @brief Construct by moving data */
    holder(T&& data) : m_data{std::move(data)} {}

    /** @brief Construct by emplace-constructing the T with the given
     *         arguments.
     */
    template <typename... Args>
    holder(Args&&... args) : m_data{std::forward<Args>(args)...}
    {}

    /** @brief Destructor */
    ~holder() = default;

    /** @brief Clone the data holder */
    std::unique_ptr<holder_base> clone() const final
    {
      return make_unique<holder>(m_data);
    }

    /** @brief Get the type_info for this object */
    std::type_info const& type() const { return typeid(T); }

    /** @brief The data object */
    T m_data;
  };// class holder

private:

  template <typename T>
  friend T const* any_cast(any const*) noexcept;

  template <typename T>
  friend T* any_cast(any*) noexcept;

  std::unique_ptr<holder_base> m_holder = nullptr;

};// class any

/** @class bad_any_cast
 *  @brief Exception class indicating an any_cast has failed.
 */
struct bad_any_cast : std::runtime_error
{
  template <typename T>
  bad_any_cast(T&& what_arg)
    : std::runtime_error{std::forward<T>(what_arg)} {}
};// struct bad_any_cast

/** @brief Swap two any objects */
inline void swap(any& lhs, any& rhs)
{
  lhs.swap(rhs);
}

/** @brief Create an any object of type T constructed with args.
 *  @ingroup stl_wrappers
 */
template <typename T, typename... Ts>
any make_any(Ts&&... args)
{
  return any{T(std::forward<Ts>(args)...)};
}

/** @brief Typesafe access to the held object.
 *
 *  @tparam T The type of the held object.
 *
 *  @param obj The any object.
 *
 *  @return If obj is not null and holds a T, a pointer to
 *          the held object. Otherwise, nullptr.
 *
 *  @ingroup stl_wrappers
 */
template <typename T>
T* any_cast(any* obj) noexcept
{
  return const_cast<T*>(
    any_cast<T>(
      static_cast<any const*>(obj)));
}

/** @brief Typesafe access to the held object, const version.
 *
 *  @tparam T The type of the held object.
 *
 *  @param obj The any object.
 *
 *  @return If obj is not null and holds a T, a pointer to
 *          the held object. Otherwise, nullptr.
 *
 *  @ingroup stl_wrappers
 */
template <typename T>
T const* any_cast(any const* obj) noexcept
{
  static_assert(!std::is_reference<T>::value,
                "T must nust be a reference type.");

  if (!obj || !obj->has_value())
    return nullptr;

  if (obj->type() != typeid(T))
  {
    return nullptr;
  }

  auto T_holder = dynamic_cast<any::holder<T> const*>(obj->m_holder.get());
  return (T_holder ? &(T_holder->m_data) : nullptr);
}

/** @brief Typesafe access to the held object.
 *
 *  @tparam T The type of the held object.
 *
 *  @param obj The any object.
 *
 *  @return The held object.
 *
 *  @throws bad_any_cast If obj does not hold a T.
 *
 *  @ingroup stl_wrappers
 */
template <typename T>
T any_cast(any& obj)
{
  using type =
    typename std::remove_cv<
      typename std::remove_reference<T>::type>::type;
  auto* ret = any_cast<type>(&obj);
  if (not ret)
    throw bad_any_cast("bad any_cast");
  return *ret;
}

/** @brief Typesafe access to the held object.
 *
 *  This will move the held object into the returned object, if
 *  appropriate.
 *
 *  @tparam T The type of the held object.
 *
 *  @param obj The any object.
 *
 *  @return The held object.
 *
 *  @throws bad_any_cast If obj does not hold a T.
 *
 *  @post The any obj holds a moved-from T
 *
 *  @ingroup stl_wrappers
 */
template <typename T>
T any_cast(any&& obj)
{
  using type =
    typename std::remove_cv<
      typename std::remove_reference<T>::type>::type;
  auto ret = any_cast<type>(&obj);
  if (not ret)
    throw bad_any_cast("bad any_cast");
  return std::move(*ret);
}

/** @brief Typesafe access to the held object.
 *
 *  @tparam T The type of the held object.
 *
 *  @param obj The any object.
 *
 *  @return The held object.
 *
 *  @throws bad_any_cast If obj does not hold a T.
 *
 *  @ingroup stl_wrappers
 */
template <typename T>
T any_cast(any const& obj)
{
  using type =
    typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  auto ret = any_cast<type>(&obj);
  if (not ret)
    throw bad_any_cast("bad any_cast");
  return *ret;
}

// "any" member function implementation

template <typename T>
any::any(T&& obj)
  : m_holder{make_unique<holder<typename std::decay<T>::type>>(
    std::forward<T>(obj))}
{}

inline any::any(any const& other)
  : m_holder{other.has_value() ? other.m_holder->clone() : nullptr} {}

inline any& any::operator=(any const& other)
{
  m_holder = (other.has_value() ? other.m_holder->clone() : nullptr);
  return *this;
}

template <typename T, typename... Args>
auto any::emplace(Args&&... args)
  -> typename std::decay<T>::type&
{
  using held_type = typename std::decay<T>::type;

  reset();
  auto tmp_holder = make_unique<holder<held_type>>(
    std::forward<Args>(args)...);
  auto& ret = tmp_holder->m_data;
  m_holder = std::move(tmp_holder);
  return ret;
}

inline void any::reset() noexcept
{
  m_holder.reset();
}

inline void any::swap(any& other) noexcept
{
  std::swap(m_holder,other.m_holder);
}

inline bool any::has_value() const noexcept
{
  return (bool) m_holder;
}

inline std::type_info const& any::type() const noexcept
{
  return m_holder ? m_holder->type() : typeid(void);
}

#endif /* End fallback implementation */

}// namespace utils
}// namespace lbann
#endif // LBANN_UTILS_ANY_HPP_INCLUDED
