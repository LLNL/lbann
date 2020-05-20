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
#ifndef LBANN_UTILS_CLONEABLE_HPP_INCLUDED
#define LBANN_UTILS_CLONEABLE_HPP_INCLUDED

#include <memory>
#include <type_traits>

/** @file
 *
 *  This file implements covariant returns via smart pointers for a
 *  polymorphic @c clone functoin. The implementation largely follows
 *  the solution presented <a
 *  href="https://www.fluentcpp.com/2017/09/12/how-to-return-a-smart-pointer-and-use-covariance/">by
 *  the FluentC++ blog</a>. Some class/tag names have been updated to
 *  be clearer, in my opinion. Additionally, a semi-useful predicate
 *  has been added to aid metaprogramming down the line.
 */

namespace lbann {

/** @brief Declare @c Base to be a virtual base.
 *
 *  This metafunction adds @c Base as a virtual base
 *  class. Constructors of @c Base are added to this class.
 *
 *  @tparam Base The class to be declared as a virtual base.
 */
template <typename Base>
struct AsVirtualBase : virtual Base
{
  using Base::Base;
};

/** @brief Declare that @c T has unimplemented virtual functions.
 *
 *  Due to metaprogramming restrictions on CRTP interfaces, we rely on
 *  the user of these mechanisms to declare when a class has
 *  unimplemented virtual functions (or "is abstract").
 *
 *  @tparam T The type that has at least one unimplemented virtual
 *  function.
 */
template <typename T>
struct HasAbstractFunction {};

/** @brief Alias for HasAbstractFunction.
 *
 *  Good OO practice suggests that non-leaf classes should be abstract
 *  -- that is, have at least one unimplemented virtual
 *  function. LBANN fits this paradigm, so this alias is appropriate.
*/
template <typename T>
using NonLeafClass = HasAbstractFunction<T>;

/** @brief Inject polymorphic clone functions into hierarchies.
 *
 *  This class uses CRTP to inject the derived class's clone()
 *  function directly into the class and uses
 *  <a href="http://www.gotw.ca/publications/mill18.htm">the
 *  Template Method</a> to virtualize it.
 *
 *  @tparam T The concrete class to be cloned.
 *  @tparam Base The base class of T.
 */
template <typename T, typename... Base>
class Cloneable
  : public Base...
{
public:
  /** @brief Return an exception-safe, memory-safe copy of this object. */
  std::unique_ptr<T> clone() const {
    return std::unique_ptr<T>{static_cast<T*>(this->do_clone_())};
  }
private:
  /** @brief Implement the covariant raw-pointer-based clone operation. */
  virtual Cloneable* do_clone_() const override {
    return new T(static_cast<T const&>(*this));
  }
};// class Cloneable

template <typename T, typename Base>
class Cloneable<T, Base>
  : public Base
{
public:
  /** @brief Return an exception-safe, memory-safe copy of this object. */
  std::unique_ptr<T> clone() const {
    return std::unique_ptr<T>{static_cast<T*>(this->do_clone_())};
  }
protected:
  using Base::Base;
private:
  /** @brief Implement the covariant raw-pointer-based clone operation. */
  virtual Cloneable* do_clone_() const override {
    return new T(static_cast<T const&>(*this));
  }
};// class Cloneable

/** @brief Specialization of Cloneable to handle stand-alone classes. */
template <typename T>
class Cloneable<T>
{
public:
  virtual ~Cloneable() = default;

  std::unique_ptr<T> clone() const {
    return std::unique_ptr<T>{static_cast<T*>(this->do_clone_())};
  }
private:
  Cloneable* do_clone_() const {
    return new T(static_cast<T const&>(*this));
  }
};// class Cloneable<T>

/** @brief Specialization of Cloneable for intermediate classes.
 *
 *  Classes that are neither the top of the hierarchy nor a leaf of
 *  the class tree should be virtual. An unfortunate consequence of
 *  the CRTP method is that the target of the CRTP, @c T in this case,
 *  is not a complete class when this class is instantiated, so
 *  metaprogramming based on @c T is very restricted. Thus, users must
 *  tag the target class with HasAbstractFunction. Doing so will
 *  ensure that the @c do_clone_() function is declared pure virtual.
 */
template <typename T, typename... Base>
class Cloneable<HasAbstractFunction<T>, Base...>
  : public Base...
{
public:
  std::unique_ptr<T> clone() const {
    return std::unique_ptr<T>{static_cast<T*>(this->do_clone_())};
  }
private:
  virtual Cloneable* do_clone_() const = 0;
};

template <typename T, typename Base>
class Cloneable<HasAbstractFunction<T>, Base>
  : public Base
{
public:
  std::unique_ptr<T> clone() const {
    return std::unique_ptr<T>{static_cast<T*>(this->do_clone_())};
  }
protected:
  using Base::Base;
private:
  virtual Cloneable* do_clone_() const = 0;
};

/** @brief Specialization of Cloneable to handle the top of hierarchies. */
template <typename T>
class Cloneable<HasAbstractFunction<T>>
{
public:
  virtual ~Cloneable() = default;

  std::unique_ptr<T> clone() const {
    return std::unique_ptr<T>{static_cast<T*>(this->do_clone_())};
  }
private:
  virtual Cloneable* do_clone_() const = 0;
};// class Cloneable<T>

/** @brief Predicate testing for Cloneable interface.
 *
 *  This predicate determines whether a class supports the Cloneable
 *  interface. If true, this class will support a smart-pointer-to-T
 *  return from a @c clone() method.
 *
 *  This predicate type suffers a deficiency that it can be fooled
 *  rather easily. It is generally not possible to determine from the
 *  specific Cloneable instantiation used for a given type. Thus,
 *  alternative strategies must be used. As it stands, any class that
 *  provides a @c clone() method that returns a @c std::unique_ptr<T>
 *  will satisfy this predicate.
 *
 *  @tparam T The type being tested.
 */
template <typename T>
struct IsCloneableT;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// The obvious case; I'd be concerned if this were ever called.
template <typename... Ts>
struct IsCloneableT<Cloneable<Ts...>> : std::true_type {};

namespace details {

struct definitely_not_a_unique_ptr;

template <typename T>
auto has_right_clone(T const& x) -> decltype(x.clone());

definitely_not_a_unique_ptr has_right_clone(...);

}// namespace details

template <typename T>
struct IsCloneableT
  : std::is_same<decltype(details::has_right_clone(std::declval<T>())),
                 std::unique_ptr<T>>
{};
#endif // DOXYGEN_SHOULD_SKIP_THIS

template <typename T>
constexpr bool IsCloneable_v() { return IsCloneableT<T>::value; };

}// namespace lbann
#endif // LBANN_UTILS_CLONEABLE_HPP_INCLUDED
