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

#ifndef LBANN_UTILS_H2_TMP_HPP_
#define LBANN_UTILS_H2_TMP_HPP_

#include <lbann_config.hpp>

#ifdef LBANN_HAS_DIHYDROGEN

#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>

#else // !LBANN_HAS_DIHYDROGEN

// WARNING: This code is deprecated and will be removed when
// DiHydrogen becomes a required dependency of LBANN.

/** @file
 *
 *  This file contains a small slice of the metaprogramming library
 *  available in DiHydrogen. This file will eventually be deleted as
 *  DiHydrogen is integrated into LBANN; however, this was seen as too
 *  large a task for the present needs.
 *
 *  @note Testing for this functionality has not been included in
 *        LBANN. It is available in the main H2 repository.
 *
 *  @warning This file is deprecated and will be removed when
 *           DiHydrogen becomes a required dependency of LBANN.
 */

#include <utility> // std::forward, std::declval

#ifndef H2_META_CORE_LAZY_HPP_
#define H2_META_CORE_LAZY_HPP_

namespace h2
{
namespace meta
{

/** @brief Suspend a given type. */
template <typename T>
struct Susp
{
    using type = T;
};

/** @brief Extract the internal type from a suspended type. */
template <typename SuspT>
using Force = typename SuspT::type;

}// namespace meta
}// namespace h2
#endif // H2_META_CORE_LAZY_HPP_

#ifndef H2_META_CORE_SFINAE_HPP_
#define H2_META_CORE_SFINAE_HPP_

namespace h2
{
namespace meta
{

/** @brief A SFINAE tool for excluding functions/overloads.
 *
 *  Contains a typedef `type` if the condition is `true`.
 */
template <bool B, typename ResultT = void>
struct EnableIfT;

/** @brief A SFINAE tool that contains a type when the condition is true. */
template <bool B, typename ResultT = void>
using EnableIf = meta::Force<EnableIfT<B, ResultT>>;

/** @brief An alias for EnableIf. */
template <bool B, typename ResultT = void>
using EnableWhen = EnableIf<B, ResultT>;

/** @brief A SFINAE tool that contains a type when the condition is false. */
template <bool B, typename ResultT = void>
using EnableUnless = EnableWhen<!B, ResultT>;

/** @brief A version of EnableIf that operates on valued types. */
template <typename B, typename ResultT = void>
using EnableIfV = EnableIf<B::value, ResultT>;

/** @brief An alias for EnableIfV. */
template <typename B, typename ResultT = void>
using EnableWhenV = EnableWhen<B::value, ResultT>;

/** @brief A version of EnableUnless that operates on valued types. */
template <typename B, typename ResultT = void>
using EnableUnlessV = EnableUnless<B::value, ResultT>;

/** @brief Representation of a substitution failure.
 *
 *  This follows an idiom I first encountered in _The C++ Programming
 *  Language_ by Bjarne Stroustrop.
 */
struct SubstitutionFailure;

/** @brief Representation of a substitution success. */
template <typename T>
struct SubstitutionSuccess
{
    static constexpr bool value = true;
};

/** @brief Substitution failure is not success. */
template <>
struct SubstitutionSuccess<SubstitutionFailure>
{
    static constexpr bool value = false;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <bool B, typename ResultT> struct EnableIfT {};

template <typename ResultT>
struct EnableIfT<true, ResultT>
{
    using type = ResultT;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

}// namespace meta
}// namespace h2
#endif // H2_META_CORE_SFINAE_HPP_

#ifndef H2_META_CORE_INVOCABLE_HPP_
#define H2_META_CORE_INVOCABLE_HPP_

#include <utility>

namespace h2
{
namespace meta
{

/** @brief Test whether F can be invoked with the given arguments. */
template <typename F, typename... Args>
struct IsInvocableVT;

/** @brief Test whether F can be invoked with the given arguments. */
template <typename F, typename... Args>
inline constexpr bool IsInvocableV()
{
    return IsInvocableVT<F, Args...>::value;
}

/** @brief Test whether F can be invoked with the given arguments. */
template <typename F, typename... Args>
inline constexpr bool IsInvocable = IsInvocableV<F, Args...>();

#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace details
{

// This is a detail nobody needs to see.
template <typename F, typename... Args>
struct GetInvocationResultT
{
private:
    template <typename F_deduce, typename... Args_deduce>
    static auto check(F_deduce f, Args_deduce&&... args)
        -> decltype(f(std::forward<Args_deduce>(args)...));
    static SubstitutionFailure check(...);
public:
    using type = decltype(check(std::declval<F>(), std::declval<Args>()...));
};

template <typename F, typename... Args>
using GetInvocationResult = meta::Force<GetInvocationResultT<F, Args...>>;

}// namespace details

template <typename F, typename... Args>
struct IsInvocableVT
    : SubstitutionSuccess<details::GetInvocationResult<F, Args...>>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
}// namespace meta
}// namespace h2
#endif // H2_META_CORE_INVOCABLE_HPP_

#ifndef H2_META_CORE_VALUEASTYPE_HPP_
#define H2_META_CORE_VALUEASTYPE_HPP_

namespace h2
{
namespace meta
{

/** @brief A constexpr value represented as a type. */
template <typename T, T Value>
struct ValueAsTypeT
{
    static constexpr T value = Value;
    using value_type = T;
    using type = ValueAsTypeT;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
};

/** @brief A constexpr value represented as a type. */
template <typename T, T Value>
using ValueAsType = Force<ValueAsTypeT<T, Value>>;

/** @brief A representation of boolean `true` values as a type. */
using TrueType = ValueAsType<bool, true>;

/** @brief A representation of boolean `false` values as a type. */
using FalseType = ValueAsType<bool, false>;

}// namespace meta
}// namespace h2
#endif // H2_META_CORE_VALUEASTYPE_HPP_

#ifndef H2_META_TYPELIST_TYPELIST_HPP_
#define H2_META_TYPELIST_TYPELIST_HPP_

namespace h2
{
namespace meta
{

/** @struct TypeList
 *  @brief A basic type list.
 *
 *  Functions that act on typelists are in the tlist namespace. There
 *  are basic accessors that offer either Lisp- or Haskell-like
 *  semantics. In a post-C++11 world, Haskell semantics are probably
 *  closer to what is natural in template metaprogramming.
 *
 *  When Lisp-family semantic choices need to be made (e.g., what
 *  happens when you take the car of the empty list), the ANSI Common
 *  Lisp standard is followed.
 *
 *  When ML-family semantic choices need to be made, Haskell
 *  conventions are adopted.
 */
template <typename... Ts>
struct TypeList;

/** @brief A short-hand alias for TypeLists. */
template <typename... Ts>
using TL = TypeList<Ts...>;

/** @brief Basic metamethods on TypeLists. */
namespace tlist
{
/** @brief The empty list. */
using Empty = TypeList<>;

/** @brief The empty list. */
using Nil = Empty;

}// namespace tlist

// Implementation

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// This gives typelists boolean value semantics. It's not clear if
// this matters.
template <typename... Ts>
struct TypeList : TrueType {};

template <>
struct TypeList<> : FalseType {};

#endif // DOXYGEN_SHOULD_SKIP_THIS

}// namespace meta
}// namespace h2
#endif // H2_META_TYPELIST_TYPELIST_HPP_

#ifndef H2_META_TYPELIST_LISPACCESSORS_HPP_
#define H2_META_TYPELIST_LISPACCESSORS_HPP_

namespace h2
{
namespace meta
{
namespace tlist
{

/** @brief The basic Cons operation.
 *  @details Prepend an item to a list.
 *  @tparam T The new item to prepend to the list
 *  @tparam List The list
 */
template <typename T, typename List>
struct ConsT;

/** @brief An appending version of the Cons operation.
 *  @details A naive lisp implementation makes this an O(n) operation;
 *           however, the nature of variadic templates allows an O(1)
 *           implementation. Thus it is provided as a convenience.
 *  @tparam T The new item to prepend to the list
 *  @tparam List The list
 */
template <typename List, typename T>
struct ConsBackT;

/** @brief Get the first item in a list. */
template <typename List>
struct CarT;

/** @brief Get a copy of the list with the first item removed. */
template <typename List>
struct CdrT;

/** @brief The basic Cons operation.
 *  @details Prepend an item to a list.
 *  @tparam T The new item to prepend to the list
 *  @tparam List The list
 */
template <typename T, typename List>
using Cons = Force<ConsT<T,List>>;

/** @brief An appending version of the Cons operation.
 *  @details Append an item to a list.
 *  @tparam List The list
 *  @tparam T The new item to prepend to the list
 */
template <typename List, typename T>
using ConsBack = Force<ConsBackT<List,T>>;

/** @brief Get the first item in a list
 *  @tparam List The list.
 */
template <typename List>
using Car = Force<CarT<List>>;

/** @brief Get a copy of the list with the first item removed
 *  @tparam List The list
 */
template <typename List>
using Cdr = Force<CdrT<List>>;

// A few Lisp-y things. The CL spec goes out to 4 operations.

// 2 operations
template <typename List> using Caar = Car<Car<List>>;
template <typename List> using Cadr = Car<Cdr<List>>;
template <typename List> using Cdar = Cdr<Car<List>>;
template <typename List> using Cddr = Cdr<Cdr<List>>;

// 3 operations
template <typename List> using Caaar = Car<Caar<List>>;
template <typename List> using Caadr = Car<Cadr<List>>;
template <typename List> using Cadar = Car<Cdar<List>>;
template <typename List> using Cdaar = Cdr<Caar<List>>;
template <typename List> using Caddr = Car<Cddr<List>>;
template <typename List> using Cddar = Cdr<Cdar<List>>;
template <typename List> using Cdadr = Cdr<Cadr<List>>;
template <typename List> using Cdddr = Cdr<Cddr<List>>;

// 4 operations
template <typename List> using Caaaar = Car<Caaar<List>>;
template <typename List> using Caaadr = Car<Caadr<List>>;
template <typename List> using Caadar = Car<Cadar<List>>;
template <typename List> using Cadaar = Car<Cdaar<List>>;
template <typename List> using Cdaaar = Cdr<Caaar<List>>;
template <typename List> using Caaddr = Car<Caddr<List>>;
template <typename List> using Cadadr = Car<Cdadr<List>>;
template <typename List> using Cdaadr = Cdr<Caadr<List>>;
template <typename List> using Cdadar = Cdr<Cadar<List>>;
template <typename List> using Cddaar = Cdr<Cdaar<List>>;
template <typename List> using Caddar = Car<Cddar<List>>;
template <typename List> using Cadddr = Car<Cdddr<List>>;
template <typename List> using Cdaddr = Cdr<Caddr<List>>;
template <typename List> using Cddadr = Cdr<Cdadr<List>>;
template <typename List> using Cdddar = Cdr<Cddar<List>>;
template <typename List> using Cddddr = Cdr<Cdddr<List>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Cons
template <typename T, typename... Ts>
struct ConsT<T,TypeList<Ts...>>
{
    using type = TypeList<T,Ts...>;
};

// ConsBack
template <typename T, typename... Ts>
struct ConsBackT<TypeList<Ts...>, T>
{
    using type = TypeList<Ts..., T>;
};

// Car
template <typename T, typename... Ts>
struct CarT<TypeList<T,Ts...>>
{
    using type = T;
};

template <>
struct CarT<Empty>
{
    using type = Nil;
};

// Cdr
template <typename T, typename... Ts>
struct CdrT<TypeList<T, Ts...>>
{
    using type = TypeList<Ts...>;
};

template <>
struct CdrT<Empty>
{
    using type = Empty;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
}// namespace tlist
}// namespace meta
}// namespace h2
#endif // H2_META_TYPELIST_LISPACCESSORS_HPP_

#ifndef H2_META_TYPELIST_APPEND_HPP_
#define H2_META_TYPELIST_APPEND_HPP_

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Join multiple lists into one. */
template <typename... Lists>
struct AppendT;

/** @brief Join multiple lists into one */
template <typename... Lists>
using Append = Force<AppendT<Lists...>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Single list
template <typename... ListTs>
struct AppendT<TL<ListTs...>>
{
    using type = TL<ListTs...>;
};

// Two lists
template <typename... ListOneTs, typename... ListTwoTs>
struct AppendT<TypeList<ListOneTs...>, TypeList<ListTwoTs...>>
{
    using type = TL<ListOneTs..., ListTwoTs...>;
};

// Many lists
template <typename FirstList, typename... OtherLists>
struct AppendT<FirstList, OtherLists...>
    : AppendT<FirstList, Append<OtherLists...>>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
#endif // H2_META_TYPELIST_APPEND_HPP_

#ifndef H2_META_TYPELIST_EXPAND_HPP_
#define H2_META_TYPELIST_EXPAND_HPP_

namespace h2
{
namespace meta
{
namespace tlist
{

/** @brief Expand a template and parameters into a typelist */
template <template <typename> class UnaryT, typename... Ts>
struct ExpandT;

/** @brief Expand a template and parameters into a typelist */
template <template <typename> class UnaryT, typename... Ts>
using Expand = Force<ExpandT<UnaryT, Ts...>>;

/** @brief Expand a template and parameters stored in a typelist into
 *  a typelist.
 */
template <template <typename> class UnaryT, typename TList>
struct ExpandTLT;

/** @brief Expand a template and parameters stored in a typelist into
 *  a typelist.
 */
template <template <typename> class UnaryT, typename TList>
using ExpandTL = Force<ExpandTLT<UnaryT, TList>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <template <typename> class UnaryT, typename... Ts>
struct ExpandT
{
    using type = TL<UnaryT<Ts>...>;
};

template <template <typename> class UnaryT, typename... Ts>
struct ExpandTLT<UnaryT, TL<Ts...>>
{
    using type = Expand<UnaryT, Ts...>;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
}// namespace tlist
}// namespace meta
}// namespace h2
#endif // H2_META_TYPELIST_EXPAND_HPP_

#ifndef H2_META_TYPELIST_MEMBER_HPP_
#define H2_META_TYPELIST_MEMBER_HPP_

namespace h2
{
namespace meta
{
namespace tlist
{

/** @brief Determine if T is a member of List. */
template <typename T, typename List>
struct MemberVT;

/** @brief Determine if T is a member of List. */
template <typename T, typename List>
constexpr bool MemberV() { return MemberVT<T, List>::value; }

template <typename T, typename List>
inline constexpr bool Member = MemberV<T,List>();

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Base case
template <typename T>
struct MemberVT<T, Empty>
  : FalseType
{};

// Match case
template <typename T, typename... Ts>
struct MemberVT<T, TL<T, Ts...>>
  : TrueType
{};

// Recursive case
template <typename T, typename Head, typename... Tail>
struct MemberVT<T, TL<Head, Tail...>>
  : MemberVT<T, TL<Tail...>>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
}// namespace tlist
}// namespace meta
}// namespace h2
#endif // H2_META_TYPELIST_MEMBER_HPP_

#ifndef H2_MULTIMETHODS_SWITCHDISPATCHER_HPP_
#define H2_MULTIMETHODS_SWITCHDISPATCHER_HPP_

namespace h2
{
namespace multimethods
{

/** @brief Dispatch a functor call based on the dynamic type of the arguments.
 *
 *  @tparam FunctorT The type of the functor to dispatch. It must
 *          implement `operator()`. All overloads must have the same
 *          return type.
 *  @tparam ReturnT The return type of all overloads of `operator()`.
 *  @tparam ArgumentTs The types of the arguments to the
 *          functor. Arguments that are part of a `BaseTypesPair` will
 *          undergo dynamic deduction.
 *
 *  @section switch-dispatch-intro Introduction
 *
 *  The problem of multiple dispatch is that, occasionally, objects
 *  need to interact at the public API level via references to their
 *  base class(es) but have implementations that vary based on the
 *  concrete (dynamic) types of the objects. Handling this dispatch
 *  manually is messy, prone to duplication, and difficult to
 *  maintain. This dispatcher implements one solution to this problem
 *  by deducing the dynamic type of certain types of arguments in a
 *  brute-force fashion. That is, code is generated for all possible
 *  combinations of dynamically-deduced types, though some may end in
 *  exceptions being thrown if no viable dispatch is found.
 *
 *  @section switch-dispatch-algo Algorithm
 *
 *  This implements a "switch-on-type" approach to multiple dispatch,
 *  and it can handle any number of dynamically-deduced arguments. The
 *  type of each argument is determined in order, first to last, by
 *  checking a user-provided list of possible dynamic types. The
 *  checks are done by use of `dynamic_cast`, so there is extensive
 *  use of Runtime Type Information (RTTI). If an argument's dynamic
 *  type cannot be deduced, it is left to the user to handle dispatch
 *  errors. How that is handled is entirely outside the scope of this
 *  dispatcher; for more information, see @ref
 *  switch-dispatch-usage-functor "the expections on functors".
 *
 *  @subsection switch-dispatch-algo-inspiration Inspiration
 *
 *  This is inspired by the StaticDispatcher in _Modern C++ Design_ by
 *  Alexei Alexandrescu, with some improvements for modern C++
 *  standards. Most notably, this seamlessly handles an arbitrary
 *  number of arguments, whereas that reference only demonstrates
 *  double dispatch, the two-argument case. This also admits
 *  additional "unclosed" arguments (i.e., not held as members in the
 *  functor), though this is somewhat clunky and not strictly
 *  necessary (because they could just be closed in the functor).
 *
 *  @section switch-dispatch-usage Usage
 *
 *  Multiple dispatch should always be hidden in at least one layer of
 *  indirection and should not be part of a public implementation
 *  (i.e., "client code"). There are two components to using this
 *  dispatcher, preparation of the functor and the multiple dispatch
 *  call site. These are covered in more detail below.
 *
 *  @subsection switch-dispatch-usage-functor Functor Preparation
 *
 *  This section details the requirements on the functor that is
 *  passed into the dispatcher.
 *
 *  The dispatcher is responsible for determining the dynamic type of
 *  each "virtual" argument; there is no way for it to dispatch
 *  directly to an overloaded function (since function names are not
 *  first-class symbols as they are in, say, LISP languages). Thus we
 *  take the standard approach of adding a layer of indirection,
 *  namely running dispatch through an object with suitably overloaded
 *  member functions. This object is a "functor" (Alexandrescu calls
 *  them "executors"), a callable object.
 *
 *  The functor is required to have `operator()` implemented for every
 *  combination of types that is dispatchable. For dispatch to have
 *  guaranteed success, the overload set must contain every possible
 *  combination of types from the given typelists, and every possible
 *  dynamic type for each argument must be present in the given
 *  typelists. Additionally, each overload must have the same return
 *  type. Note that templates or "partially dynamically-typed"
 *  overloads are able to cover various cases, as needed. For example,
 *  if (some of) the overload set is already available as free
 *  functions, a template would be an easy way to thunk the dispatch
 *  to these free functions.
 *
 *  While it is *strongly* encouraged to treat the functor as a
 *  closure around the non-deduced arguments, it is possible to expose
 *  additional "unenclosed" arguments that are not deduced in the
 *  functor interface. These arguments must be positioned *before* the
 *  deduced arguments in formal argument list for `operator()`. For
 *  example, the following is a valid use of an additional argument:
 *
 *  @code{.cpp}
 *  struct MyFunctor {
 *    void operator()(int x, deduced& a, deduced& b) {...}
 *  };
 *  @endcode
 *
 *  The following is an *invalid* use of an additional argument:
 *
 *  @code{.cpp}
 *  struct MyFunctor {
 *    // ERROR: Additional argument splits deduced arguments
 *    void operator()(deduced& a, int x, deduced& b) {...}
 *    // ERROR: Additional argument follows deduced arguments
 *    void operator()(deduced& a, deduced& b, int x) {...}
 *  };
 *  @endcode
 *
 *  The reason for this restriction is technical, and may be lifted in
 *  the future. Note that the ordering of formal arguments to the
 *  dispatcher will be given in a @ref
 *  switch-dispatch-usage-call-site-arguments "different order".
 *
 *  @subsubsection switch-dispatch-usage-functor-errors Error handling
 *
 *  Handling errors is deferred to the functor as well. There are two
 *  types of possible errors that can come out of the dynamic dispatch
 *  process, and the functor class must provide a mechanism for
 *  dealing with each of them.
 *
 *  First, the dynamic type of an argument might not be found in that
 *  argument's typelist. For this, the functor is required to provide
 *  the function `ReturnT DeductionError(...)`. Currently, the
 *  argument list must be variadic; this is a detail of the dispatch
 *  engine that is being ironed out and will hopefully disappear. When
 *  that happens, the requirement will be "... the function `ReturnT
 *  DeductionError(base_typed_signature)`.
 *
 *  Second, the functor may not be callable with the deduced
 *  types. The functor is required to provide a function equivalent to
 *  `ReturnT DispatchError(Args)` in this case, where `Args` matches
 *  the argument list for `operator()` with dynamically-deduced
 *  arguments replaced by their respective base-class references. More
 *  complex techniques (such as templates) could also be used to
 *  provide more detailed functionality.
 *
 *  Ultimately, what happens inside these error-handling functions is
 *  up to the implementation of the functor; no expection or
 *  requirement is imposed by this dispatcher. That is, these cases
 *  are only known to be errors with respect to the dynamic dispatch
 *  engine; it is use-case-specific whether this constitutes a program
 *  error. These functions merely provide a signal to the functor that
 *  this has situation has occurred.
 *
 *  It is important to note that these functions are always required
 *  to be present in a functor. There may be particular use-cases of
 *  this dispatcher that can be implemented such that these cases
 *  cannot occur at runtime; the error functions are still required to
 *  be present. They may be empty.
 *
 *  @subsection switch-dispatch-usage-call-site Call-site Particulars
 *
 *  This section details the use-patterns and idiosyncracies of using
 *  this dispatcher to achieve multiple dispatch.
 *
 *  It bears repeating that this dispatch engine does not directly
 *  operate on overloaded functions; it requires @ref
 *  switch-dispatch-usage-functor "functors with special structure".
 *  Once that has been designed as described, usage is
 *  straight-forward. First, the template arguments to the dispatcher
 *  must be created. Then, the arguments to the dispatcher must be
 *  ordered correctly.
 *
 *  @subsubsection switch-dispatch-usage-call-site-tparams Template Parameters
 *
 *  For a functor with `N` dynamically-deduced arguments, there will
 *  be `2+2*N` template parameters to the dispatcher. The first two
 *  are very simple: the type of the functor and the type that is
 *  returned by its `operator()` (or the overload set that will be
 *  exploited in this dispatch). Following that, the remaining `2*N`
 *  arguments must be given in pairs: first a base type, then a list
 *  of concrete types against which to test the formal argument. These
 *  must be given in the same order as the dynamically-deduced formal
 *  arguments, and there must be one pair for each formal argument,
 *  even if that means repeating pairs. This may be optimized away in
 *  the future.
 *
 *  @subsubsection switch-dispatch-usage-call-site-arguments Formal Arguments
 *
 *  The dispatcher exposes a single static API: `Exec(...)`. This
 *  function has return type as specified in the template
 *  parameters. The arguments are as follows:
 *
 *    -# A functor object, by value.
 *    -# The arguments that will be dynamically deduced, in the same
 *       order that they will be passed to the functor's `operator()`.
 *    -# The extra "unclosed" arguments, in the same order that they will
 *       be passed to the functor's `operator()`.
 *
 *  Note that these last two groups are ordered differently than when
 *  implementing the functor. This is intentional. Work is in-progress
 *  to resolve this confusion.
 *
 *  @warning This method of multiple dispatch is robust, but it relies
 *  on `dynamic_cast` to check the type of each argument. This heavy
 *  use of RTTI could affect performance if not used carefully. It is
 *  left to users of this dispatch engine to determine whether this
 *  cost is acceptable. In general, it is advisable to avoid multiple
 *  dispatch issues inside tight loops and other performance-critical
 *  sections.
 *
 *  @warning If the functor is implemented using templates, this could
 *  implicitly instantiate all combinations of parameters if care has
 *  not been taken to prevent this. If this incurs too high a
 *  compilation cost, perhaps consider controlling instantiation via
 *  explicit template instantiation, using ETI declarations where
 *  appropriate.
 *
 */
template <
    typename FunctorT,
    typename ReturnT,
    typename... ArgumentTs>
class SwitchDispatcher;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <
    typename FunctorT,
    typename ReturnT,
    typename ThisBase,
    typename ThisList,
    typename... ArgumentTs>
class SwitchDispatcher<FunctorT, ReturnT,
                       ThisBase, ThisList,
                       ArgumentTs...>
{
    static_assert(sizeof...(ArgumentTs) % 2 == 0,
                  "Must pass ArgumentTs as (Base, TL<DTypes>).");

public:
    template <typename... Args>
    static ReturnT Exec(FunctorT F, ThisBase& arg, Args&&... others)
    {
        using Head = meta::tlist::Car<ThisList>;
        using Tail = meta::tlist::Cdr<ThisList>;

        if (auto* arg_dc = dynamic_cast<Head*>(&arg))
            return SwitchDispatcher<FunctorT, ReturnT, ArgumentTs...>::
                Exec(F, std::forward<Args>(others)..., *arg_dc);
        else
            return SwitchDispatcher<FunctorT, ReturnT,
                                    ThisBase, Tail,
                                    ArgumentTs...>::
                Exec(F, arg, std::forward<Args>(others)...);
    }
};

// Base case
template <
    typename FunctorT,
    typename ReturnT>
class SwitchDispatcher<FunctorT, ReturnT>
{
    template <typename... Ts>
    using Invocable = meta::IsInvocableVT<FunctorT, Ts...>;

public:
    template <typename... Args,
              meta::EnableWhenV<Invocable<Args...>,int> = 0>
    static ReturnT Exec(FunctorT F, Args&&... others)
    {
        return F(std::forward<Args>(others)...);
    }

    // All types were deduced, but there is no suitable dispatch for
    // this case.
    template <typename... Args,
              meta::EnableUnlessV<Invocable<Args...>,int> = 0>
    static ReturnT Exec(FunctorT F, Args&&... args)
    {
        return F.DispatchError(std::forward<Args>(args)...);
    }
};

// Deduction failure case
template <
    typename FunctorT,
    typename ReturnT,
    typename ThisBase,
    typename... ArgumentTs>
class SwitchDispatcher<FunctorT, ReturnT,
                       ThisBase, meta::tlist::Empty,
                       ArgumentTs...>
{
public:
    template <typename... Args>
    static ReturnT Exec(FunctorT F, Args&&... args)
    {
        return F.DeductionError(std::forward<Args>(args)...);
    }
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

}// namespace multimethods
}// namespace h2
#endif // H2_MULTIMETHODS_SWITCHDISPATCHER_HPP_

#endif // LBANN_HAS_DIHYDROGEN
#endif // LBANN_UTILS_H2_TMP_HPP_
