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

#ifndef LBANN_UTILS_H2_TMP_IMPL_HPP_
#define LBANN_UTILS_H2_TMP_IMPL_HPP_

#include "lbann/utils/h2_tmp.hpp"

#ifdef LBANN_HAS_DIHYDROGEN

#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>

#else // !LBANN_HAS_DIHYDROGEN

#ifndef H2_MULTIMETHODS_SWITCHDISPATCHER_HPP_
#define H2_MULTIMETHODS_SWITCHDISPATCHER_HPP_

namespace h2
{
namespace multimethods
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <
    typename FunctorT,
    typename ReturnT,
    typename ThisBase,
    typename ThisList,
    typename... ArgumentTs>
template <typename... Args>
ReturnT
SwitchDispatcher<FunctorT, ReturnT,
                       ThisBase, ThisList,
                       ArgumentTs...>
::Exec(FunctorT F, ThisBase& arg, Args&&... others)
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

// Base case
template <
    typename FunctorT,
    typename ReturnT>
template <typename... Args,
          meta::EnableWhenV<Invocable<Args...>,int> = 0>
ReturnT
SwitchDispatcher<FunctorT, ReturnT>
::Exec(FunctorT F, Args&&... others)
{
    return F(std::forward<Args>(others)...);
}

// All types were deduced, but there is no suitable dispatch for
// this case.
template <
    typename FunctorT,
    typename ReturnT>
template <typename... Args,
          meta::EnableUnlessV<Invocable<Args...>,int> = 0>
ReturnT
SwitchDispatcher<FunctorT, ReturnT>
::Exec(FunctorT F, Args&&... args)
{
    return F.DispatchError(std::forward<Args>(args)...);
}

// Deduction failure case
template <
    typename FunctorT,
    typename ReturnT,
    typename ThisBase,
    typename... ArgumentTs>
template <typename... Args>
ReturnT
SwitchDispatcher<FunctorT, ReturnT,
                       ThisBase, meta::tlist::Empty,
                       ArgumentTs...>
::Exec(FunctorT F, Args&&... args)
{
    return F.DeductionError(std::forward<Args>(args)...);
}

#endif // DOXYGEN_SHOULD_SKIP_THIS

}// namespace multimethods
}// namespace h2
#endif // H2_MULTIMETHODS_SWITCHDISPATCHER_HPP_

#endif // LBANN_HAS_DIHYDROGEN
#endif // LBANN_UTILS_H2_TMP_IMPL_HPP_
