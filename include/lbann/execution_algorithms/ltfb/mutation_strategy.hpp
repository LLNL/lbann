////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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
#ifndef LBANN_EXECUTION_ALGORITHMS_LTFB_MUTATION_STRATEGY_HPP_INCLUDED
#define LBANN_EXECUTION_ALGORITHMS_LTFB_MUTATION_STRATEGY_HPP_INCLUDED

#include "lbann/models/model.hpp"
#include "lbann/utils/cloneable.hpp"

namespace lbann {
namespace ltfb {

class MutationStrategy : public Cloneable<HasAbstractFunction<MutationStrategy>>
{
public:
  MutationStrategy(){};
  virtual ~MutationStrategy() = default;

  /** @brief Apply a change to the model.
   *  @param[in,out] m The model to change.
   *  @param[in] step The current execution step in LTFB
   */
  virtual void mutate(model& m, const int& step) = 0;
};

// No Mutation
class NullMutation final : public Cloneable<NullMutation, MutationStrategy>
{

public:
  NullMutation() = default;
  void mutate(model& m, const int& step) final {}
};

// Replace activation layers
class ReplaceActivation final
  : public Cloneable<ReplaceActivation, MutationStrategy>
{
public:
  ReplaceActivation() = default;
  void mutate(model& m, const int& step) final;
};

} // namespace ltfb
} // namespace lbann
#endif // LBANN_EXECUTION_ALGORITHMS_LTFB_MUTATION_STRATEGY_HPP_INCLUDED
