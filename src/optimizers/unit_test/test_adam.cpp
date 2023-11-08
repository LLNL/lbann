////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
#include "Catch2BasicSupport.hpp"
#include <lbann/optimizers/adam.hpp>

#include "optimizer_common.hpp"

#include <sstream>

// See test_sgd.cpp for a detailed, annotated test case.

namespace {

template <typename TensorDataType>
struct AdamBuilder
{
  static lbann::adam<TensorDataType> Stateful()
  {
    lbann::adam<TensorDataType> ret(
      /*learning_rate=*/TensorDataType(3.f),
      /*beta1=*/TensorDataType(1.f),
      /*beta2=*/TensorDataType(4.f),
      /*eps=*/TensorDataType(2.f),
      /*adamw_weight_decay=*/TensorDataType(5.f));

    // These probably shouldn't be set here, but let's pretend
    // something's happened to perturb the state.
    ret.set_current_beta1(TensorDataType(5.f));
    ret.set_current_beta2(TensorDataType(6.f));
    return ret;
  }

  static lbann::adam<TensorDataType> Default()
  {
    return lbann::adam<TensorDataType>(
      /*learning_rate=*/TensorDataType(0.0f),
      /*beta1=*/TensorDataType(0.0f),
      /*beta2=*/TensorDataType(0.0f),
      /*eps=*/TensorDataType(0.0f),
      /*adamw_weight_decay=*/TensorDataType(0.0f));
  }
}; // struct AdamBuilder

} // namespace

TEMPLATE_LIST_TEST_CASE("Adam Optimizer serialization",
                        "[optimizer][serialize]",
                        AllArchiveTypes)
{
  using ValueType = tlist::Car<TestType>;

  using ArchiveTypes = tlist::Cdr<TestType>;
  using OutputArchiveType = tlist::Car<ArchiveTypes>;
  using InputArchiveType = tlist::Cadr<ArchiveTypes>;

  using OptimizerType = lbann::adam<ValueType>;
  using BuilderType = AdamBuilder<ValueType>;

  std::stringstream ss;

  OptimizerType opt = BuilderType::Stateful();
  OptimizerType opt_restore = BuilderType::Default();

  // Verify that the optimizers differ in the first place.
  CHECK_FALSE(opt.get_learning_rate() == opt_restore.get_learning_rate());
  CHECK_FALSE(opt.get_beta1() == opt_restore.get_beta1());
  CHECK_FALSE(opt.get_beta2() == opt_restore.get_beta2());
  CHECK_FALSE(opt.get_current_beta1() == opt_restore.get_current_beta1());
  CHECK_FALSE(opt.get_current_beta2() == opt_restore.get_current_beta2());
  CHECK_FALSE(opt.get_eps() == opt_restore.get_eps());
  CHECK_FALSE(opt.get_adamw_weight_decay() ==
              opt_restore.get_adamw_weight_decay());

  {
    OutputArchiveType oarchive(ss);
    CHECK_NOTHROW(oarchive(opt));
  }

  {
    InputArchiveType iarchive(ss);
    CHECK_NOTHROW(iarchive(opt_restore));
  }

  CHECK(opt.get_learning_rate() == opt_restore.get_learning_rate());
  CHECK(opt.get_beta1() == opt_restore.get_beta1());
  CHECK(opt.get_beta2() == opt_restore.get_beta2());
  CHECK(opt.get_current_beta1() == opt_restore.get_current_beta1());
  CHECK(opt.get_current_beta2() == opt_restore.get_current_beta2());
  CHECK(opt.get_eps() == opt_restore.get_eps());
  CHECK(opt.get_adamw_weight_decay() == opt_restore.get_adamw_weight_decay());
}
