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
// Must include this for all the Catch2 machinery
#include "Catch2BasicSupport.hpp"

// Some common infrastructure for testing optimizers
#include "optimizer_common.hpp"

// Classes under test
#include <lbann/optimizers/sgd.hpp>

// Needed for doing diskless testing.
#include <sstream>

// This test file is highly annotated and is meant to serve as the
// exemplar of the serialization tests. Subsequent testing will be
// more compact.

namespace {

// We need a way to construct objects. One must at least pretend
// to have meaningful state, and the other must have some "default"
// state (really, it just has to be different from the "meaningful
// state" so that we can detect a change in the restored value).
//
// We accomplish this using little builder functor templates to handle
// the TensorDataType. Since there are no optimzer classes with common
// construction arguments, we do this on an optimizer-specific basis.
//
// An important note here is that safely serializable values are more
// important than "realistic values" for this sort of test. Obviously
// for "binary" mode, any serialization/deserialization should be
// exact. But for the text modes, there's no guarantee they will be
// serialized and deserialized to any precision. We don't want the
// test to fail because of rounding of ASCII-read string-to-numbers.
//
// If it would be better to save/load from
// "unique/shared-pointer-to-base" instead, these functors could
// easily be modified to accommodate that instead.

template <typename TensorDataType>
struct SGDBuilder
{
  static lbann::sgd<TensorDataType> Stateful()
  {
    return lbann::sgd<TensorDataType>(
      /*learning_rate=*/TensorDataType(2.f),
      /*momentum=*/TensorDataType(3.f),
      /*nesterov=*/true);
  }

  static lbann::sgd<TensorDataType> Default()
  {
    return lbann::sgd<TensorDataType>(
      /*learning_rate=*/TensorDataType(0.0f),
      /*momentum=*/TensorDataType(0.0f),
      /*nesterov=*/false);
  }
}; // struct SGDBuilder

} // namespace

TEMPLATE_LIST_TEST_CASE("SGD Optimizer serialization",
                        "[optimizer][serialize]",
                        AllArchiveTypes)
{
  using ValueType = tlist::Car<TestType>;

  using ArchiveTypes = tlist::Cdr<TestType>;
  using OutputArchiveType = tlist::Car<ArchiveTypes>;
  using InputArchiveType = tlist::Cadr<ArchiveTypes>;

  using OptimizerType = lbann::sgd<ValueType>;
  using BuilderType = SGDBuilder<ValueType>;

  std::stringstream ss;

  OptimizerType opt = BuilderType::Stateful();
  OptimizerType opt_restore = BuilderType::Default();

  // Verify that the optimizers differ in the first place.
  CHECK_FALSE(opt.get_learning_rate() == opt_restore.get_learning_rate());
  CHECK_FALSE(opt.get_momentum() == opt_restore.get_momentum());
  CHECK_FALSE(opt.using_nesterov() == opt_restore.using_nesterov());

  {
    OutputArchiveType oarchive(ss);
    CHECK_NOTHROW(oarchive(opt));
  }

  {
    InputArchiveType iarchive(ss);
    CHECK_NOTHROW(iarchive(opt_restore));
  }

  // Verify that the restoration was successful.
  CHECK(opt.get_learning_rate() == opt_restore.get_learning_rate());
  CHECK(opt.get_momentum() == opt_restore.get_momentum());
  CHECK(opt.using_nesterov() == opt_restore.using_nesterov());
}
