////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
#include <catch2/catch.hpp>
#include "TestHelpers.hpp"
#include "lbann/utils/cloneable.hpp"
#include <memory>

using namespace lbann;

struct Base : Cloneable<NonLeafClass<Base>>
{
};
struct Derived : Cloneable<Derived, Base>
{
};

struct DiamondBase : Cloneable<NonLeafClass<DiamondBase>>
{
};
struct DiamondDerivedLeft : Cloneable<HasAbstractFunction<DiamondDerivedLeft>,
                                      AsVirtualBase<DiamondBase>>
{
};
struct DiamondDerivedRight : Cloneable<HasAbstractFunction<DiamondDerivedRight>,
                                       AsVirtualBase<DiamondBase>>
{
};
struct DiamondDerivedBottom
  : Cloneable<DiamondDerivedBottom, DiamondDerivedLeft, DiamondDerivedRight>
{
};

struct StandaloneCloneable : Cloneable<StandaloneCloneable>
{
};

TEST_CASE("Testing cloneable mechanism -- standalone class", "[utils]")
{
  StandaloneCloneable a;
  std::unique_ptr<StandaloneCloneable> bptr = a.clone();
  REQUIRE(bptr);
}

TEST_CASE("Testing cloneable mechanism -- shallow hierarchy", "[utils]")
{
  Derived a;
  std::unique_ptr<Derived> bptr = a.clone();
  REQUIRE(bptr);
  std::unique_ptr<Base> cptr = bptr->clone();
  REQUIRE(cptr);
  std::unique_ptr<Base> dptr = cptr->clone();
  REQUIRE(dptr);
}

TEST_CASE("Testing cloneable mechanism -- inheritance diamond", "[utils]")
{
  DiamondDerivedBottom a;
  std::unique_ptr<DiamondDerivedBottom> bptr = a.clone();
  REQUIRE(bptr);
  std::unique_ptr<DiamondDerivedLeft> cptr = a.clone();
  REQUIRE(cptr);
  std::unique_ptr<DiamondDerivedRight> dptr = a.clone();
  REQUIRE(dptr);
  // This call will fail if the shared base class is not properly resolved.
  std::unique_ptr<DiamondBase> eptr = a.clone();
  REQUIRE(eptr);

  REQUIRE(dynamic_cast<DiamondDerivedLeft const*>(eptr.get()));
  REQUIRE(dynamic_cast<DiamondDerivedRight const*>(eptr.get()));
  REQUIRE(dynamic_cast<DiamondDerivedBottom const*>(eptr.get()));
}

struct DeepBase : Cloneable<HasAbstractFunction<DeepBase>>
{
};
struct DeepMidDerived : Cloneable<HasAbstractFunction<DeepMidDerived>, DeepBase>
{
};
struct DeepMostDerived : Cloneable<DeepMostDerived, DeepMidDerived>
{
};

TEST_CASE("Testing cloneable mechanism -- deep hierarchy", "[utils]")
{
  DeepMostDerived a;
  std::unique_ptr<DeepMostDerived> bptr = a.clone();
  REQUIRE(bptr);
  std::unique_ptr<DeepMidDerived> cptr = a.clone();
  REQUIRE(cptr);
  std::unique_ptr<DeepBase> dptr = a.clone();
  REQUIRE(dptr);

  std::unique_ptr<DeepMidDerived> eptr = cptr->clone();
  REQUIRE(eptr);
  std::unique_ptr<DeepBase> fptr = eptr->clone();
  REQUIRE(fptr);

  std::unique_ptr<DeepBase> gptr = fptr->clone();
  REQUIRE(gptr);
  REQUIRE(dynamic_cast<DeepMidDerived const*>(gptr.get()));
  REQUIRE(dynamic_cast<DeepMostDerived const*>(gptr.get()));
}

// Static unit tests

static_assert(IsCloneable_v<Cloneable<Base>>(),
              "Cloneable<...> should be cloneable");
static_assert(IsCloneable_v<Cloneable<NonLeafClass<Base>>>(),
              "Cloneable<...> should be cloneable");

// All of the tested classes above should be cloneable
static_assert(IsCloneable_v<Base>(), "Base should be cloneable.");
static_assert(IsCloneable_v<Derived>(), "Derived should be cloneable.");

static_assert(IsCloneable_v<StandaloneCloneable>(),
              "StandaloneCloneable should be cloneable.");

static_assert(IsCloneable_v<DiamondBase>(), "DiamondBase should be cloneable.");
static_assert(IsCloneable_v<DiamondDerivedLeft>(),
              "DiamondDerivedLeft should be cloneable.");
static_assert(IsCloneable_v<DiamondDerivedRight>(),
              "DiamondDerivedRight should be cloneable.");
static_assert(IsCloneable_v<DiamondDerivedBottom>(),
              "DiamondDerivedBottom should be cloneable.");

static_assert(IsCloneable_v<DeepBase>(), "DeepBase should be cloneable.");
static_assert(IsCloneable_v<DeepMidDerived>(),
              "DeepMidDerived should be cloneable.");
static_assert(IsCloneable_v<DeepMostDerived>(),
              "DeepMostDerived should be cloneable.");

// Basic types should not be cloneable
static_assert(!IsCloneable_v<int>(), "int should not be cloneable.");
static_assert(!IsCloneable_v<char>(), "char should not be cloneable.");
static_assert(!IsCloneable_v<float>(), "float should not be cloneable.");
static_assert(!IsCloneable_v<double>(), "double should not be cloneable.");

// A class that returns by raw pointer doesn't fit the Cloneable
// concept as defined here.
struct NotTheRightCloneable
{
  NotTheRightCloneable* clone() const;
};

static_assert(!IsCloneable_v<NotTheRightCloneable>(),
              "NotTheRightCloneable should not be cloneable.");

// The predicate is not able to catch degenerate cases that happen to
// provide the right clone signature but don't use the cloneable
// method. This would be difficult to enforce as well, especially
// since the Cloneable infrastructure could be written out long-hand.
struct ShouldBeOk
{
  std::unique_ptr<ShouldBeOk> clone() const;
};

static_assert(IsCloneable_v<ShouldBeOk>(), "ShouldBeOk should be cloneable.");

// The predicate should be able to handle the naive
// "polymorphic-clone-to-unique_ptr" approach. Of course the base
// class will not be caught, but derived classes will.
struct NoCovariantCloneBase
{
  virtual ~NoCovariantCloneBase() = default;
  virtual std::unique_ptr<NoCovariantCloneBase> clone() const = 0;
};
struct NoCovariantCloneConcrete : NoCovariantCloneBase
{
  std::unique_ptr<NoCovariantCloneBase> clone() const override;
};

static_assert(IsCloneable_v<NoCovariantCloneBase>(),
              "NoCovariantCloneBase should appear to be cloneable.");
static_assert(!IsCloneable_v<NoCovariantCloneConcrete>(),
              "NoCovariantCloneConcrete should not be cloneable.");

// Cloning all objects in a contain full of cloneable objects.

TEST_CASE("Testing cloneable mechanism -- vector clone", "[utils]")
{
  SECTION("Unique Ptr")
  {
    using container_type = std::vector<std::unique_ptr<DeepBase>>;
    container_type ptrs;
    ptrs.push_back(std::make_unique<DeepMostDerived>());
    ptrs.push_back(std::make_unique<DeepMostDerived>());
    ptrs.push_back(std::make_unique<DeepMostDerived>());

    auto cloned_ptrs = clone_all(ptrs);
    static_assert(std::is_same_v<decltype(cloned_ptrs), container_type>, "");
    for (auto const& p : cloned_ptrs)
      CHECK(unit_test::utilities::IsValidPtr(p));
  }
  SECTION("Shared Ptr")
  {
    using container_type = std::vector<std::shared_ptr<DeepBase>>;
    container_type ptrs;
    ptrs.push_back(std::make_unique<DeepMostDerived>());
    ptrs.push_back(std::make_unique<DeepMostDerived>());
    ptrs.push_back(std::make_unique<DeepMostDerived>());

    auto cloned_ptrs = clone_all(ptrs);
    static_assert(std::is_same_v<decltype(cloned_ptrs), container_type>, "");
    for (auto const& p : cloned_ptrs)
      CHECK(unit_test::utilities::IsValidPtr(p));
  }
}

// Sanity check the IsCloneablePtr predicate.
static_assert(IsCloneablePtr<DeepMidDerived*>,
              "DeepMidDerived supports Cloneable.");
static_assert(IsCloneablePtr<DeepMidDerived const*>,
              "DeepMidDerived supports Cloneable.");
static_assert(IsCloneablePtr<std::shared_ptr<DeepMostDerived>>,
              "DeepMostDerived supports Cloneable.");
static_assert(IsCloneablePtr<std::shared_ptr<DeepMostDerived const>>,
              "DeepMostDerived supports Cloneable.");
static_assert(IsCloneablePtr<std::unique_ptr<DeepBase>>,
              "DeepBase supports Cloneable.");
static_assert(IsCloneablePtr<std::unique_ptr<DeepBase const>>,
              "DeepBase supports Cloneable.");

static_assert(!IsCloneablePtr<int*>, "int does not support Cloneable.");
static_assert(!IsCloneablePtr<std::unique_ptr<char[]>>,
              "char does not support Cloneable.");
