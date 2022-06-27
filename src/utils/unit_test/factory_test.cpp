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
// Be sure to include this!
#include "Catch2BasicSupport.hpp"

// The code being tested
#include <lbann/utils/factory.hpp>

// Other includes
#include <lbann/utils/memory.hpp>

namespace
{
struct widget_base {
    virtual ~widget_base() = default;
};
struct widget : widget_base {};
struct gizmo : widget_base {};
}

enum class generic_key
{
  INVALID,
  WIDGET,
  GIZMO
};

template <typename T> struct Key;

template <>
struct Key<std::string>
{
  static std::string get(generic_key key)
  {
    switch (key)
    {
    case generic_key::WIDGET:
      return "widget";
    case generic_key::GIZMO:
      return "gizmo";
    case generic_key::INVALID:
      return "invalid";
    }
    return "";
  }
};

template <>
struct Key<int>
{
  static int get(generic_key key) noexcept
  {
    return static_cast<int>(key);
  }
};

// This tests factories keyed with strings and ints. BDD-style
// nomenclature is used inside the test case.
TEMPLATE_TEST_CASE(
  "testing the factory class", "[factory][utilities]", std::string, int)
{
  using widget_factory
    = lbann::generic_factory<widget_base,TestType>;
  using key = Key<TestType>;

  GIVEN("an object factory")
  {
    widget_factory factory;

    WHEN("Two new builders are registered")
    {
      factory.register_builder(
        key::get(generic_key::WIDGET),[]()
        {
          return std::unique_ptr<widget_base>(
            std::make_unique<widget>());
        });

      factory.register_builder(
        key::get(generic_key::GIZMO),[]()
        {
          return std::unique_ptr<widget_base>(
            std::make_unique<gizmo>());
        });

      THEN("The factory knows about two builders")
      {
        auto names = factory.registered_ids();
        REQUIRE(std::distance(names.begin(), names.end()) == 2UL);
      }
      AND_WHEN("A builder is added with an existing key")
      {
        factory.register_builder(
          key::get(generic_key::GIZMO),[]()
          {
            return std::unique_ptr<widget_base>(
              std::make_unique<gizmo>());
          });

        THEN("The factory still knows about only two factories")
        {
          auto names = factory.registered_ids();
          REQUIRE(std::distance(names.begin(), names.end()) == 2UL);
        }
      }

      AND_WHEN("A new object is requested with a valid key")
      {
        auto obj = factory.create_object(key::get(generic_key::WIDGET));

        THEN("The returned object is the right type.")
        {
          widget* obj_ptr = dynamic_cast<widget*>(obj.get());
          REQUIRE(obj_ptr != nullptr);
        }
      }

      AND_WHEN("A new object is requested with with an invalid key")
      {
        THEN("An exception is thrown.")
        {
          std::unique_ptr<widget_base> obj;
          REQUIRE_THROWS_AS(
            obj = factory.create_object(key::get(generic_key::INVALID)),
            lbann::exception);
        }
      }

      AND_WHEN("A key is removed")
      {
        auto success = factory.unregister(key::get(generic_key::WIDGET));
        THEN("The number of known factories has decreased.")
        {
          REQUIRE(success == true);
          auto names = factory.registered_ids();
          REQUIRE(std::distance(names.begin(), names.end()) == 1UL);
        }

        THEN("The remaining key is still valid.")
        {
          auto obj = factory.create_object(key::get(generic_key::GIZMO));
          gizmo* obj_ptr = dynamic_cast<gizmo*>(obj.get());
          REQUIRE(obj_ptr != nullptr);
        }

        THEN("An exception is thrown when trying to create an "
             "object with a removed key.")
        {
          std::unique_ptr<widget_base> obj;
          REQUIRE_THROWS_AS(
            obj = factory.create_object(key::get(generic_key::WIDGET)),
            lbann::exception);
        }
      }
    }
  }
}
