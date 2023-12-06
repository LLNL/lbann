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

#include "MPITestHelpers.hpp"

#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>
#include <h2/patterns/multimethods/SwitchDispatcher.hpp>
#include <lbann/base.hpp>
#include <lbann/utils/memory.hpp>
#include <lbann/weights/data_type_weights.hpp>
#include <lbann/weights/weights.hpp>
#include <lbann/weights/weights_proxy.hpp>

// Some convenience typedefs

template <typename T>
using DataTypeWeights = lbann::data_type_weights<T>;

template <typename T>
using ConstantInitializer = lbann::constant_initializer<T>;

template <typename T>
using CircCirc =
  El::DistMatrix<T, El::CIRC, El::CIRC, El::ELEMENT, El::Device::CPU>;

// Helper functions

namespace {

template <typename T>
size_t count_differing_values(T const& val, El::AbstractMatrix<T> const& mat_in)
{
  El::AbstractMatrixReadDeviceProxy<T, El::Device::CPU> proxy(mat_in);
  auto const& mat = proxy.GetLocked();

  size_t nnz = 0;
  for (El::Int col = 0; col < mat.Width(); ++col)
    for (El::Int row = 0; row < mat.Height(); ++row)
      nnz += (mat.CRef(row, col) != val);
  return nnz;
}

template <typename T>
size_t count_nonzeros(El::AbstractMatrix<T> const& mat_in)
{
  return count_differing_values(El::To<T>(0.f), mat_in);
}

template <typename T>
El::AbstractMatrix<T> const& get_local_values(DataTypeWeights<T> const& dtw)
{
  return dtw.get_values().LockedMatrix();
}

template <typename T>
El::AbstractMatrix<T> const&
get_local_values(lbann::WeightsProxy<T> const& proxy)
{
  return proxy.values().LockedMatrix();
}
} // namespace

// Test to make sure I understand how weights need to be setup.
TEST_CASE("Basic weights tests", "[mpi][weights]")
{
  using DataType = float;

  auto& world_comm = unit_test::utilities::current_world_comm();
  size_t const size_of_world = world_comm.get_procs_in_world();

  // Setup the weights object -- let's hope I do this right. It's not
  // like it's documented anywhere.

  size_t const weights_height = 3 * size_of_world;
  size_t const weights_width = 2 * size_of_world;

  // Create the object
  DataTypeWeights<DataType> dtw(world_comm);

  REQUIRE_NOTHROW(dtw.set_dims({weights_height}, {weights_width}));
  CHECK(dtw.get_matrix_height() == weights_height);
  CHECK(dtw.get_matrix_width() == weights_width);

  SECTION("Setup with no initializer.")
  {
    REQUIRE_NOTHROW(dtw.setup());
    CHECK(count_nonzeros(get_local_values(dtw)) == 0UL);
  }

  SECTION("Setup with constant initializer.")
  {
    DataType const value = El::To<DataType>(1.3);
    REQUIRE_NOTHROW(dtw.set_initializer(
      std::make_unique<ConstantInitializer<DataType>>(value)));
    REQUIRE_NOTHROW(dtw.setup());
    CHECK(count_differing_values(value, get_local_values(dtw)) == 0UL);

    CHECK(dtw.get_values().Height() ==
          El::To<El::Int>(dtw.get_matrix_height()));
    CHECK(dtw.get_values().Width() == El::To<El::Int>(dtw.get_matrix_width()));
  }
}

template <typename TypePair>
using MasterType = h2::meta::tlist::Car<TypePair>;
template <typename TypePair>
using ProxyType = h2::meta::tlist::Cadr<TypePair>;

// Clean up the output slightly
template <typename MasterT, typename ProxyT>
using MasterProxyPair = h2::meta::TL<MasterT, ProxyT>;

// These grew out of a realization that an initial implementation of
// some of these functions violated the class contract in their usage
// of the internal pointers.
TEST_CASE("Empty WeightsProxy tests.", "[mpi][weights][proxy]")
{
  lbann::WeightsProxy<float> proxy;

  SECTION("Copy construction.")
  {
    lbann::WeightsProxy<float> proxy_copy(proxy);
    REQUIRE(proxy.empty());
    REQUIRE(proxy_copy.empty());
  }

  SECTION("Move construction.")
  {
    lbann::WeightsProxy<float> proxy_move(std::move(proxy));
    REQUIRE(proxy.empty());
    REQUIRE(proxy_move.empty());
  }

  SECTION("Copy assignment.")
  {
    lbann::WeightsProxy<float> proxy_copy;
    REQUIRE(proxy.empty());
    REQUIRE_NOTHROW(proxy_copy = proxy);
    REQUIRE(proxy.empty());
    REQUIRE(proxy_copy.empty());
  }
}

TEMPLATE_TEST_CASE("Weights proxy tests.",
                   "[mpi][weights][proxy]",
                   (MasterProxyPair<float, float>)
#ifdef LBANN_HAS_DOUBLE
                     ,
                   (MasterProxyPair<double, float>)
#endif // LBANN_HAS_DOUBLE
)
{
  using MasterDataType = MasterType<TestType>;
  using DataType = ProxyType<TestType>;

  auto& world_comm = unit_test::utilities::current_world_comm();
  size_t const size_of_world = world_comm.get_procs_in_world();

  // Setup the weights object
  size_t const weights_height = 3 * size_of_world;
  size_t const weights_width = 2 * size_of_world;

  // Create the master weights object.
  auto dtw = std::make_shared<DataTypeWeights<MasterDataType>>(world_comm);

  // Create and set the initializer; using a constant initializer
  // here. This must be done at the master data type.
  MasterDataType const value = El::To<MasterDataType>(5.17);
  REQUIRE_NOTHROW(dtw->set_initializer(
    std::make_unique<ConstantInitializer<MasterDataType>>(value)));

  // Set the size for the weights.
  REQUIRE_NOTHROW(dtw->set_dims({weights_height}, {weights_width}));

  // Setup the weights object.
  REQUIRE_NOTHROW(dtw->setup());

  // Phew. Start testing the proxy.
  lbann::WeightsProxy<DataType> proxy(dtw);

  SECTION("Proxy accesses values correctly.")
  {
    // Proxy should not be empty.
    REQUIRE(!proxy.empty());

    // At this point, the proxy should have the right size.
    CHECK(proxy.values().Height() == El::To<El::Int>(dtw->get_matrix_height()));
    CHECK(proxy.values().Width() == El::To<El::Int>(dtw->get_matrix_width()));

    REQUIRE_NOTHROW(proxy.synchronize_with_master());

    // At this point, the proxy should have the right values.
    auto const dt_value = El::To<DataType>(value);
    CHECK(count_differing_values(dt_value, get_local_values(proxy)) == 0);
  }

  // This SECTION uses `double` since we don't independently test the
  // <double,double> combination.
  SECTION("Copy-from-other-type construction")
  {
    // Test "copy from other type" construction.
    lbann::WeightsProxy<double> proxy_copy(proxy);
    REQUIRE(!proxy_copy.empty());
    CHECK(&proxy.master_weights() == &proxy_copy.master_weights());
    CHECK(proxy_copy.values().Height() == proxy.values().Height());
    CHECK(proxy_copy.values().Width() == proxy.values().Width());
  }

  SECTION("WeightsProxy move construction")
  {
    lbann::WeightsProxy<DataType> proxy_move(std::move(proxy));
    REQUIRE(proxy.empty());
    REQUIRE(!proxy_move.empty());
    CHECK(&proxy_move.master_weights() == dtw.get());
    CHECK(proxy_move.values().Height() == dtw->get_values().Height());
    CHECK(proxy_move.values().Width() == dtw->get_values().Width());
  }

  SECTION("WeightsProxy swap operation")
  {
    lbann::WeightsProxy<DataType> proxy_other;
    REQUIRE(proxy_other.empty());

    // Do the swap (no point REQUIRE_NOTHROW since the swap operation
    // is noexcept -- it will terminate if an exception is encountered).
    std::swap(proxy, proxy_other);
    REQUIRE(proxy.empty());
    REQUIRE(!proxy_other.empty());
    CHECK(&proxy_other.master_weights() == dtw.get());
    CHECK(proxy_other.values().Height() == dtw->get_values().Height());
    CHECK(proxy_other.values().Width() == dtw->get_values().Width());
  }

  SECTION("WeightsProxy copy assignment")
  {
    lbann::WeightsProxy<DataType> proxy_other;
    REQUIRE(proxy_other.empty());

    proxy_other = proxy;

    REQUIRE(!proxy.empty());
    REQUIRE(!proxy_other.empty());

    CHECK(&proxy_other.master_weights() == &proxy.master_weights());
    CHECK(proxy_other.values().Height() == proxy.values().Height());
    CHECK(proxy_other.values().Width() == proxy.values().Width());
  }

  SECTION("WeightsProxy move assignment")
  {
    lbann::WeightsProxy<DataType> proxy_other;
    REQUIRE(proxy_other.empty());

    proxy_other = std::move(proxy);
    REQUIRE(proxy.empty());
    REQUIRE(!proxy_other.empty());
    CHECK(&proxy_other.master_weights() == dtw.get());
    CHECK(proxy_other.values().Height() == dtw->get_values().Height());
    CHECK(proxy_other.values().Width() == dtw->get_values().Width());
  }

  // Verify the default-construction path to sanity.
  SECTION("Default-constructed proxy.")
  {
    lbann::WeightsProxy<DataType> proxy_default;
    REQUIRE(proxy_default.empty());
    REQUIRE_NOTHROW(proxy_default.setup(dtw));
    REQUIRE(!proxy_default.empty());
    CHECK(&proxy_default.master_weights() == dtw.get());

    // At this point, the proxy_default should have the right size.
    CHECK(proxy_default.values().Height() ==
          El::To<El::Int>(dtw->get_matrix_height()));
    CHECK(proxy_default.values().Width() ==
          El::To<El::Int>(dtw->get_matrix_width()));
  }
}
