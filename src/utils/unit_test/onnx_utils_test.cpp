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
#include "TestHelpers.hpp"

#include <h2/meta/core/IfThenElse.hpp>
#include <lbann/utils/onnx_utils.hpp>

float get_value(onnx::TensorProto const& p, size_t index, lbann::TypeTag<float>)
{
  return p.float_data(index);
}
double
get_value(onnx::TensorProto const& p, size_t index, lbann::TypeTag<double>)
{
  return p.double_data(index);
}
template <typename T>
El::Complex<T> get_value(onnx::TensorProto const& p,
                         size_t index,
                         lbann::TypeTag<El::Complex<T>>)
{
  return {get_value(p, 2 * index, lbann::TypeTag<T>{}),
          get_value(p, 2 * index + 1, lbann::TypeTag<T>{})};
}

template <typename T>
bool check_value(onnx::TensorProto const& p, size_t index, T const& true_value)
{
  auto const proto_value = get_value(p, index, lbann::TypeTag<T>{});
  return (true_value == proto_value);
}

#define CHECK_SERIALIZATION(true_vals, proto_msg)                              \
  size_t n_weights = 0UL;                                                      \
  size_t const n_dims = proto.dims_size();                                     \
  for (size_t ii = 0; ii < n_dims; ++ii)                                       \
    n_weights += proto.dims(ii);                                               \
                                                                               \
  auto const& weights = true_vals.LockedMatrix();                              \
  auto const mat_width = weights.Width();                                      \
  size_t idx = 0UL;                                                            \
                                                                               \
  do {                                                                         \
    auto const row = idx / mat_width;                                          \
    auto const col = idx % mat_width;                                          \
                                                                               \
    INFO("(row, col) == (" << row << ", " << col << "); index = " << idx       \
                           << " / " << n_weights);                             \
    CHECK(get_value(proto_msg, idx, lbann::TypeTag<DType>{}) ==                \
          weights.CRef(row, col));                                             \
                                                                               \
    ++idx;                                                                     \
  } while (idx < n_weights)

template <typename T, El::Device D>
struct TypeDevicePack
{
  using type = T;
  static constexpr auto device = D;
}; // struct TypeDevicePack

using AllTypes = El::TypeList<
#ifdef LBANN_HAS_HALF
  TypeDevicePack<cpu_half_type, El::Device::CPU>,
#ifdef LBANN_HAS_GPU_FP16
  TypeDevicePack<gpu_half_type, El::Device::GPU>,
#endif // LBANN_HAS_GPU_FP16
#endif // LBANN_HAS_HALF
#ifdef LBANN_HAS_GPU
  TypeDevicePack<float, El::Device::GPU>,
  TypeDevicePack<double, El::Device::GPU>,
  TypeDevicePack<El::Complex<float>, El::Device::GPU>,
  TypeDevicePack<El::Complex<double>, El::Device::GPU>,
#endif // LBANN_HAS_GPU
  TypeDevicePack<float, El::Device::CPU>,
  TypeDevicePack<double, El::Device::CPU>,
  TypeDevicePack<El::Complex<float>, El::Device::CPU>,
  TypeDevicePack<El::Complex<double>, El::Device::CPU>>;

template <typename T>
struct DataTypeForSavedValues
{
  using type = T;
};

#ifdef LBANN_HAS_HALF
template <>
struct DataTypeForSavedValues<cpu_half_type>
{
  using type = float;
};
#ifdef LBANN_HAS_GPU_FP16
template <>
struct DataTypeForSavedValues<gpu_half_type>
{
  using type = float;
};
#endif // LBANN_HAS_GPU_FP16
#endif // LBANN_HAS_HALF

// Ignoring all the above crap, this is just trying to test {fp16,
// float, double, Complex<float>, Complex<double>} x {CPU, GPU}. The
// test case will create "weights" tensors, serialize them to
// TensorProto messages, and then validate the values stored in the
// message.

TEMPLATE_LIST_TEST_CASE("Serializing a DistMatrix to ONNX",
                        "[onnx][utils]",
                        AllTypes)
{
  using namespace El;
  using DType = typename TestType::type;
  using TrueDType = typename DataTypeForSavedValues<DType>::type;
  constexpr auto Device = TestType::device;

  auto& comm = unit_test::utilities::current_world_comm();
  auto const& grid = comm.get_trainer_grid();

  onnx::TensorProto proto;

  // Master copy of the weights, for value checking.
  DistMatrix<TrueDType, STAR, STAR, ELEMENT, Device::CPU> m_true(grid, 0);

  // Every process will have a 2x2 local matrix.
  SECTION("Fully connected layers.")
  {
    auto const height = 2 * grid.Height();
    auto const width = 3 * grid.Width();
    std::vector<size_t> height_dims = {static_cast<size_t>(height)};
    std::vector<size_t> width_dims = {static_cast<size_t>(width)};
    SECTION("Model-parallel")
    {
      DistMatrix<DType, MC, MR, ELEMENT, Device> m(grid, 0);
      Uniform(m, height, width);
      El::Copy(m, m_true);
      REQUIRE_NOTHROW(
        lbann::serialize_to_onnx(m, height_dims, width_dims, proto));

      CHECK_SERIALIZATION(m_true, proto);
    }

    SECTION("Data-parallel")
    {
      DistMatrix<DType, STAR, STAR, ELEMENT, Device> m(grid, 0);
      Uniform(m, height, width);
      El::Copy(m, m_true);
      REQUIRE_NOTHROW(
        lbann::serialize_to_onnx(m, height_dims, width_dims, proto));

      CHECK_SERIALIZATION(m_true, proto);
    }
  }

  SECTION("Convolutional layers.")
  {
    SECTION("Kernel")
    {
      std::vector<size_t> kernel_dims = {6UL, 6UL, 5UL, 5UL};
      std::vector<size_t> width_dims;
      DistMatrix<DType, STAR, STAR, ELEMENT, Device> k(grid, 0);
      Uniform(k, lbann::get_linear_size(kernel_dims), 1);
      El::Copy(k, m_true);
      REQUIRE_NOTHROW(
        lbann::serialize_to_onnx(k, kernel_dims, width_dims, proto));

      CHECK_SERIALIZATION(m_true, proto);
    }
    SECTION("Bias")
    {
      std::vector<size_t> bias_dims = {6UL};
      std::vector<size_t> width_dims;
      DistMatrix<DType, STAR, STAR, ELEMENT, Device> b(grid, 0);
      Uniform(b, lbann::get_linear_size(bias_dims), 1);
      El::Copy(b, m_true);
      REQUIRE_NOTHROW(
        lbann::serialize_to_onnx(b, bias_dims, width_dims, proto));

      CHECK_SERIALIZATION(m_true, proto);
    }
  }
}
