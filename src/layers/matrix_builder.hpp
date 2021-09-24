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

#ifndef NON_PUBLIC_LBANN_SRC_LAYERS_MATRIX_BUILDER_INCLUDED
#define NON_PUBLIC_LBANN_SRC_LAYERS_MATRIX_BUILDER_INCLUDED

#include "lbann/base.hpp"
#include "lbann/utils/memory.hpp"

#include <memory>
#include <type_traits>

namespace lbann {
namespace details {
namespace meta {
// Quick, forced if-then-else
template <bool B, typename T, typename F>
using IfThenElse = typename std::conditional<B, T, F>::type;
}// namespace meta

template <typename T>
class MatrixBuilder
{
public:
  using size_type = El::Int;
  using data_type = T;
  using matrix_type = El::AbstractDistMatrix<T>;
  using matrix_ptr_type = std::unique_ptr<matrix_type>;

public:
  virtual ~MatrixBuilder() = default;
  virtual matrix_ptr_type MakeEmpty(El::Grid const& g, El::Int root) const = 0;
  virtual matrix_ptr_type MakeWithSize(
    El::Grid const& g, El::Int root,
    size_type height, size_type width) const = 0;

};// class MatrixBuilder

// Uses memory mode = 1 for CPU (pinned memory) and for GPU (CUB memory).
template <typename T, data_layout L, El::Device D>
class DefaultMemoryMatrixBuilder : public MatrixBuilder<T>
{
  using base_type = MatrixBuilder<T>;
  using concrete_matrix_type =
    meta::IfThenElse<L == data_layout::DATA_PARALLEL,
                     El::DistMatrix<T, El::STAR, El::VC, El::ELEMENT, D>,
                     El::DistMatrix<T, El::MC  , El::MR, El::ELEMENT, D>>;

#if defined(HYDROGEN_HAVE_GPU) && defined(HYDROGEN_HAVE_CUB)
  // Pinned host memory; memory-pooled device memory
  static constexpr unsigned memory_mode_ = 1U;
#elif defined(HYDROGEN_HAVE_GPU)
  // Pinned host memory; directly-allocated device memory
  static constexpr unsigned memory_mode_ = (D == El::Device::CPU ? 1U : 0U);
#else
  // Default memory
  static constexpr unsigned memory_mode_ =
    El::DefaultMemoryMode<El::Device::CPU>();
#endif // defined(HYDROGEN_HAVE_GPU) && defined(HYDROGEN_HAVE_CUB)

public:
  using size_type = typename base_type::size_type;
  using matrix_ptr_type = typename base_type::matrix_ptr_type;

public:
  matrix_ptr_type MakeEmpty(El::Grid const& g, El::Int root) const final
  {
    auto ret = make_unique<concrete_matrix_type>(g, root);
    ret->Matrix().SetMemoryMode(memory_mode_);
    return ret;
  }

  matrix_ptr_type MakeWithSize(El::Grid const& g, El::Int root,
                               size_type height, size_type width) const final
  {
    auto ret = this->MakeEmpty(g, root);
    ret->Resize(height, width);
    return ret;
  }

};// class DefaultMemoryMatrixBuilder

}// namespace details
}// namespace lbann
#endif // NON_PUBLIC_LBANN_SRC_LAYERS_MATRIX_BUILDER_INCLUDED
