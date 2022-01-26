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

#ifndef LBANN_UTILS_TENSOR_IMPL_HPP
#define LBANN_UTILS_TENSOR_IMPL_HPP

#include "lbann/utils/tensor.hpp"
#include "lbann_config.hpp"

namespace lbann {

template <typename TDT>
void do_tensor_copy(const BaseDistMat& src,
                    El::AbstractDistMatrix<TDT>& tgt) {
  bool copy_async = false;
#if defined(LBANN_HAS_GPU) && defined(ASYNC_INPUT_MEMORY_TRANSFER)
  auto src_dist_data = src.DistData();
  auto tgt_dist_data = tgt.DistData();
  // Asynchronously copy CPU data to GPU data if they are otherwise aligned
  if ((src.dist_data.device == El::Device::CPU)
      && (tgt_dist_data.device == El::Device::GPU)) {
    src_dist_data.device = El::Device::GPU;
    copy_async = (src_dist_data == tgt_dist_data);
  }
#endif // defined(LBANN_HAS_GPU) && defined(ASYNC_INPUT_MEMORY_TRANSFER)
  if (copy_async) {
    El::CopyAsync(src, tgt);
  }
  else {
    if (src.DistData().grid == tgt.DistData().grid) {
      El::Copy(src, tgt);
    }
    else {
      utils::details::do_tensor_copy_between_grids(src, tgt);
    }
  }
}

template <typename TDT>
void utils::details::do_tensor_copy_between_grids(
  const BaseDistMat& src,
  El::AbstractDistMatrix<TDT>& tgt) {

  // Determine matrix class and forward to template function
  /// @todo Do this more systematically and support all matrix classes
  const auto& tgt_dist = tgt.DistData();
  bool did_copy = false;
#undef LBANN_TEMPLATE_INSTANTIATION
#define LBANN_TEMPLATE_INSTANTIATION(ColDist, RowDist, Device)          \
  do {                                                                  \
    if (tgt_dist.colDist == ColDist                                     \
        && tgt_dist.rowDist == RowDist                                  \
        && tgt_dist.device == Device) {                                 \
      using TgtMatrixType                                               \
        = El::DistMatrix<TDT, ColDist, RowDist, El::ELEMENT, Device>;   \
      utils::details::do_tensor_copy_between_grids(                     \
        src,                                                            \
        dynamic_cast<TgtMatrixType&>(tgt));                             \
      did_copy = true;                                                  \
    }                                                                   \
  } while (false)
  LBANN_TEMPLATE_INSTANTIATION(El::STAR, El::VC,   El::Device::CPU);
  LBANN_TEMPLATE_INSTANTIATION(El::MC,   El::MR,   El::Device::CPU);
  LBANN_TEMPLATE_INSTANTIATION(El::STAR, El::STAR, El::Device::CPU);
#ifdef LBANN_HAS_GPU
  LBANN_TEMPLATE_INSTANTIATION(El::STAR, El::VC,   El::Device::GPU);
  LBANN_TEMPLATE_INSTANTIATION(El::MC,   El::MR,   El::Device::GPU);
  LBANN_TEMPLATE_INSTANTIATION(El::STAR, El::STAR, El::Device::GPU);
#endif // LBANN_HAS_GPU
#undef LBANN_TEMPLATE_INSTANTIATION

  // Check if copy succeeded
  if (!did_copy) {
    const auto& src_dist = src.DistData();
    LBANN_ERROR(
      "Failed to copy between two tensors on different grids ",
      "(src: colDist=",int(src_dist.colDist),", ",
      "rowDist=",int(src_dist.rowDist),", ",
      "device=",int(src_dist.device),"; "
      "tgt: colDist=",int(tgt_dist.colDist),", ",
      "rowDist=",int(tgt_dist.rowDist),", ",
      "device=",int(tgt_dist.device),")");
  }

}

template <typename TDT, El::Dist ColDist, El::Dist RowDist, El::DistWrap Wrap, El::Device Device>
void utils::details::do_tensor_copy_between_grids(
  const BaseDistMat& src,
  El::DistMatrix<TDT,ColDist,RowDist,Wrap,Device>& tgt) {

  // Make sure matrix layouts are identical
  using TgtMatrixType = El::DistMatrix<TDT,ColDist,RowDist,Wrap,Device>;
  auto src_dist = src.DistData();
  TgtMatrixType temp(*src_dist.grid, src_dist.root);
  if (temp.DistData() == src_dist) {
    El::LockedView(temp, dynamic_cast<const TgtMatrixType&>(src));
  }
  else {
    temp.Resize(src.Height(), src.Width());
    if (temp.Participating()) {
      El::Copy(src, temp);
    }
  }

  // Translate matrix between grids
  tgt.Resize(src.Height(), src.Width());
  El::copy::Translate(temp, tgt);

}

template <typename TDT>
void view_or_copy_tensor(const BaseDistMat& src,
                         El::AbstractDistMatrix<TDT>& tgt) {

  if (src.DistData() == tgt.DistData()) {
    El::LockedView(tgt,
                   dynamic_cast<const El::AbstractDistMatrix<TDT>&>(src));
  }
  else {
    do_tensor_copy(src, tgt);
  }
}

}

#endif // LBANN_UTILS_TENSOR_IMPL_HPP
