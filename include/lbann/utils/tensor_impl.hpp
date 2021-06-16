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
    El::Copy(src, tgt);
  }
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
