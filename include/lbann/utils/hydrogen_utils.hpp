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

namespace lbann {

#include <El.hpp>
#include <lbann/utils/memory.hpp>

template <typename TensorDataType, typename EvalDataType>
struct ViewIfPossibleOrCopy {
  static std::unique_ptr<El::AbstractMatrix<EvalDataType>> get(El::AbstractMatrix<TensorDataType> const& x)
  {
    switch (x.GetDevice()) {
    case El::Device::CPU:
      return get(static_cast<El::Matrix<TensorDataType, El::Device::CPU> const&>(x));
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      return get(static_cast<El::Matrix<TensorDataType, El::Device::GPU> const&>(x));
#endif
    default: return nullptr;
    }
  }
  template <El::Device D>
  static std::unique_ptr<El::Matrix<EvalDataType, D>> get(El::Matrix<TensorDataType, D> const& x)
  {
    auto ret = std::make_unique<El::Matrix<EvalDataType, D>>();
    El::Copy(x, *ret);
    return ret;
  }
};

// Specialize for same data type -- make a view instead.
template <typename DataType>
struct ViewIfPossibleOrCopy<DataType,DataType> {
  static std::unique_ptr<El::AbstractMatrix<DataType>> get(El::AbstractMatrix<DataType> const& x)
  {
    switch (x.GetDevice()) {
    case El::Device::CPU:
      return get(static_cast<El::Matrix<DataType, El::Device::CPU> const&>(x));
#ifdef LBANN_HAS_GPU
    case El::Device::GPU:
      return get(static_cast<El::Matrix<DataType, El::Device::GPU> const&>(x));
#endif
    default: return nullptr;
    }
  }
  template <El::Device D>
  static std::unique_ptr<El::Matrix<DataType, D>> get(El::Matrix<DataType, D> const& x)
  {
    auto ret = std::make_unique<El::Matrix<DataType, D>>();
    El::LockedView(*ret, x);
    return ret;
  }
};

}
