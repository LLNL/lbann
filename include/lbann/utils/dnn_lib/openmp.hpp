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

#ifndef LBANN_UTILS_DNN_LIB_OPENMP_HPP
#define LBANN_UTILS_DNN_LIB_OPENMP_HPP

#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/dnn_enums.hpp"
#include "lbann/utils/exception.hpp"

#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>

namespace lbann {

/** @class openmp_backend
 *  @brief DNN library backend for hand-rolled, OMP-based implementations.
 *  @details This backend only supports CPUs right now.
 */
struct openmp_backend
{
  static constexpr auto device = El::Device::CPU;

  /** @struct unnecessary
   *  Type indicating an unnecessary, i.e., placeholder, type.
   */
  struct unnecessary
  {
  };

  template <typename T>
  static auto data_type()
  {
    return unnecessary{};
  }

  class TensorDescriptor
  {
  public:
    /** @brief The backend type to which this type belongs. */
    using backend_type = openmp_backend;

    /** @brief The device type associated with this descriptor. */
    static constexpr auto device = backend_type::device;

    /** @brief The DNNL handle being managed. */
    using dnnTensorDescriptor_t = unnecessary;

    /** @brief The data type enumerator type for oneDNN. */
    using dnnDataType_t = unnecessary;

    /** @brief The data type enumerator type for oneDNN. */
    using dnnTensorFormat_t = unnecessary;

  public:
    /** @name Lifecycle management */
    ///@{
    /** @brief Construct an empty descriptor. */
    TensorDescriptor() = default;
    /** @brief Destructor. */
    ~TensorDescriptor() noexcept = default;
    /** @brief Copy constructor. */
    TensorDescriptor(TensorDescriptor const&) = default;
    /** @brief Move constructor. */
    TensorDescriptor(TensorDescriptor&&) = default;
    /** @brief Copy assignment. */
    TensorDescriptor& operator=(TensorDescriptor const&) = default;
    /** @brief Move assignment. */
    TensorDescriptor& operator=(TensorDescriptor&&) = default;

    /** @brief Exchange contents with another descriptor. */
    void swap(TensorDescriptor& other) {}
    /** @brief Take ownership of DNN library object */
    void reset(dnnTensorDescriptor_t) {}

    /** @brief Return DNN library object and release ownership */
    dnnTensorDescriptor_t release() noexcept { return {}; }

    /** @brief Return DNN library object without releasing ownership */
    dnnTensorDescriptor_t get() const noexcept { return {}; }

    /** @brief Return DNN library object without releasing ownership */
    operator dnnTensorDescriptor_t() const noexcept { return this->get(); }

    /** @brief Create DNN library object
     *
     *  Nothing is required in this case.
     */
    void create() noexcept {}

    /** @brief Configure DNN library object
     *
     *  Creates DNN library object if needed.
     */
    template <typename DimT>
    void
    set(dnnDataType_t, std::vector<DimT> const&, std::vector<DimT> const& = {})
    {}

    /** @brief Configure DNN library object
     *
     *  Creates DNN library object if needed.
     */
    template <typename... IntTs>
    void set(dnnDataType_t, IntTs...)
    {}
#if !(defined LBANN_HAS_CUDNN)
    // This function is required for API compatibility.
    void set(dnnDataType_t /*data_type*/,
             dnnTensorFormat_t /*format*/,
             std::vector<int> const& /*dims*/)
    {}
#endif // !LBANN_HAS_CUDNN

  }; // class TensorDescriptor

  template <typename DataT, typename ScalarT>
  static void softmax_forward(ScalarT const& alpha_in,
                              TensorDescriptor const& xDesc,
                              El::Matrix<DataT, device> const& x,
                              ScalarT const& beta_in,
                              TensorDescriptor const& yDesc,
                              El::Matrix<DataT, device>& y,
                              El::SyncInfo<device> const& si,
                              softmax_mode mode,
                              softmax_alg alg = softmax_alg::ACCURATE);

  template <typename DataT, typename ScalarT>
  static void logsoftmax_forward(ScalarT const& alpha_in,
                                 TensorDescriptor const& xDesc,
                                 El::Matrix<DataT, device> const& x,
                                 ScalarT const& beta_in,
                                 TensorDescriptor const& yDesc,
                                 El::Matrix<DataT, device>& y,
                                 El::SyncInfo<device> const& si,
                                 softmax_mode mode)
  {
    LBANN_ERROR("Not yet implemented.");
  }

  template <typename DataT, typename ScalarT>
  static void softmax_backward(ScalarT const& alpha_in,
                               TensorDescriptor const& yDesc,
                               El::Matrix<DataT, device> const& y,
                               TensorDescriptor const& dyDesc,
                               El::Matrix<DataT, device> const& dy,
                               ScalarT const& beta_in,
                               TensorDescriptor const& dxDesc,
                               El::Matrix<DataT, device>& dx,
                               El::SyncInfo<device> const& si,
                               softmax_mode mode,
                               softmax_alg alg = softmax_alg::ACCURATE);

  template <typename DataT, typename ScalarT>
  static void logsoftmax_backward(ScalarT const& alpha_in,
                                  TensorDescriptor const& yDesc,
                                  El::Matrix<DataT, device> const& y,
                                  TensorDescriptor const& dyDesc,
                                  El::Matrix<DataT, device> const& dy,
                                  ScalarT const& beta_in,
                                  TensorDescriptor const& dxDesc,
                                  El::Matrix<DataT, device>& dx,
                                  El::SyncInfo<device> const& si,
                                  softmax_mode mode,
                                  softmax_alg alg = softmax_alg::ACCURATE)
  {
    LBANN_ERROR("Not yet implemented.");
  }

}; // struct openmp_backend

} // namespace lbann
#endif // LBANN_UTILS_DNN_LIB_OPENMP_HPP
