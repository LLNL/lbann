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

#ifndef LBANN_UTILS_DNN_LIB_ONEDNN_HPP
#define LBANN_UTILS_DNN_LIB_ONEDNN_HPP

#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/dnn_enums.hpp"
#include "lbann/utils/exception.hpp"

#include <h2/meta/Core.hpp>
#include <h2/meta/TypeList.hpp>

#ifdef LBANN_HAS_ONEDNN

#include <dnnl.hpp>

namespace lbann {
namespace onednn {
namespace details {

/** @class TypeMapT
 *  @brief Map C++ types to OneDNN enum values.
 */
template <typename T>
struct TypeMapT;

/** @class IsSupportedTypeT
 *  @brief Predicate indicating if a type is supported by oneDNN.
 */
template <typename T>
struct IsSupportedTypeT : std::false_type
{
};

#define ADD_ONEDNN_TYPE_MAP(CPPTYPE, EVAL)                                     \
  template <>                                                                  \
  struct TypeMapT<CPPTYPE>                                                     \
    : std::integral_constant<dnnl::memory::data_type,                          \
                             dnnl::memory::data_type::EVAL>                    \
  {                                                                            \
  };                                                                           \
  template <>                                                                  \
  struct IsSupportedTypeT<CPPTYPE> : std::true_type                            \
  {                                                                            \
  }

// Basic types
ADD_ONEDNN_TYPE_MAP(float, f32);
ADD_ONEDNN_TYPE_MAP(int32_t, s32);
ADD_ONEDNN_TYPE_MAP(int8_t, s8);
ADD_ONEDNN_TYPE_MAP(uint8_t, u8);

// 16-bit floating point
// TODO: bfloat16 types (not yet supported in Hydrogen)
#if defined LBANN_HAS_HALF
ADD_ONEDNN_TYPE_MAP(cpu_fp16, f16);
#endif
#if defined LBANN_HAS_GPU_FP16
ADD_ONEDNN_TYPE_MAP(fp16, f16);
#endif

} // namespace details

template <typename T>
using TypeMap = typename details::TypeMapT<T>;

template <typename T>
inline constexpr auto DataTypeValue = TypeMap<T>::value;

template <typename T>
inline constexpr bool IsSupportedType = details::IsSupportedTypeT<T>::value;

template <typename T, ::h2::meta::EnableWhen<IsSupportedType<T>, int> = 0>
inline constexpr dnnl::memory::data_type get_data_type()
{
  return DataTypeValue<T>;
}

template <typename T, ::h2::meta::EnableUnless<IsSupportedType<T>, int> = 0>
inline dnnl::memory::data_type get_data_type()
{
  LBANN_ERROR("Type \"",
              El::TypeName<T>(),
              "\" is not supported "
              "by the oneDNN runtime.");
}

template <El::Device D>
dnnl::engine& get_device_engine();

template <El::Device D>
dnnl::stream get_stream(dnnl::engine const& e, El::SyncInfo<D> const&);

} // namespace onednn

template <El::Device D>
struct onednn_backend
{
  static constexpr auto device = D;

  template <typename T>
  static auto data_type()
  {
    return onednn::get_data_type<T>();
  }

  class TensorDescriptor
  {
  public:
    /** @brief The device type associated with this descriptor. */
    static constexpr auto device = D;

    /** @brief The backend type to which this type belongs. */
    using backend_type = onednn_backend<D>;

    /** @brief The DNNL handle being managed. */
    using dnnTensorDescriptor_t = dnnl::memory;

    /** @brief The data type enumerator type for oneDNN. */
    using dnnDataType_t = dnnl::memory::data_type;

    /** @brief The tensor format type. */
    using dnnTensorFormat_t = dnnl::memory::format_tag;

    /** @brief The internal, device-independent memory descriptor type. */
    using internal_descriptor_type = dnnl::memory::desc;

  public:
    /** @name Lifecycle management */
    ///@{
    /** @brief Construct an empty descriptor. */
    TensorDescriptor() = default;

    /** @brief Construct from existing memory object. */
    explicit TensorDescriptor(dnnTensorDescriptor_t desc)
      : desc_{std::move(desc)}
    {}

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
    void swap(TensorDescriptor& other) { std::swap(desc_, other.desc_); }

    /** @brief Take ownership of DNN library object */
    void reset(dnnTensorDescriptor_t desc = dnnTensorDescriptor_t{})
    {
      desc_ = dnnl::memory{std::move(desc)};
    }

    /** @brief Return DNN library object and release ownership */
    dnnTensorDescriptor_t release() noexcept
    {
      dnnTensorDescriptor_t tmp = desc_;
      desc_ = dnnl::memory{};
      return tmp;
    }

    /** @brief Return DNN library object without releasing ownership */
    dnnTensorDescriptor_t get() const noexcept { return desc_; }

    /** @brief Return DNN library object without releasing ownership */
    operator dnnTensorDescriptor_t() const noexcept { return desc_; }

    /** @brief Create DNN library object
     *
     *  Nothing is required in this case.
     */
    void create() noexcept {}

    void set(dnnDataType_t data_type,
             dnnl::memory::dims dims,
             dnnl::memory::dims strides = {})
    {
      if (strides.empty())
        strides = get_packed_strides(dims);

      auto md = dnnl::memory::desc(dims, data_type, strides);
      desc_ =
        dnnl::memory(md, onednn::get_device_engine<D>(), DNNL_MEMORY_NONE);
    }

    /** @brief Configure DNN library object
     *
     *  Creates DNN library object if needed.
     */
    template <typename DimT>
    void set(dnnDataType_t data_type,
             std::vector<DimT> const& dims_in,
             std::vector<DimT> const& strides_in = {})
    {
      dnnl::memory::dims dims{cbegin(dims_in), cend(dims_in)};
      dnnl::memory::dims strides{cbegin(strides_in), cend(strides_in)};
      this->set(data_type, dims, strides);
    }

    /** @brief Configure DNN library object
     *
     *  Creates DNN library object if needed.
     */
    template <typename... IntTs>
    void set(dnnDataType_t data_type, IntTs... dims)
    {
      set(data_type, {static_cast<dnnl::memory::dim>(dims)...});
    }
    // This function is required for API compatibility.
    void set(dnnDataType_t data_type,
             dnnTensorFormat_t /*format*/,
             const std::vector<int>& dims)
    {
      this->set(data_type, dims);
    }

  private:
    /** @brief The DNNL memory handle.
     *
     *  @note This handle is tied to a specific device. This is
     *        implicitly true for MIOpen and cuDNN handles, which only
     *        exist for GPU devices. It is only for oneDNN that we need
     *        to make this explicit distinction.
     */
    dnnl::memory desc_;

  }; // class TensorDescriptor

  template <typename DataT, typename ScalarT>
  static void softmax_forward(ScalarT const& alpha_in,
                              TensorDescriptor const& xDesc,
                              El::Matrix<DataT, D> const& x,
                              ScalarT const& beta_in,
                              TensorDescriptor const& yDesc,
                              El::Matrix<DataT, D>& y,
                              El::SyncInfo<D> const& si,
                              softmax_mode mode,
                              softmax_alg alg = softmax_alg::ACCURATE);

  template <typename DataT, typename ScalarT>
  static void logsoftmax_forward(ScalarT const& alpha_in,
                                 TensorDescriptor const& xDesc,
                                 El::Matrix<DataT, D> const& x,
                                 ScalarT const& beta_in,
                                 TensorDescriptor const& yDesc,
                                 El::Matrix<DataT, D>& y,
                                 El::SyncInfo<D> const& si,
                                 softmax_mode mode)
  {
    LBANN_ERROR("Not yet implemented.");
  }

  template <typename DataT, typename ScalarT>
  static void softmax_backward(ScalarT const& alpha_in,
                               TensorDescriptor const& yDesc,
                               El::Matrix<DataT, D> const& y,
                               TensorDescriptor const& dyDesc,
                               El::Matrix<DataT, D> const& dy,
                               ScalarT const& beta_in,
                               TensorDescriptor const& dxDesc,
                               El::Matrix<DataT, D>& dx,
                               El::SyncInfo<D> const& si,
                               softmax_mode mode,
                               softmax_alg alg = softmax_alg::ACCURATE);

  template <typename DataT, typename ScalarT>
  static void logsoftmax_backward(ScalarT const& alpha_in,
                                  TensorDescriptor const& yDesc,
                                  El::Matrix<DataT, D> const& y,
                                  TensorDescriptor const& dyDesc,
                                  El::Matrix<DataT, D> const& dy,
                                  ScalarT const& beta_in,
                                  TensorDescriptor const& dxDesc,
                                  El::Matrix<DataT, D>& dx,
                                  El::SyncInfo<D> const& si,
                                  softmax_mode mode,
                                  softmax_alg alg = softmax_alg::ACCURATE)
  {
    LBANN_ERROR("Not yet implemented.");
  }

}; // struct onednn_backend

} // namespace lbann
#endif // LBANN_HAS_ONEDNN
#endif // LBANN_UTILS_DNN_LIB_ONEDNN_HPP
