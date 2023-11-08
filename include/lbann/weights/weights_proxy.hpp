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
#ifndef LBANN_WEIGHTS_WEIGHTS_PROXY_HPP_INCLUDED
#define LBANN_WEIGHTS_WEIGHTS_PROXY_HPP_INCLUDED

#include "lbann_config.hpp"

#include "lbann/base.hpp"
#include "lbann/weights/data_type_weights.hpp"
#include "lbann/weights/weights.hpp"

#if defined LBANN_DEBUG
#include "lbann/utils/exception.hpp"
#define LBANN_DEBUG_ASSERT_POINTER(ptr)                                        \
  do {                                                                         \
    if (!ptr)                                                                  \
      LBANN_ERROR("Pointer \"" #ptr "\" is null.");                            \
  } while (0)
#define LBANN_IN_DEBUG_MODE true
#else
#define LBANN_DEBUG_ASSERT_POINTER(ptr)
#define LBANN_IN_DEBUG_MODE false
#endif

namespace lbann {

/** @class WeightsProxy
 *  @brief Proxy a weights object as a different data type.
 *
 *  This class is intended to be an implementation detail of the
 *  layers' interactions with weights objects. Thus, the
 *  implementation employs a programming-by-contract approach in an
 *  effort to avoid, e.g., safe dereferences to internal pointer
 *  members.
 *
 *  The class contract is as follows:
 *
 *    - It is invalid to attempt to access the values() or
 *      master_weights() of a WeightsProxy object for which empty()
 *      returns @c true.
 *    - It is invalid to derive meaning from the values() of a
 *      WeightsProxy object after construction or after modifying the
 *      master weights externally until synchronize_with_master() is
 *      called on that WeightsProxy object
 *    - If the memory for the values matrix of a master weights object
 *      watched by a WeightsProxy object is replaced for any reason,
 *      the user is responsible for calling setup() on that object
 *      again. The local values are subsequently considered invalid
 *      until synchronize_with_master() is called on that object.
 *
 *  @tparam TensorDataType The type to which the weights are proxied.
 */
template <typename TensorDataType>
class WeightsProxy
{
  /** @brief The type of weights values. */
  using ValuesType = El::AbstractDistMatrix<TensorDataType>;
  /** @brief Convenience typedef for poitners to weights values. */
  using ValuesPtrType = std::unique_ptr<ValuesType>;

public:
  /** @name Constructors */
  ///@{

  /** @brief Construct an empty proxy. */
  WeightsProxy() = default;

  /** @brief Construct a proxy given the master object.
   *
   *  @param w Master weights object, which must have a valid storage
   *           matrix initialized internally.
   */
  WeightsProxy(ViewingWeightsPtr const& w)
  {
    if (!w.expired()) {
      this->setup(w);
    }
  }

  /** @brief Copy a WeightsProxy object.
   *
   *  Creates a new proxy to the same weights object.
   */
  WeightsProxy(WeightsProxy const& other)
  {
    if (!other.master_weights_.expired()) {
      this->setup(other.master_weights_);
    }
  }

  /** @brief Copy a WeightsProxy object.
   *
   *  Creates a new proxy to the same weights object.
   *
   *  @tparam T (Deduced) The type of the input weights object's
   *                      values.
   */
  template <typename T>
  WeightsProxy(WeightsProxy<T> const& other) : WeightsProxy()
  {
    auto ptr = other.master_weights_pointer();
    if (!ptr.expired()) {
      this->setup(ptr);
    }
  }

  /** @brief Move a WeightsProxy object.
   *
   *  Unlike copy construction, move construction is only supported
   *  for WeightsProxy objects of the same static type.
   */
  WeightsProxy(WeightsProxy&& other) noexcept
    : master_weights_{std::move(other.master_weights_)},
      values_{std::move(other.values_)}
  {
    other.clear();
  }

  /** @brief Destructor */
  ~WeightsProxy() noexcept { this->clear(); }

  ///@}

  /** @name Assignment operators */
  ///@{

  /** @brief Copy assignment operator. */
  WeightsProxy& operator=(WeightsProxy const& other)
  {
    WeightsProxy<TensorDataType>(other).swap(*this);
    return *this;
  }

  /** @brief Assignment from WeightsProxy object of a different type.
   *
   *  After assignment, @c this and @c other both proxy the weights
   *  proxied by @c other.
   *
   *  @tparam T (Deduced) The type of the input weights object's
   *                      values.
   */
  template <typename T>
  WeightsProxy& operator=(WeightsProxy<T> const& other)
  {
    WeightsProxy<TensorDataType>(other).swap(*this);
    return *this;
  }

  /** @brief Move assignment from another proxy object. */
  WeightsProxy& operator=(WeightsProxy&& other) noexcept
  {
    // "Move-and-swap" idiom
    WeightsProxy<TensorDataType>(std::move(other)).swap(*this);
    return *this;
  }

  ///@}
  /** @name Master object management and synchronization. */
  ///@{

  /** @brief Restore the default state of the proxy.
   *
   *  After this function is called, the object will be empty().
   */
  void clear() noexcept
  {
    master_weights_.reset();
    values_.reset();
  }

  /** @brief Provide setup function for delayed construction.
   *
   *  This overwrites any existing data.
   *
   *  @param w The weights object to be proxied.
   */
  void setup(ViewingWeightsPtr const& w)
  {
    master_weights_ = w;
    if (master_weights_.expired()) {
      values_.reset();
    }
    else {
      values_ = setup_values_(*master_weights_.lock());
    }
  }

  /** @brief Synchronize the held values with the master set.
   *
   *  If empty(), this function takes the view that there is no master
   *  with which to synchronize, so no action is required -- it is a
   *  no-op.
   */
  void synchronize_with_master()
  {
    if (!empty()) {
      const auto& master_values = master_weights_.lock()->get_values();
      if (values_->Viewing()) {
        El::LockedView(*values_,
                       dynamic_cast<const ValuesType&>(master_values));
      }
      else {
        El::Copy(master_values, *values_);
      }
    }
  }

  ///@}
  /** @name Queries and accessors */
  ///@{

  /** @brief Check if the proxy is referencing a weights object. */
  bool empty() const noexcept { return values_ == nullptr; }

  /** @brief Access the values.
   *
   *  The contract of this class specifies that this function is only
   *  valid if not empty(). Users are expected to ensure this
   *  contract.
   */
  ValuesType const& values() const noexcept(!LBANN_IN_DEBUG_MODE)
  {
    LBANN_DEBUG_ASSERT_POINTER(values_);
    return *values_;
  }

  /** @brief Access the master weights object directly.
   *
   *  The contract of this class specifies that this function is only
   *  valid if not empty(). Users are expected to ensure this
   *  contract.
   */
  weights const& master_weights() const
  {
    LBANN_DEBUG_ASSERT_POINTER(master_weights_.lock());
    return *master_weights_.lock();
  }

  ViewingWeightsPtr master_weights_pointer() const
    noexcept(!LBANN_IN_DEBUG_MODE)
  {
    LBANN_DEBUG_ASSERT_POINTER(master_weights_.lock());
    return master_weights_;
  }

  ///@}
  /** @name Utility functions */
  ///@{

  /** @brief Swap contents with another WeightsProxy object. */
  void swap(WeightsProxy<TensorDataType>& other)
  {
    std::swap(master_weights_, other.master_weights_);
    std::swap(values_, other.values_);
  }

  ///@}

private:
  /** @name Private setup functions */
  ///@{

  /** @brief Establish the view of the master data. */
  ValuesPtrType
  setup_values_(data_type_weights<TensorDataType> const& dtw) const
  {
    auto const& vals = dtw.get_values();
    ValuesPtrType ret(vals.Construct(vals.Grid(), vals.Root()));
    El::LockedView(*ret, vals);
    return ret;
  }

  /** @brief Establish the target matrix storage.
   *
   *  This only participates in overload resolution if @c OtherT is
   *  different from @c TensorDataType, which has a dedicated
   *  overload.
   *
   *  @tparam OtherT (Deduced) The type of the input weights object's
   *                 values.
   */
  template <typename OtherT>
  ValuesPtrType setup_values_(data_type_weights<OtherT> const& w) const
  {
    return setup_values_as_copy_(w);
  }

  /** @brief Establish the target matrix storage. */
  ValuesPtrType setup_values_(weights const& w) const
  {
    if (auto dtw = dynamic_cast<data_type_weights<TensorDataType> const*>(&w))
      return setup_values_(*dtw);
    return setup_values_as_copy_(w);
  }

  /** @brief Create the matrix object to store the copied weights. */
  ValuesPtrType setup_values_as_copy_(weights const& w) const
  {
    // In this case, w has some other dynamic type. So we need to
    // deep-copy every time. Thus, we allocate a target for this deep
    // copy here.
    ValuesPtrType ret{ValuesType::Instantiate(w.get_matrix_distribution())};
    ret->Resize(w.get_matrix_height(), w.get_matrix_width());
    return ret;
  }
  ///@}
private:
  // These members should never observably differ in nullity.
  /** @name Private members */
  ///@{

  /** @brief The proxied master weights. */
  ViewingWeightsPtr master_weights_;

  /** @brief The values in this data type. */
  ValuesPtrType values_;

  ///@}
};

// Conform to LBANN's scheme
template <typename TensorDataType>
using weights_proxy = WeightsProxy<TensorDataType>;

} // namespace lbann
#undef LBANN_IN_DEBUG_MODE
#endif // LBANN_WEIGHTS_WEIGHTS_PROXY_HPP_INCLUDED
