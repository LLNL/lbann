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
#ifndef LBANN_WEIGHTS_WEIGHTS_PROXY_HPP_INCLUDED
#define LBANN_WEIGHTS_WEIGHTS_PROXY_HPP_INCLUDED

#include "lbann/weights/weights.hpp"

namespace lbann {

/** @class WeightsProxy
 *  @brief Proxy a weights object as a different data type.
 *
 *  Access is read-only; updates to the proxied weights must be
 *  handled directly through the origin object.
 *
 *  I'm thinking about ways to enforce this without users having to
 *  think (how horrible would that be!), but right now the contract
 *  for this class is that the weights are read from the master copy
 *  at the beginning of forward prop and then remain constant
 *  throughout the iteration. At the end of the iteration, the proxy
 *  values are considered invalid once gradient updates have begun.
 *
 *  @tparam TensorDataType The type to which the weights are proxied.
 */
template <typename TensorDataType>
class WeightsProxy
{
  using DataTypeWeights = data_type_weights<TensorDataType>;
  using ValuesType = El::AbstractDistMatrix<TensorDataType>;
  using ValuesPtrType = std::unique_ptr<ValuesType>;
public:

  /** @brief Construct an empty proxy. */
  WeightsProxy() = default;

  /** @brief Construct a proxy given the master object.
   *
   *  @param w Master weights object, which must have a valid storage
   *           matrix initialized internally.
   */
  WeightsProxy(weights const& w)
    : master_weights_{&w},
      values_{setup_values_(w)}
  {}

  /** @brief Copy a WeightsProxy object.
   *
   *  Creates a new WeightsProxy to the same object.
   */
  WeightsProxy(WeightsProxy const& other)
    : WeightsProxy(other.master_weights_)
  {}

  /** @brief Move a WeightsProxy object. */
  WeightsProxy(WeightsProxy&& other)
    : master_weights_{other.master_weights_},
      values_{std::move(other.values_)}
  {}

  WeightsProxy& operator=(WeightsProxy const& other)
  {
    this->setup(other.master_weights_);
    return *this;
  }

  WeightsProxy& operator=(WeightsProxy&& other)
  {
    WeightsProxy<TensorDataType> tmp(std::move(other));
    this->swap(tmp);
    return *this;
  }

  /** @brief Check if the proxy is referencing a weights object. */
  bool empty() const noexcept { return values_ == nullptr; }

  /** @brief Provide setup function for delayed construction.
   *
   *  This overwrites any existing data.
   *
   *  @param w The weights object to be proxied.
   */
  void setup(weights const& w)
  {
    auto vals = setup_values_(w);
    master_weights_ = &w;
    std::swap(vals, values_);
  }

  /** @brief Swap contents with another WeightsProxy object. */
  void swap(WeightsProxy<TensorDataType>& other)
  {
    std::swap(master_weights_, other.master_weights_);
    std::swap(values_, other.values_);
  }

  /** @brief Synchronize the held values with the master set. */
  void synchronize_with_master()
  {
    if (!values_->Viewing())
      El::Copy(master_weights_->get_values(), *values_);
  }

  /** @brief Access the values. */
  ValuesType const& values() const noexcept { return *values_; }

  /** @brief Access the master weights object directly. */
  weights const& master_weights() const noexcept { return *master_weights_; }

private:
  /** @brief Establish the view of the master data. */
  ValuesPtrType setup_values_(
    data_type_weights<TensorDataType> const& dtw) const
  {
    auto const& vals = dtw.get_values();
    ValuesPtrType ret(vals.Construct(vals.Grid(), vals.Root()));
    El::LockedView(*ret, vals);
    return ret;
  }

  /** @brief Establish the target matrix storage. */
  ValuesPtrType setup_values_(weights const& w) const
  {
    if (auto dtw = dynamic_cast<data_type_weights<TensorDataType> const*>(&w))
      return setup_values_(*dtw);

    // In this case, w has some other dynamic type. So we need to
    // deep-copy every time. Thus, we allocate a target for this deep
    // copy here.
    ValuesPtrType ret{ValuesType::Instantiate(w.get_matrix_distribution())};
    ret->Resize(w.get_matrix_height(), w.get_matrix_width());
    return ret;
  }

private:
  /** @brief The proxied master weights. */
  weights const* master_weights_ = nullptr;

  /** @brief The values in this data type. */
  ValuesPtrType values_;
};

// Conform to LBANN's scheme
template <typename TensorDataType>
using weights_proxy = WeightsProxy<TensorDataType>;

}// namespace lbann
#endif // LBANN_WEIGHTS_WEIGHTS_PROXY_HPP_INCLUDED
