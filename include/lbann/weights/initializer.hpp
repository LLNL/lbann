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

#ifndef LBANN_WEIGHTS_INITIALIZER_HPP
#define LBANN_WEIGHTS_INITIALIZER_HPP

#include "lbann/base.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/description.hpp"

#include <google/protobuf/message.h>

namespace lbann {

/** @brief Scheme for initializing weight values. */
class weights_initializer
  : public Cloneable<HasAbstractFunction<weights_initializer>> {
public:
  weights_initializer() = default;
  virtual ~weights_initializer() = default;

  /** Human-readable string describing concrete class. */
  virtual std::string get_type() const = 0;

  /** Human-readable description of class instance. */
  virtual description get_description() const;

};

/** @brief Scheme for initializing weight values. */
template <typename TensorDataType>
class data_type_weights_initializer
  : public Cloneable<
      HasAbstractFunction<data_type_weights_initializer<TensorDataType>>,
      weights_initializer> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

public:
  data_type_weights_initializer() = default;
  virtual ~data_type_weights_initializer() = default;

  /** Human-readable string describing concrete class. */
  std::string get_type() const override { return "data_type_weights"; }

  /** Initialize entries in a weights matrix. */
  virtual void fill(AbsDistMatrixType& matrix) = 0;

};

/** @brief Fill weights with a constant value. */
template <typename TensorDataType>
class constant_initializer
  : public Cloneable<constant_initializer<TensorDataType>,
                     data_type_weights_initializer<TensorDataType>> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

public:
  constant_initializer(TensorDataType value)
    : m_value(value)
  {}
  std::string get_type() const override { return "constant"; }
  description get_description() const override;
  void fill(AbsDistMatrixType& matrix) override;

private:

  /** Weights value. */
  TensorDataType m_value;

};

/** @brief Fill weights with values from a list.
 *
 *  The number of weight entries must exactly match the number of
 *  provided values.
 */
template <typename TensorDataType>
class value_initializer
  : public Cloneable<value_initializer<TensorDataType>,
                     data_type_weights_initializer<TensorDataType>> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

public:
  value_initializer(std::vector<TensorDataType> values)
    : m_values{std::move(values)}
  {}
  std::string get_type() const override { return "value"; }
  void fill(AbsDistMatrixType& matrix) override;

private:

  /** List of weights values. */
  std::vector<TensorDataType> m_values;

};

/** @brief Draw weights values from a uniform random distribution. */
template <typename TensorDataType>
class uniform_initializer
  : public Cloneable<uniform_initializer<TensorDataType>,
                     data_type_weights_initializer<TensorDataType>> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

 public:
  uniform_initializer(TensorDataType min = El::To<TensorDataType>(0),
                      TensorDataType max = El::To<TensorDataType>(1))
    : m_min{min}, m_max{max}
  {}
  std::string get_type() const override{ return "uniform"; }
  description get_description() const override;
  void fill(AbsDistMatrixType& matrix) override;

private:

  /** Uniform distribution minimum. */
  TensorDataType m_min;
  /** Uniform distribution maximum. */
  TensorDataType m_max;

};

/** @brief Draw weights values from a normal random distribution. */
template <typename TensorDataType>
class normal_initializer
  : public Cloneable<normal_initializer<TensorDataType>,
                     data_type_weights_initializer<TensorDataType>> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:
  normal_initializer(
    TensorDataType mean = El::TypeTraits<TensorDataType>::Zero(),
    TensorDataType standard_deviation = El::TypeTraits<TensorDataType>::One())
    : m_mean{mean},
      m_standard_deviation{standard_deviation}
  {}
  std::string get_type() const override { return "normal"; }
  description get_description() const override;
  void fill(AbsDistMatrixType& matrix) override;

private:

  /** Normal distribution mean. */
  TensorDataType m_mean;
  /** Normal distribution standard deviation. */
  TensorDataType m_standard_deviation;

};

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_constant_initializer_from_pbuf(google::protobuf::Message const& msg);

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_value_initializer_from_pbuf(google::protobuf::Message const& msg);

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_uniform_initializer_from_pbuf(google::protobuf::Message const& msg);

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_normal_initializer_from_pbuf(google::protobuf::Message const& msg);

#ifndef LBANN_INITIALIZER_INSTANTIATE
#define PROTO(T)                                          \
  extern template class data_type_weights_initializer<T>; \
  extern template class constant_initializer<T>;          \
  extern template class value_initializer<T>;             \
  extern template class uniform_initializer<T>;           \
  extern template class normal_initializer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF
#endif // LBANN_INITIALIZER_INSTANTIATE

} // namespace lbann

#endif // LBANN_WEIGHTS_INITIALIZER_HPP
