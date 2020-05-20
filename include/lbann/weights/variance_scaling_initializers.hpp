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

#ifndef LBANN_WEIGHTS_VARIANCE_SCALING_INITIALIZER_HPP
#define LBANN_WEIGHTS_VARIANCE_SCALING_INITIALIZER_HPP

#include "lbann/weights/initializer.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/random.hpp"

namespace lbann {

/** @brief Generalization of "Xavier" initialization.
 *
 *  Weights values are randomly sampled from a probability
 *  distribution with a variance determined by a "fan-in" and a
 *  "fan-out" parameter.
 *
 *  Weights with variance scaling initialization are only compatible
 *  with layers that set fan-in and fan-out parameters, e.g. the
 *  convolution and fully-connected layers.
 */
template <typename TensorDataType>
class variance_scaling_initializer
  : public Cloneable<
      HasAbstractFunction<variance_scaling_initializer<TensorDataType>>,
      data_type_weights_initializer<TensorDataType>> {
public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  ///@}

public:
  variance_scaling_initializer(probability_distribution dist);
  description get_description() const override;
  void fill(AbsDistMatrixType& matrix) override;

  /** Set fan-in parameter. */
  void set_fan_in(El::Int fan_in) { m_fan_in = fan_in; }
  /** Set fan-out parameter. */
  void set_fan_out(El::Int fan_out) { m_fan_out = fan_out; }

private:
  /** Get probability distribution variance. */
  virtual TensorDataType get_variance(El::Int fan_in, El::Int fan_out) = 0;

private:
  /** Probability distribution. */
  probability_distribution m_prob_dist;
  /** Fan-in parameter. */
  El::Int m_fan_in;
  /** Fan-out parameter.*/
  El::Int m_fan_out;

};

/** @brief Fill weights with variance of 2 / (fan-in + fan-out).
 *
 *  Also called Xavier initialization.
 */
template <typename TensorDataType>
class glorot_initializer
  : public Cloneable<glorot_initializer<TensorDataType>,
                     variance_scaling_initializer<TensorDataType>> {
  using BaseType = Cloneable<glorot_initializer<TensorDataType>,
                             variance_scaling_initializer<TensorDataType>>;
public:
  glorot_initializer(probability_distribution prob_dist)
    : BaseType(prob_dist) {}
  std::string get_type() const override { return "Glorot"; }
private:
  TensorDataType get_variance(El::Int fan_in, El::Int fan_out) override;
};

/** @brief Fill weights with variance of 2 / fan-in. */
template <typename TensorDataType>
class he_initializer
  : public Cloneable<he_initializer<TensorDataType>,
                     variance_scaling_initializer<TensorDataType>> {
  using BaseType = Cloneable<he_initializer<TensorDataType>,
                             variance_scaling_initializer<TensorDataType>>;
public:
  he_initializer(probability_distribution prob_dist)
    : BaseType(prob_dist) {}
  std::string get_type() const override { return "He"; }
private:
  TensorDataType get_variance(El::Int fan_in, El::Int fan_out) override;
};

/** @brief Fill weights with variance of 1 / fan-in. */
template <typename TensorDataType>
class lecun_initializer
  : public Cloneable<lecun_initializer<TensorDataType>,
                     variance_scaling_initializer<TensorDataType>> {
  using BaseType = Cloneable<lecun_initializer<TensorDataType>,
                             variance_scaling_initializer<TensorDataType>>;
public:
  lecun_initializer(probability_distribution prob_dist)
    : BaseType(prob_dist) {}
  std::string get_type() const override { return "LeCun"; }
private:
  TensorDataType get_variance(El::Int fan_in, El::Int fan_out) override;
};

template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_glorot_initializer_from_pbuf(google::protobuf::Message const& msg);
template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_he_initializer_from_pbuf(google::protobuf::Message const& msg);
template <typename TensorDataType>
std::unique_ptr<weights_initializer>
build_lecun_initializer_from_pbuf(google::protobuf::Message const& msg);

#ifndef LBANN_VARIANCE_SCALING_INITIALIZER_INSTANTIATE
#define PROTO(T)                               \
  extern template class glorot_initializer<T>; \
  extern template class he_initializer<T>;     \
  extern template class lecun_initializer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF
#endif // LBANN_VARIANCE_SCALING_INITIALIZER_INSTANTIATE

} // namespace lbann

#endif // LBANN_WEIGHTS_VARIANCE_SCALING_INITIALIZER_HPP
