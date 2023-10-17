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

#ifndef LBANN_OPTIMIZERS_ADAGRAD_HPP_INCLUDED
#define LBANN_OPTIMIZERS_ADAGRAD_HPP_INCLUDED

#include "lbann/io/persist.hpp"
#include "lbann/optimizers/data_type_optimizer.hpp"
#include "lbann/proto/optimizers.pb.h"

namespace lbann {

/** AdaGrad optimizer.
 *
 *  Reference:
 *
 *  John Duchi, Elad Hazan, and Yoram Singer. "Adaptive subgradient
 *  methods for online learning and stochastic optimization." Journal
 *  of Machine Learning Research 12, no. Jul (2011): 2121-2159.
 */
template <typename TensorDataType>
class adagrad : public Cloneable<adagrad<TensorDataType>,
                                 data_type_optimizer<TensorDataType>>
{
  using BaseType =
    Cloneable<adagrad<TensorDataType>, data_type_optimizer<TensorDataType>>;

public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief The optimizer base type of this object. */
  using OptimizerType = data_type_optimizer<TensorDataType>;

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  ///@}

public:
  adagrad(TensorDataType learning_rate, TensorDataType eps = 1e-8);
  adagrad(const adagrad& other);
  adagrad& operator=(const adagrad& other);
  ~adagrad() override = default;

  /** Archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  /** Human-readable type name. */
  std::string get_type() const override { return "AdaGrad"; }
  /** Human-readable description. */
  description get_description() const override;
  /** @brief Returns the optimizer state size in bytes. */
  size_t get_state_size() const override;

  using OptimizerType::setup;
  void setup(WeightsType* w = nullptr) override;

  /** Add optimizer data to prototext */
  void write_proto(lbann_data::Optimizer& opt) const final;

protected:
  friend cereal::access;

  /** @brief Default constructor.
   *  @details This constructor exists as an implementation detail of
   *  the serialization code. It is not for general use.
   */
  adagrad() : adagrad(El::To<TensorDataType>(1.f), El::To<TensorDataType>(1e-8))
  {}

  /** Computation for an optimization step. */
  void step_compute(AbsDistMatrixType& values,
                    const AbsDistMatrixType& gradient) override;

private:
  /** Small factor to avoid division by zero. */
  TensorDataType m_eps;
  /** AdaGrad cache. */
  std::unique_ptr<AbsDistMatrixType> m_cache;

  /** CPU implementation of optimization step. */
  void step_compute_cpu(AbsDistMatrixType& values,
                        const AbsDistMatrixType& gradient);
#ifdef LBANN_HAS_DNN_LIB
  /** GPU implementation of optimization step. */
  void step_compute_gpu(AbsDistMatrixType& values,
                        const AbsDistMatrixType& gradient);
#endif // LBANN_HAS_DNN_LIB
};

template <typename TensorDataType>
std::unique_ptr<optimizer>
build_adagrad_optimizer_from_pbuf(google::protobuf::Message const&);

} // namespace lbann

#endif // LBANN_OPTIMIZERS_ADAGRAD_HPP_INCLUDED
