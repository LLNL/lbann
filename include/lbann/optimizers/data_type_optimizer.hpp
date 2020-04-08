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

#ifndef LBANN_OPTIMIZERS_DATA_TYPE_OPTIMIZER_HPP_INCLUDED
#define LBANN_OPTIMIZERS_DATA_TYPE_OPTIMIZER_HPP_INCLUDED

#include "lbann/optimizers/optimizer.hpp"

namespace lbann {

// Forward declarations
template <typename TensorDataType>
class data_type_weights;

template <typename TensorDataType>
class data_type_optimizer : public optimizer {
  friend class data_type_weights<TensorDataType>;

public:
  /** @name Public Types */
  ///@{

  /** @brief The tensor type expected in this object. */
  using AbsDistMatrixType = El::AbstractDistMatrix<TensorDataType>;

  /** @brief The concrete weights type used by this object. */
  using WeightsType = data_type_weights<TensorDataType>;

  ///@}

public:
  data_type_optimizer(TensorDataType learning_rate = 0);
  data_type_optimizer(const data_type_optimizer& other);
  data_type_optimizer& operator=(const data_type_optimizer& other);
  virtual ~data_type_optimizer() = default;

  /** @brief Create a copy of the class instance.
   *
   *  The caller is responsible for deallocating the returned object.
   */
  virtual data_type_optimizer* copy() const override = 0;

  /** Archive for checkpoint and restart */
  template <class Archive> void serialize(Archive & ar) {
    ar(cereal::base_class<optimizer>(this),
       CEREAL_NVP(m_learning_rate));
  }

  /** @brief Human-readable description. */
  virtual description get_description() const override;

  /** @brief Weights being optimized. */
  data_type_weights<TensorDataType>& get_weights();
  /** @brief Weights being optimized. */
  const data_type_weights<TensorDataType>& get_weights() const;
  /** @brief Weights being optimized. */
  void set_weights(data_type_weights<TensorDataType>* w) { m_weights = w; }

  /** @brief Objective function gradient w.r.t. the weights.
   *
   *  An allreduce may be launched and/or synchronized if needed.
   */
  AbsDistMatrixType& get_gradient();

  /** @brief Add to the objective function gradient w.r.t. the weights.
   *  @param gradient           Contribution to gradient.
   *  @param scale              Scaling factor for gradient
   *                            contribution.
   *  @param allreduce_needed   Whether the gradient contribution
   *                            requires an allreduce over its redundant
   *                            communicator. If false, duplicated data
   *                            (over the redundant communicator) is
   *                            assumed to be identical. If true, an
   *                            allreduce is performed lazily when the
   *                            gradient is accessed.
   */
  void add_to_gradient(const AbsDistMatrixType& gradient,
                       TensorDataType scale = TensorDataType(1),
                       bool allreduce_needed = false);
  /** @brief Zero out the objective function gradient w.r.t. the weights. */
  void clear_gradient() override;
  /** @brief Get the gradient buffer.
   *
   *  This provides access to the underlying gradient buffer, which may be
   *  directly summed into. This buffer should be considered ephemeral and not
   *  stored. The caller must also ensure the buffer has an appropriate
   *  distribution. buf_scale provides the caller with a scale factor that must
   *  be applied to the gradient buffer before writing to it, and in_scale
   *  provides a scaling factor that must be applied to the user's data.
   *  Essentially, this enables computations of the form
   *  gradient = buf_scale*gradient + in_scale*new_gradient
   *  This is an expert-mode function and is intended to help eliminate copies
   *  and facilitate kernel fusion.
   *
   *  @param buf_scale A scale factor provided to the caller to scale the
   *  returned buffer by.
   *  @param in_scale A scale factor provided to the caller to scale their
   *  gradient contributions by.
   *  @param allreduce_needed Whether this gradient contribution will need to
   *  be allreduced.
   */
  AbsDistMatrixType& get_gradient_buffer(TensorDataType& buf_scale,
                                  TensorDataType& in_scale,
                                  bool allreduce_needed = false);

  /** @brief Must be called before training.
   *
   *  @param w Weights being optimized. If null, no change is made to
   *  the weights.
   */
  virtual void setup(data_type_weights<TensorDataType>* w = nullptr);

  /** @brief Unregister a gradient source.
   *
   *  When an object adds its contribution to the objective function
   *  gradient during back prop, it should unregister itself. If there
   *  are no more gradient sources remaining, a non-blocking allreduce
   *  will be launched on the gradient, if needed.
   */
  void remove_gradient_source(const void* source) override;

  /** @brief Optimization step. */
  void step() override;

  /** @brief Scaling factor for optimization step sizes. */
  TensorDataType get_learning_rate() const;
  /** @brief Scaling factor for optimization step sizes. */
  void set_learning_rate(TensorDataType learning_rate);

protected:

  /** @brief Computation for an optimization step.
   *
   *  @c values and @c gradient can be assumed to have the same
   *  distribution.
   */
  virtual void step_compute(AbsDistMatrixType& values,
                            const AbsDistMatrixType& gradient) = 0;

private:

  /** @brief Weights being optimized. */
  data_type_weights<TensorDataType>* m_weights = nullptr;

  /** @brief Objective function gradient w.r.t. weights. */
  std::unique_ptr<AbsDistMatrixType> m_gradient;

  /** @brief Workspace matrix.
   *
   *  Helps ensure gradient contributions are in the right
   *  distribution. Most of the time, this should just be a matrix
   *  view.
   */
  std::unique_ptr<AbsDistMatrixType> m_gradient_v;

  /** @brief Communication request object for gradient allreduce.
   *
   *  Used to synchronize non-blocking allreduce.
   */
  Al::request m_gradient_allreduce_req;

  /** @brief Scaling factor for optimization step sizes.
   *
   *  This is not used by the base optimizer class, but is currently
   *  used by all derived optimizer classes. There are several cases
   *  where it is convenient to expose this in the base class,
   *  e.g. for variable learning rate schedules.
   *  @todo Consider moving this to the derived classes.
   */
  TensorDataType m_learning_rate;

  /** @brief Launch non-blocking allreduce on the gradient, if needed.
   *
   *  Does nothing if an allreduce is not needed or has already been
   *  started.
   */
  void start_gradient_allreduce();

  /** @brief Synchronize non-blocking allreduce on the gradient, if needed.
   *
   *  Does nothing if an allreduce isn't needed. Throws an exception
   *  if an allreduce is needed but hasn't been started.
   */
  void finish_gradient_allreduce();
};

#ifndef LBANN_DATA_TYPE_OPTIMIZER_INSTANTIATE
#define PROTO(T)                           \
  extern template class data_type_optimizer<T>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF
#endif // LBANN_DATA_TYPE_OPTIMIZER_INSTANTIATE

} // namespace lbann

#endif // LBANN_OPTIMIZERS_DATA_TYPE_OPTIMIZER_HPP_INCLUDED
