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

#ifndef LBANN_OPTIMIZERS_OPTIMIZER_HPP_INCLUDED
#define LBANN_OPTIMIZERS_OPTIMIZER_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/comm_nb_request.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/compiler_control.hpp"
#ifdef LBANN_HAS_GPU
#include "lbann/utils/gpu/helpers.hpp"
#endif // LBANN_HAS_GPU
#include "lbann/utils/description.hpp"
#include "lbann/utils/memory.hpp"

#include <memory>
#include <string>
#include <typeindex>
#include <unordered_set>

namespace lbann_data {
class Optimizer;
}

namespace lbann {

/** @brief Status of values in objective function gradient. */
enum class optimizer_gradient_status
{
  /** @brief Values can be accessed immediately. */
  ready,
  /** @brief Values have been cleared.
   *  @details Buffer must be zeroed out before accessing.
   */
  cleared,
  /** @brief Synchronization (allreduce, reducescatter) is needed before
   * accessing values. */
  sync_needed,
  /** @brief Value synchronization is in progress.
   *  @details Non-blocking collective must be synchronized before
   *  accessing.
   */
  sync_started,
};

/** @brief Human-readable string for status of gradient in optimizer. */
std::string to_string(optimizer_gradient_status status);

// Forward declarations
class lbann_comm;
class persist;
class weights;

/** @brief Abstract base class for gradient-based optimization algorithms.
 *
 *  Uses a variant of stochastic gradient descent to optimize the
 *  values in a @c weights instance. The weights values are
 *  iteratively adjusted to minimize an objective function. Each
 *  optimization step requires the objective function gradient
 *  w.r.t. the weights.
 */
class optimizer : public Cloneable<HasAbstractFunction<optimizer>>
{
public:
  /** @name Constructors and Destructor */
  ///@{

  optimizer();
  virtual ~optimizer() = default;

  ///@}

  /** @brief Human-readable type name. */
  virtual std::string get_type() const = 0;
  /** @brief Human-readable description. */
  virtual description get_description() const;

  virtual double get_learning_rate() const = 0;
  virtual void set_learning_rate(double) = 0;

  /** @name Gradient update management */
  ///@{

  virtual void setup(weights* w) = 0;

  /** @brief Add to the objective function gradient w.r.t. the weights.
   *  @param contrib            Contribution to gradient.
   *  @param scale              Scaling factor for gradient
   *                            contribution.
   *  @param sync_needed        Whether the gradient contribution
   *                            requires a synchronization (e.g., allreduce)
   *                            over its redundant
   *                            communicator. If false, duplicated data
   *                            (over the redundant communicator) is
   *                            assumed to be identical. If true, a
   *                            synchronization is performed lazily when the
   *                            gradient is accessed.
   */
  template <typename TensorDataType>
  void add_to_gradient(El::AbstractDistMatrix<TensorDataType> const& contrib,
                       TensorDataType scale = 1.f,
                       bool sync_needed = false);

  /** @brief Zero out the objective function gradient w.r.t. the weights. */
  void clear_gradient();

  /** @brief Objects that are expected to contribute to the gradient. */

  El::Int get_num_gradient_sources() const;
  /** @brief Register a gradient source.
   *
   *  Any object that uses the weights and influences the objective
   *  function is expected to contribute to the objective function
   *  gradient. These objects should register themselves during
   *  forward prop.
   */
  void add_gradient_source(const void* source);

  /** @brief Unregister a gradient source.
   *
   *  When an object adds its contribution to the objective function
   *  gradient during back prop, it should unregister itself. If there
   *  are no more gradient sources remaining, a non-blocking synchronization
   *  collective will be launched on the gradient, if needed.
   */
  void remove_gradient_source(const void* source);

  /** @brief Perform optimization step. */
  virtual void step() = 0;

  /** @brief Get the gradient buffer.
   *
   *  This provides access to the underlying gradient buffer, which
   *  may be directly summed into. This buffer should be considered
   *  ephemeral and not stored. The caller must also ensure the buffer
   *  has an appropriate distribution. buf_scale provides the caller
   *  with a scale factor that must be applied to the gradient buffer
   *  before writing to it, and in_scale provides a scaling factor
   *  that must be applied to the user's data.  Essentially, this
   *  enables computations of the form
   *  @verbatim
   *    gradient = buf_scale*gradient + in_scale*new_gradient
   *  @endverbatim
   *  This is an expert-mode function and is intended to help
   *  eliminate copies and facilitate kernel fusion.
   *
   *  @param buf_scale A scale factor provided to the caller to scale
   *                   the returned buffer by.
   *  @param in_scale A scale factor provided to the caller to scale
   *                  their gradient contributions by.
   *  @param sync_needed Whether this gradient contribution will need to
   *                     be synchronized across ranks.
   */
  template <typename TensorDataType>
  El::AbstractDistMatrix<TensorDataType>&
  get_gradient_buffer(TensorDataType& buf_scale,
                      TensorDataType& in_scale,
                      bool sync_needed = false);

  ///@}
  /** @brief Communicator access */
  ///@{

  /** @brief Access LBANN communicator. */
  lbann_comm& get_comm() { return *m_comm; }

  /** @brief Access LBANN communicator. */
  const lbann_comm& get_comm() const { return *m_comm; }

  ///@}
  /** @brief Statistics access and management */
  ///@{

  /** @brief Time spent in optimization step. */
  EvalType get_step_time() const { return m_step_time; }

  /** @brief Reset stats counters. */
  virtual void reset_counters() { m_step_time = 0; }

  ///@}
  /** @name Checkpointing */
  ///@{

  /** @brief Store state to archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  ///@}

  /** @brief Add optimizer data to prototext */
  virtual void write_proto(lbann_data::Optimizer& proto) const = 0;

  /** @brief Manage gradient information. */
  class GradientHelper
  {
  public:
    virtual ~GradientHelper() = default;
    optimizer_gradient_status get_status() const noexcept { return status_; }
    void set_status(optimizer_gradient_status s) noexcept { status_ = s; }
    virtual El::BaseDistMatrix& local_gradient() noexcept = 0;
    virtual El::BaseDistMatrix const& local_gradient() const noexcept = 0;
    virtual El::BaseDistMatrix& global_gradient() noexcept = 0;
    virtual El::BaseDistMatrix const& global_gradient() const noexcept = 0;
    virtual void start_sync(lbann_comm&) = 0;
    virtual void complete_sync(lbann_comm&) = 0;
    virtual void clear() = 0;

  private:
    optimizer_gradient_status status_ = optimizer_gradient_status::cleared;
  }; // class GradientHelper

  template <typename TensorDataType>
  class GradientHelperImpl : public GradientHelper
  {
  public:
    using AbsDistMatType = El::AbstractDistMatrix<TensorDataType>;

  public:
    GradientHelperImpl(El::Int height,
                       El::Int width,
                       El::DistData dist_data,
                       El::DistData grad_dist_data,
                       bool sharded_weights)
      : local_gradient_contrib_{AbsDistMatType::Instantiate(dist_data)},
        global_gradient_{AbsDistMatType::Instantiate(grad_dist_data)},
        sharded_weights_(sharded_weights)
    {
      El::Zeros(*local_gradient_contrib_, height, width);
      if (grad_dist_data == dist_data) {
        El::View(*global_gradient_, *local_gradient_contrib_);
      }
      else {
        El::Zeros(*global_gradient_, height, width);
      }
    }
    AbsDistMatType& local_gradient() noexcept override
    {
      return *local_gradient_contrib_;
    }
    AbsDistMatType const& local_gradient() const noexcept override
    {
      return *local_gradient_contrib_;
    }
    AbsDistMatType& global_gradient() noexcept override
    {
      return *global_gradient_;
    }
    AbsDistMatType const& global_gradient() const noexcept override
    {
      return *global_gradient_;
    }
    void start_sync(lbann_comm& comm) override;
    void complete_sync(lbann_comm& comm) override;
    void clear() override;

  private:
    /** Matches the distribution of gathered (unsharded) weights in backprop. */
    std::unique_ptr<AbsDistMatType> local_gradient_contrib_;

    /** Matches the distribution of data_type_optimizer<T>::m_gradient (i.e.,
     *  post synchronization). Will view said matrix if only one data type
     *  exists.
     */
    std::unique_ptr<AbsDistMatType> global_gradient_;

    Al::request sync_req_;
    bool sharded_weights_;
  }; // class GradientHelperImpl

  /** @brief Copy construct/copy assign */
  optimizer(const optimizer& other);
  optimizer& operator=(const optimizer& other);

  /** @brief Return the current gradient status */
  optimizer_gradient_status get_gradient_status() const
  {
    return m_gradient_status;
  }
  void set_gradient_status(const optimizer_gradient_status status)
  {
    m_gradient_status = status;
  }
  std::unordered_set<const void*>& get_gradient_sources()
  {
    return m_gradient_sources;
  }
  void set_comm(lbann_comm& comm) { m_comm = &comm; }

  void set_step_time(EvalType time) { m_step_time = time; }

  void inc_step_time(EvalType time) { m_step_time += time; }

  /** Are parent weights sharded across ranks? */
  virtual bool is_sharded() const = 0;

  virtual std::tuple<El::Int, El::Int, El::DistData, El::DistData>
  get_matrix_info() const = 0;

  template <typename TensorDataType>
  void accumulate_all_gradient_contributions(
    El::AbstractDistMatrix<TensorDataType>& gradient);

  /** @brief Launch non-blocking synchronization on the gradient, if needed.
   *
   *  Does nothing if an allreduce/reducescatter is not needed or has already
   *  been started.
   */
  void start_gradient_sync();

  /** @brief Synchronize non-blocking collectives on the gradient, if needed.
   *
   *  Does nothing if a synchronization isn't needed. Throws an exception
   *  if a synchronization is needed but hasn't been started.
   */
  void finish_gradient_sync();

private:
  /** @brief LBANN communicator. */
  lbann_comm* m_comm;

  /** @brief Sources of gradient contributions.
   *
   *  This set contains pointers to objects (e.g. layers and objective
   *  function terms) that contribute to the objective function
   *  gradient. Objects should register themselves as they use the
   *  weights during forward prop and unregister themselves as they
   *  add their gradient contributions. Once this set is empty, it is
   *  safe to launch a non-blocking synchronization collective on the gradient,
   *  if needed.
   */
  std::unordered_set<const void*> m_gradient_sources;

  /** @brief Status of values in objective function gradient. */
  optimizer_gradient_status m_gradient_status =
    optimizer_gradient_status::cleared;

  /** @brief Time spent in optimization step. */
  EvalType m_step_time = 0;

  /** @brief Map from data types to gradient contributions.
   *  @todo Refactor this out. It's a hack.
   */
  using gradient_manager_type = GradientHelper;
  using gradient_manager_ptr = std::unique_ptr<gradient_manager_type>;
  std::unordered_map<std::type_index, gradient_manager_ptr>
    m_local_gradient_contributions;
};

} // namespace lbann

#endif // LBANN_OPTIMIZERS_OPTIMIZER_HPP_INCLUDED
