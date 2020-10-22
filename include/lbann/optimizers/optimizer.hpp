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

#ifndef LBANN_OPTIMIZERS_OPTIMIZER_HPP_INCLUDED
#define LBANN_OPTIMIZERS_OPTIMIZER_HPP_INCLUDED

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/cloneable.hpp"
#include "lbann/utils/compiler_control.hpp"
#ifdef LBANN_HAS_GPU
#include "lbann/utils/gpu/helpers.hpp"
#endif // LBANN_HAS_GPU
#include "lbann/utils/description.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/weights/weights.hpp"

#include <cereal/types/utility.hpp>

#include <memory>
#include <string>
#include <unordered_set>

namespace lbann {

/** @brief Status of values in objective function gradient. */
enum class optimizer_gradient_status {
  /** @brief Values can be accessed immediately. */
  ready,
  /** @brief Values have been cleared.
   *  @details Buffer must be zeroed out before accessing.
   */
  cleared,
  /** @brief Allreduce is needed before accessing values. */
  allreduce_needed,
  /** @brief Allreduce on values is in progress.
   *  @details Non-blocking allreduce must be synchronized before
   *  accessing.
   */
  allreduce_started,
};

/** @brief Human-readable string for status of gradient in optimizer. */
std::string to_string(optimizer_gradient_status status);

// Forward declarations
class persist;

/** @brief Abstract base class for gradient-based optimization algorithms.
 *
 *  Uses a variant of stochastic gradient descent to optimize the
 *  values in a @c weights instance. The weights values are
 *  iteratively adjusted to minimize an objective function. Each
 *  optimization step requires the objective function gradient
 *  w.r.t. the weights.
 */
class optimizer : public Cloneable<HasAbstractFunction<optimizer>> {
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

  /** @name Gradient update management */
  ///@{

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
  template <typename TensorDataType>
  void add_to_gradient(El::AbstractDistMatrix<TensorDataType> const& contrib,
                       TensorDataType scale = 1.f,
                       bool allreduce_needed = false) {
    TensorDataType buf_scale, in_scale;
    auto& grad = get_gradient_buffer(buf_scale, in_scale, allreduce_needed);
    El::Scale(buf_scale, grad);
    El::Axpy(in_scale*scale, contrib, grad);
  }

  /** @brief Zero out the objective function gradient w.r.t. the weights. */
  void clear_gradient() {
    for (auto& g : gradients_) {
      if (g.second->get_status() ==
          optimizer_gradient_status::allreduce_started) {
        g.second->complete_allreduce(*m_comm);
      }
      g.second->clear();
    }
    this->get_gradient_sources().clear();
  }

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
   *  are no more gradient sources remaining, a non-blocking allreduce
   *  will be launched on the gradient, if needed.
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
   *  @param allreduce_needed Whether this gradient contribution will need to
   *                          be allreduced.
   */
  template <typename TensorDataType>
  El::AbstractDistMatrix<TensorDataType>& get_gradient_buffer(
    TensorDataType& buf_scale,
    TensorDataType& in_scale,
    bool allreduce_needed = false);

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
  template <class Archive> void serialize(Archive & ar) {
    // Do not save the optimizer's step time
  }

  virtual bool save_to_checkpoint_shared(persist& p, std::string m_name) = 0;
  virtual bool load_from_checkpoint_shared(persist& p, std::string m_name) = 0;
  virtual bool save_to_checkpoint_distributed(persist& p, std::string m_name) = 0;
  virtual bool load_from_checkpoint_distributed(persist& p, std::string m_name) = 0;
  ///@}

protected:
  /** @brief Manage gradient information. */
  class GradientHelper {
  public:
    virtual ~GradientHelper() = default;
    optimizer_gradient_status get_status() const noexcept { return status_; }
    void set_status(optimizer_gradient_status s) noexcept { status_ = s; }
    virtual El::BaseDistMatrix& gradient() noexcept = 0;
    virtual El::BaseDistMatrix const& gradient() const noexcept = 0;
    virtual void start_allreduce(lbann_comm&) = 0;
    virtual void complete_allreduce(lbann_comm&) = 0;
    virtual void clear() = 0;
  private:
    optimizer_gradient_status status_ = optimizer_gradient_status::cleared;
  };// class GradientHelper

  template <typename TensorDataType>
  class GradientHelperImpl : public GradientHelper {
  public:
    using AbsDistMatType = El::AbstractDistMatrix<TensorDataType>;
  public:
    GradientHelperImpl(El::Int height, El::Int width, El::DistData dist_data)
      : gradient_{AbsDistMatType::Instantiate(dist_data)}
    {
      El::Zeros(*gradient_, height, width);
    }
    AbsDistMatType& gradient() noexcept override { return *gradient_; }
    AbsDistMatType const& gradient() const noexcept override {
      return *gradient_;
    }
    void start_allreduce(lbann_comm& comm) override {
      switch (this->get_status()) {
      case optimizer_gradient_status::allreduce_needed:
        comm.nb_allreduce(*gradient_,
                          gradient_->RedundantComm(),
                          allreduce_req_);
        this->set_status(optimizer_gradient_status::allreduce_started);
        break;
      case optimizer_gradient_status::ready:
      case optimizer_gradient_status::cleared:
      case optimizer_gradient_status::allreduce_started:
        break;
      default: LBANN_ERROR("unexpected gradient status "
                           "(" + to_string(this->get_status()) + ")");
      }
    }
    void complete_allreduce(lbann_comm& comm) override {
      switch (this->get_status()) {
      case optimizer_gradient_status::allreduce_started:
        comm.wait(allreduce_req_);
        this->set_status(optimizer_gradient_status::ready);
        break;
      case optimizer_gradient_status::ready:
      case optimizer_gradient_status::cleared:
        break;
      case optimizer_gradient_status::allreduce_needed:
        LBANN_ERROR("attempted to finish gradient allreduce "
                    "before starting it");
        break;
      default:
        LBANN_ERROR("unexpected gradient status "
                    "(" + to_string(this->get_status()) + ")");
      }
    }
    void clear() {
      this->set_status(optimizer_gradient_status::cleared);
    }
  private:
    std::unique_ptr<AbsDistMatType> gradient_;
    Al::request allreduce_req_;
  };// class GradientHelperImpl

  /** @brief Copy construct/copy assign */
  optimizer(const optimizer& other);
  optimizer& operator=(const optimizer& other);

  /** @brief Return the current gradient status */
  optimizer_gradient_status get_gradient_status() const {
    return m_gradient_status;
  }
  void set_gradient_status(const optimizer_gradient_status status) {
    m_gradient_status = status;
  }
  std::unordered_set<const void*>& get_gradient_sources() {
    return m_gradient_sources;
  }
  void set_comm(lbann_comm& comm) { m_comm = &comm; }

  void set_step_time(EvalType time) { m_step_time = time; }

  void inc_step_time(EvalType time) { m_step_time += time; }

  virtual std::tuple<El::Int,El::Int,El::DistData> get_matrix_info() const = 0;

  template <typename TensorDataType>
  void accumulate_all_gradient_contributions(
    El::AbstractDistMatrix<TensorDataType>& gradient);

  /** @brief Launch non-blocking allreduce on the gradient, if needed.
   *
   *  Does nothing if an allreduce is not needed or has already been
   *  started.
   */
  void start_gradient_allreduce() {
    for (auto& grad_mgr : gradients_) {
      grad_mgr.second->start_allreduce(*m_comm);
    }
  }

  /** @brief Synchronize non-blocking allreduce on the gradient, if needed.
   *
   *  Does nothing if an allreduce isn't needed. Throws an exception
   *  if an allreduce is needed but hasn't been started.
   */
  void finish_gradient_allreduce() {
    for (auto& grad_mgr : gradients_) {
      grad_mgr.second->complete_allreduce(*m_comm);
    }
  }
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
   *  safe to launch a non-blocking allreduce on the gradient, if
   *  needed.
   */
  std::unordered_set<const void*> m_gradient_sources;

  /** @brief Status of values in objective function gradient. */
  optimizer_gradient_status m_gradient_status = optimizer_gradient_status::cleared;

  /** @brief Time spent in optimization step. */
  EvalType m_step_time = 0;

  /** @brief Map from data types to gradient contributions.
   *  @todo Refactor this out. It's a hack.
   */
  using gradient_manager_type = GradientHelper;
  using gradient_manager_ptr = std::unique_ptr<gradient_manager_type>;
  std::unordered_map<std::type_index, gradient_manager_ptr> gradients_;

};

template <typename TensorDataType>
El::AbstractDistMatrix<TensorDataType>& optimizer::get_gradient_buffer(
  TensorDataType& buf_scale,
  TensorDataType& in_scale,
  bool allreduce_needed) {

  // Anon enum to clarify "get<#>" calls below.
  enum { HEIGHT=0, WIDTH, DISTDATA };
  using GradMgrType = GradientHelperImpl<TensorDataType>;

  auto& grad_mgr_ptr = gradients_[std::type_index(typeid(TensorDataType))];
  // If the manager hasn't been created, let's make it.
  if (!grad_mgr_ptr) {
    auto mat_info = this->get_matrix_info();
    grad_mgr_ptr = make_unique<GradMgrType>(
      std::get<HEIGHT>(mat_info),
      std::get<WIDTH>(mat_info),
      std::get<DISTDATA>(mat_info));
    grad_mgr_ptr->set_status(optimizer_gradient_status::cleared);
  }
  // Get the underlying matrix back out.
  auto& grad_mgr = static_cast<GradMgrType&>(*grad_mgr_ptr);
  // Complete outstanding allreduce, if needed.
  if (grad_mgr.get_status() == optimizer_gradient_status::allreduce_started) {
    grad_mgr.complete_allreduce(*(this->m_comm));
  }
  auto& buffer = grad_mgr.gradient();

  // Determine scaling factor and transition state.
  switch (grad_mgr.get_status()) {
  case optimizer_gradient_status::ready:
    buf_scale = DataType(1);
    in_scale = DataType(1);
    if (allreduce_needed) {
      buf_scale /= buffer.RedundantSize();
      grad_mgr.set_status(optimizer_gradient_status::allreduce_needed);
    }
    break;
  case optimizer_gradient_status::cleared:
    buf_scale = DataType(0);
    in_scale = DataType(1);
    grad_mgr.set_status(allreduce_needed ?
                        optimizer_gradient_status::allreduce_needed :
                        optimizer_gradient_status::ready);
    break;
  case optimizer_gradient_status::allreduce_needed:
    buf_scale = DataType(1);
    // Properly scale data that does not need to be allreduced.
    in_scale = (allreduce_needed ?
                DataType(1) :
                DataType(1) / buffer.RedundantSize());
    break;
  case optimizer_gradient_status::allreduce_started:
  default:
    LBANN_ERROR("unexpected gradient status ("
                + to_string(grad_mgr.get_status()) + ")");
  }
  return buffer;
}

template <typename TensorDataType>
void optimizer::accumulate_all_gradient_contributions(
  El::AbstractDistMatrix<TensorDataType>& gradient)
{
  using AbsDistMatType = El::AbstractDistMatrix<TensorDataType>;
  static const TensorDataType one = TensorDataType(1.f);

  // There are a few cases to note here:
  //   1. One update of the same type.
  //   2. One update of a different type.
  //   3. Multiple updates of multiple types. In this case, some work
  //      can be saved if one of the updates has the same type as
  //      "gradient".

  // Some general information
  auto num_updates = this->gradients_.size();
  auto const this_type_idx = std::type_index(typeid(TensorDataType));

  if (num_updates == 0UL)
    return;

  // Handle the case that one of the updates is TensorDataType. In
  // this case, the input gradients matrix can be made to "view" the
  // update, rather than requiring a copy.
  auto this_type_contrib = this->gradients_.find(this_type_idx);
  if (this_type_contrib != this->gradients_.end()) {
    // Check for invariant consistency.
    auto const& grad_mgr = *(this_type_contrib->second);
    if (grad_mgr.get_status() != optimizer_gradient_status::ready) {
      LBANN_ERROR("Expected ready status. Got: ",
                  to_string(grad_mgr.get_status()));
    }
    // Sync the input gradient with the contribution, one way or another.
    auto const& contrib =
      dynamic_cast<AbsDistMatType const&>(grad_mgr.gradient());
    if (contrib.DistData() == gradient.DistData()) {
      El::LockedView(gradient, contrib);
    }
    else {
      LBANN_ERROR("Should never need this copy.");
      El::Copy(contrib, gradient);
    }
    --num_updates;
  }
  else {
    // No sync possible; zero out the matrix instead
    El::Zero(gradient);
  }

  // Handle the case that only 1 update of a different type is needed.
  if (num_updates == 1UL && this->gradients_.size() == 1UL) {
    auto const& grad_mgr = *(this->gradients_.begin()->second);
    if (grad_mgr.get_status() != optimizer_gradient_status::ready) {
      LBANN_ERROR("Expected ready status. Got: ",
                  to_string(grad_mgr.get_status()));
    }
    El::Copy(grad_mgr.gradient(), gradient);
  }
  else if (this->gradients_.size() > 1UL) {
    // Need a temporary matrix for the type-casted copy.
    auto tmp = std::unique_ptr<AbsDistMatType>{
      gradient.Construct(gradient.Grid(), gradient.Root())};

    for (auto const& grad_mgr_v : this->gradients_) {
      if (grad_mgr_v.first == this_type_idx)
        continue;
      auto const& grad_mgr = *(grad_mgr_v.second);
      if (grad_mgr.get_status() != optimizer_gradient_status::ready) {
        LBANN_ERROR("Expected ready status. Got: ",
                    to_string(grad_mgr.get_status()));
      }
      auto const& grad_base = grad_mgr.gradient();
      El::Copy(grad_base, *tmp);
      El::Axpy(one, *tmp, gradient);
    }
  }
}

} // namespace lbann

#endif // LBANN_OPTIMIZERS_OPTIMIZER_HPP_INCLUDED
