////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_TRAINING_ALGORITHM_HPP
#define LBANN_TRAINING_ALGORITHM_HPP

#include "lbann/base.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/execution_contexts/execution_context.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/cloneable.hpp"

namespace lbann {

// Forward-declare this.
class execution_context;

/** @class training_algorithm
 *  @brief Base class for LBANN training_algorithms.
 *
 *  A "training algorithm" is defined as a method for modifying one or
 *  more models, where "model" is defined in the LBANN sense (that is,
 *  a model object typically consists of a machine learning model plus
 *  a "sub-DAG" for computing a training-specific objective
 *  function). At this time, we only have support for training a
 *  single model unit, though some ad hoc methods exist for training
 *  multi-model scenarios such as GANs.
 *
 *  Logically, the inputs to a training algorithm are a model
 *  architecture (encapsulated in a model object) and a data source,
 *  and the output is a trained model (or, a set of parameters that
 *  define the action of the model). Here, "trained" means that the
 *  training algorithm has evolved the parameters until user-specified
 *  stopping criteria have been met; it does necessarily imply that
 *  any underlying optimization method has converged (or even exists)
 *  or that such a convergence is even well-defined.
 *
 *  A key capability is that training algorithms should be
 *  composable. This allows metaheuristic algorithms to simply be
 *  implemented as training algorithms constructed from one or more
 *  "inner" training algorithms.
 *
 *  @todo One component that we need to address yet is the issue of
 *        logically encapsulating multiple models, as either inputs or
 *        outputs to a training algorithm. Specifically, consider the
 *        LTFB "meta-learning" method. Rather than producing the
 *        single best model, a user might be interested in the K best
 *        models. In this case, tournament-based evolution will begin
 *        with a single model (per trainer) but could output several
 *        models. Similarly, one might begin with an arbitrary
 *        collection of models that are evolved until a single best
 *        model emerges. This draws in other issues to be addressed
 *        elsewhere in LBANN such as "How do we export models?"
 *        Currently, this is done by writing to files on disk via
 *        callbacks. However, one might imagine "in-core" interation
 *        between training and inference, perhaps in an online
 *        learning scenario, in which repeatedly writing to and
 *        reading from disk is not sufficient.
 */
class training_algorithm
  : public Cloneable<HasAbstractFunction<training_algorithm>>
{
public:
  /** @name Lifecycle Management */
  ///@{
  /** @brief Constructor
   *  @param[in] name The user-defined name of the algorithm.
   */
  training_algorithm(std::string name);
  virtual ~training_algorithm() = default;
  ///@}

  /** @name Queries */
  ///@{
  /** @brief A string identifying the type of the object. */
  virtual std::string get_type() const = 0;
  /** @brief A user-defined string identifying the algorithm object. */
  std::string const& get_name() const noexcept;
  ///@}
  /** @name Execution interfaces */
  ///@{

  /** @brief Apply the algorithm to the given model.
   *  @param[in,out] context The persistent state tracked by the model.
   *  @param[in,out] model A model architecture with trainable
   *                       weights. On exit, the weights will have
   *                       been updated according to the algorithm.
   *  @param[in,out] dc The data source for this round of training.
   *  @param[in] mode IMO, superfluous. Will be removed.
   *  @param[in] term_criteria A description of when to stop training.
   */
  virtual void apply(execution_context& context,
                     model& model,
                     data_coordinator& dc,
                     execution_mode mode,
                     termination_criteria const& term_criteria) = 0;

  /** @brief Setup a collection of models.
   *  @param[in] models The collection of models to be setup.
   *  @param[in] max_mini_batch_size The largest minibatch size
   *             accepted by any model.
   *  @param[in] dr_metadata The data reader metadata that might be
   *             used when initializing certain model components.
   *  @todo Remove the dr_metadata argument.
   */
  void setup_models(std::vector<observer_ptr<model>> const& models,
                    size_t max_mini_batch_size,
                    DataReaderMetaData& dr_metadata);
  ///@}

protected:
  /** @name In-hierarchy Lifecycle Management */
  ///@{
  training_algorithm(const training_algorithm& other) = default;
  training_algorithm& operator=(const training_algorithm& other) = default;
  training_algorithm(training_algorithm&& other) = default;
  training_algorithm& operator=(training_algorithm&& other) = default;
  ///@}
private:
  /** @brief The user-defined name of the algorithm. */
  std::string m_name;
};

} // namespace lbann

#endif // LBANN_TRAINING_ALGORITHM_HPP
