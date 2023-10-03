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

#ifndef LBANN_TRAINER_HPP
#define LBANN_TRAINER_HPP

#include "lbann/base.hpp"
#include "lbann/detect_El_mpi.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/proto/lbann.pb.h"
#include "lbann/utils/hash.hpp"
#include "lbann/utils/threads/thread_pool.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lbann {

// Forward-declarations
class data_coordinator;
class description;
class lbann_comm;
class callback_base;
class ExecutionContext;
class generic_data_reader;
class TrainingAlgorithm;
class TerminationCriteria;
class model;

/** @brief User-facing class that represents a set of compute resources.
 *
 *  A trainer is responsible for managing the interactions of an
 *  `lbann_comm` object with other objects in the library, most
 *  notably `model`s and `data_coordinator`s.
 */
class trainer
{
public:
  /** @name Lifecycle management */
  ///@{

  /** @brief Construct with a communicator and data coordinator.
   *  @param[in] comm A reference to a valid `lbann_comm` object.
   *  @param[in] dc The data coordinator used by this trainer.
   *  @param[in] mini_batch_size The minibatch size? What's a minibatch? That
   *                             sounds like an SGD thing...
   *  @param[in] alg The training algorithm to use.
   *  @todo I don't know why `mini_batch_size` is here.
   */
  trainer(lbann_comm* comm,
          std::unique_ptr<data_coordinator> dc,
          size_t mini_batch_size,
          std::unique_ptr<TrainingAlgorithm> alg = nullptr);

  ~trainer();

  ///@}
  /** @name Serialization */
  ///@{

  /** @brief Archive for checkpoint and restart */
  template <class Archive>
  void serialize(Archive& ar);

  ///@}
  /** @name Configuration */
  ///@{

  /** @brief Set the trainer's name.
   *  @details This is an arbitrary string that may be useful in
   *  multi-trainer scenarios, e.g, LTFB, jag, etc.
   */
  void set_name(std::string const& name);

  /** @brief Set the random seeds used for the trainer */
  void set_random_seeds(int root_random_seed,
                        int random_seed,
                        int data_seq_random_seed)
  {
    m_root_random_seed = root_random_seed;
    m_random_seed = random_seed;
    m_data_seq_random_seed = data_seq_random_seed;
  }

  void add_callback(std::shared_ptr<callback_base> cb)
  {
    if (cb == nullptr) {
      throw lbann_exception(
        "model: Attempted to add null pointer as a callback.");
    }
    m_callbacks.push_back(std::move(cb));
  }

  /** @brief Set up the trainer. */
  void setup(std::unique_ptr<thread_pool> io_thread_pool,
             std::map<execution_mode, generic_data_reader*> data_readers);

  /** @brief Set a flag that can be used to enable / disable the
   *         background I/O activities.
   */
  void allow_background_io_activity(bool enable)
  {
    m_background_io_allowed = enable;
  }

  ///@}
  /** @name Queries */
  ///@{

  /** Return the trainer's name; this is an arbitrary string
   *  that may be useful in multi-trainer scenarios, e.g,
   *  LTFB, jag
   */
  std::string get_name() const { return m_name; }

  /** Human-readable description. */
  description get_description() const;

  int get_random_seed() const noexcept { return m_random_seed; }
  int get_data_seq_random_seed() const noexcept
  {
    return m_data_seq_random_seed;
  }

  /** @brief Get the list of callbacks for the trainer. */
  std::vector<observer_ptr<callback_base>> get_callbacks() const
  {
    std::vector<observer_ptr<callback_base>> callback_list;
    callback_list.reserve(m_callbacks.size());
    for (const auto& ptr : m_callbacks) {
      callback_list.push_back(ptr.get());
    }
    return callback_list;
  }

  std::vector<std::shared_ptr<callback_base>>& get_callbacks_with_ownership()
  {
    return m_callbacks;
  }

  const data_coordinator& get_data_coordinator() const
  {
    if (m_data_coordinator == nullptr) {
      LBANN_ERROR("data_coordinator is nullptr");
    }
    return *m_data_coordinator;
  }

  data_coordinator& get_data_coordinator()
  {
    return const_cast<data_coordinator&>(
      static_cast<const trainer&>(*this).get_data_coordinator());
  }

  /** @brief Get the I/O thread pool */
  thread_pool& get_io_thread_pool() const
  {
    if (!m_io_thread_pool) {
      LBANN_ERROR("m_io_thread_pool is null");
    }
    return *(m_io_thread_pool.get());
  }

  /** @brief Get the trainer's comm. */
  lbann_comm* get_comm() const noexcept { return m_comm; }

  /** @brief Get the trainer's persist object */
  persist& get_persist_obj() noexcept { return m_persist; }

  /** @brief Get the trainer's maximum mini-batch size. */
  size_t get_max_mini_batch_size() const noexcept
  {
    return m_max_mini_batch_size;
  }

  /** @brief Are background I/O activities enabled by the input layers */
  bool background_io_activity_allowed() const noexcept
  {
    return m_background_io_allowed;
  }

  ///@}

  using execution_context_key_pair_t =
    typename std::pair<observer_ptr<model>, execution_mode>;

  execution_context_key_pair_t
  check_and_build_execution_context(TrainingAlgorithm& alg,
                                    observer_ptr<model> model,
                                    execution_mode mode);

  execution_context_key_pair_t
  check_and_build_execution_context(ExecutionContext& c,
                                    model& model,
                                    execution_mode mode);

  ExecutionContext& get_execution_context(observer_ptr<model> model,
                                          execution_mode mode);

  ExecutionContext& get_execution_context(execution_context_key_pair_t key);

  bool execution_context_valid(model& m, execution_mode mode) const noexcept;

  bool execution_context_valid(execution_context_key_pair_t key) const noexcept;

  /** @name Training and evaluation interface */
  ///@{

  void
  train(observer_ptr<model> model, El::Int num_epochs, El::Int num_batches = 0);

  void evaluate(observer_ptr<model> model,
                execution_mode mode,
                El::Int num_batches = 0);

  ///@}
  /** @name Sub-grid management */
  ///@{
  std::vector<El::Grid*> get_grids() const;
  void add_grid(std::unique_ptr<El::Grid> g);
  ///@}
  /** @name Checkpointing */
  ///@{

  /** @brief Create a shared checkpoint of the trainer. */
  bool save_to_checkpoint_shared();

  /** @brief Restore trainer from a shared checkpoint. */
  bool load_from_checkpoint_shared(persist& p);

  /** @brief Restore model from a shared checkpoint. */
  bool load_from_checkpoint_shared(model& m, ExecutionContext& c);

  /** @brief Create a distributed checkpoint of the trainer. */
  bool save_to_checkpoint_distributed();

  /** @brief Restore a trainer from a distributed checkpoint. */
  bool load_from_checkpoint_distributed(persist& p);

  /** @brief Restore a model from a distributed checkpoint. */
  bool load_from_checkpoint_distributed(model& m, ExecutionContext& c);

  /** @brief Write trainer to proto message */
  void write_proto(lbann_data::Trainer& proto);

  ///@}

private:
  void delete_execution_context(execution_context_key_pair_t key);

  void for_each_execution_context(
    std::function<void(observer_ptr<ExecutionContext>)> fn);

private:
  /** @brief Persist object used for serializing LBANN classes. */
  persist m_persist;

  /** @brief Hash function for @c m_model_execution_context. */
  using model_execution_context_hash_t =
    pair_hash<observer_ptr<model>,
              execution_mode,
              std::hash<observer_ptr<model>>,
              enum_hash<execution_mode>>;

  using ModelContextMapType =
    std::unordered_map<std::pair<observer_ptr<model>, execution_mode>,
                       std::unique_ptr<ExecutionContext>,
                       model_execution_context_hash_t>;

  /** @brief Map from model and execution mode to its execution context */
  ModelContextMapType m_model_execution_context;

  /** @brief This trainer's name. */
  std::string m_name;

  /** @brief Current callbacks to process. */
  std::vector<std::shared_ptr<callback_base>> m_callbacks;

  /** @brief Threads available for I/O */
  std::unique_ptr<thread_pool> m_io_thread_pool;

  /** @brief Data Coordinator holding trainers data readers */
  std::unique_ptr<data_coordinator> m_data_coordinator;

  /** @brief The training algorithm being used. May be null.
   *  @details If null, a different type of execution algorithm is
   *  being used (e.g., inference).
   */
  std::unique_ptr<TrainingAlgorithm> m_training_alg;

  /** @brief Communication domain for the trainer. */
  lbann_comm* m_comm;

  /** @brief Processor grids for sub-grid parallelism
   *
   *  Does not include grid 0, which corresponds to the trainer's MPI
   *  communicator.
   */
  std::vector<std::unique_ptr<El::Grid>> m_grids;

  /** @brief Maximum possible minibatch size supported by models and
   *         layers in this trainer.
   *  @note This field will eventually be local to the particular,
   *        instance of the training context.
   */
  size_t m_max_mini_batch_size;

  /** @brief Root of the random seed tree.
   *  @details Either default or user supplied.
   */
  int m_root_random_seed;

  /** @brief Random seed used for the general RNGs. */
  int m_random_seed;

  /** @brief Random seed used for the RNG used to fetch data. */
  int m_data_seq_random_seed;

  /** @brief Flag that allows input layers to fetch data in the background. */
  bool m_background_io_allowed;
};

/** @brief Get a reference to the global trainer visible to this rank. */
trainer& get_trainer();

/** @brief Get a const reference to the global trainer visible to this
 *         rank.
 */
trainer const& get_const_trainer();

} // namespace lbann

#endif // LBANN_TRAINER_HPP
