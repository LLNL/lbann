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

#ifndef LBANN_TRAINER_HPP
#define LBANN_TRAINER_HPP

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/data_coordinator/data_coordinator.hpp"
#include "lbann/models/model.hpp"
#include "lbann/execution_contexts/execution_context.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/utils/threads/thread_pool.hpp"
#include "lbann/utils/hash.hpp"
#include <lbann.pb.h>
#include <vector>
#include <string>
#include <unordered_map>

namespace lbann {

// Forward-declare this.
class lbann_callback;
class training_algorithm;
class termination_criteria;

/** Represents an LBANN trainer and its context. */
class trainer {
public:

  /** Constructor. */
  trainer(lbann_comm *comm,
          size_t mini_batch_size,
          std::map<execution_mode, generic_data_reader *> data_readers);

  /** Copy constructor. */
  trainer(const trainer& other);
  /** Copy assignment operator. */
  trainer& operator=(const trainer& other);
  /** Destructor. */
  ~trainer();

  /** Archive for checkpoint and restart */
<<<<<<< 3dcc5bdd96b4d44d7ec3922f2e5d91922755a59e
  template <class Archive> void serialize(Archive & ar) {
    ar(CEREAL_NVP(m_persist));
=======
  template <class Archive> void serialize( Archive & ar ) {
    ar(CEREAL_NVP(m_max_mini_batch_size));
>>>>>>> Moved the data reader out of the layer factories where they are
  }

  /** Set the trainer's name; this is an arbitrary string
   *  that may be useful in multi-trainer scenarios, e.g,
   *  LTFB, jag
   */
  void set_name(std::string const& name);

  /** Return the trainer's name; this is an arbitrary string
   *  that may be useful in multi-trainer scenarios, e.g,
   *  LTFB, jag
   */
  std::string get_name() const {
    return m_name;
  }

  /** Human-readable description. */
  description get_description() const;

  /** @brief Get the list of callbacks for the trainer. */
  std::vector<observer_ptr<callback_base>> get_callbacks() {
    std::vector<observer_ptr<callback_base>> callback_list;
    callback_list.reserve(m_callbacks.size());
    for (const auto& ptr : m_callbacks) {
      callback_list.push_back(ptr.get());
    }
    return callback_list;
  }

  void add_callback(std::shared_ptr<callback_base> cb) {
    if (cb == nullptr) {
      throw lbann_exception("model: Attempted to add null pointer as a callback.");
    }
    m_callbacks.push_back(std::move(cb));
  }

  std::vector<std::shared_ptr<callback_base>>& get_callbacks_with_ownership() {
    return m_callbacks;
  }

  /** Set up the trainer. */
  void setup(std::unique_ptr<thread_pool> io_thread_pool);

  using execution_context_key_pair_t = typename std::pair<observer_ptr<model>, execution_mode>;

  execution_context_key_pair_t
  check_and_build_execution_context(training_algorithm& alg,
                                    observer_ptr<model> model,
                                    execution_mode mode);

  execution_context_key_pair_t
  check_and_build_execution_context(execution_context& c,
                                    model& model,
                                    execution_mode mode);

  execution_context& get_execution_context(observer_ptr<model> model,
                                                                 execution_mode mode);

  execution_context& get_execution_context(execution_context_key_pair_t key);

  void delete_execution_context(execution_context_key_pair_t key);

  void for_each_execution_context(std::function<void(observer_ptr<execution_context>)>fn);

  data_coordinator& get_data_coordinator() { return m_data_coordinator; }

  void apply(training_algorithm& alg,
             observer_ptr<model> model,
             execution_mode mode,
             termination_criteria const& term_criteria);

  void train(observer_ptr<model> model, El::Int num_epochs, El::Int num_batches=0);

  void evaluate(observer_ptr<model> model, execution_mode mode, El::Int num_batches=0);

  /** Return the I/O thread pool */
  thread_pool& get_io_thread_pool() const {
    if (!m_io_thread_pool) { LBANN_ERROR("m_io_thread_pool is null"); }
    return *(m_io_thread_pool.get());
  }

  /** Get the trainer's comm. */
  inline lbann_comm *get_comm() const {
    return m_comm;
  }

  /** Get the trainer's persist object */
  inline persist& get_persist_obj() {
    return m_persist;
  }

  /** Get the trainer's maximum mini-batch size. */
  inline size_t get_max_mini_batch_size() const {
    return m_max_mini_batch_size;
  }

  /** Set a flag that can be used to enable / disable the background I/O activities */
  void allow_background_io_activity(bool enable) { m_background_io_allowed = enable; }

  /** Are background I/O activities enabled by the input layers */
  bool background_io_activity_allowed() { return m_background_io_allowed; }

  // ===========================================
  // Checkpointing
  // ===========================================

  /** @brief Checkpoint model to given file descriptor, return number of bytes written */
  bool save_to_checkpoint_shared();
  /** @brief Restore model by reading checkpoint from given file descriptor, return number of bytes read */
  bool load_from_checkpoint_shared(persist& p);
  bool load_from_checkpoint_shared(model& m, execution_context& c);

  bool save_to_checkpoint_distributed();
  bool load_from_checkpoint_distributed(persist& p);
  bool load_from_checkpoint_distributed(model& m, execution_context& c);

  /** @brief Write model to proto file */
  void write_proto(lbann_data::Trainer* proto);

private:

  /** Give trainer a name. */
  std::string m_name;

  /** Communicator for the trainer. */
  lbann_comm *m_comm;

  /** @details Maximum possible minibatch size supported by models and
   *  layers in this trainer.  Note that this field will eventually be
   *  local to the particular, instance of the training context..
   */
  size_t m_max_mini_batch_size;

  /** Threads available for I/O */
  std::unique_ptr<thread_pool> m_io_thread_pool;

  /** Flag that allows input layers to fetch data in the background */
  bool m_background_io_allowed;

  /** Persist object used for serializing LBANN classes */
  persist m_persist;

  /** Hash function for @c m_model_execution_context */
  using model_execution_context_hash_t = pair_hash<observer_ptr<model>,
                                                   execution_mode,
                                                   std::hash<observer_ptr<model>>,
                                                   enum_hash<execution_mode>>;

  /** @brief Map from model and execution mode to its execution context */
  std::unordered_map<std::pair<observer_ptr<model>, execution_mode>,
                     std::unique_ptr<execution_context>,
                     model_execution_context_hash_t> m_model_execution_context;

  /** @brief Current callbacks to process. */
  std::vector<std::shared_ptr<callback_base>> m_callbacks;

  /** @brief Data Coordinator holding trainers data readers */
  data_coordinator m_data_coordinator;
};

}  // namespace lbann

#endif  // LBANN_TRAINER_HPP
