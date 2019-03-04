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
#include "lbann/models/model.hpp"
#include "lbann/execution_contexts/execution_context.hpp"
//#include "lbann/layers/layer.hpp"
#include "lbann/utils/summary.hpp"
#include "lbann/utils/graph.hpp"
#include "lbann/io/file_io.hpp"
#include "lbann/io/persist.hpp"
//#include "lbann/objective_functions/objective_function.hpp"
#include "lbann/metrics/metric.hpp"
#include "lbann/weights/weights.hpp"
//#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/threads/thread_pool.hpp"
#include <lbann.pb.h>
#include <vector>
#include <string>
#include <unordered_map>

namespace lbann {

// Forward-declare this.
class lbann_callback;
class model;
class training_algorithm;
class termination_criteria;

/** Create a hash function for hashing a std::pair type */
struct pair_hash
{
  template <class T1>
  std::size_t operator() (const std::pair<T1, execution_mode> &pair) const
  {
    using underlying_t = typename std::underlying_type<execution_mode>::type;
    return std::hash<T1>()(pair.first) ^ std::hash<underlying_t>()(static_cast<underlying_t>(pair.second));
  }
};

/** Base class for LBANN trainers. */
class trainer {
public:

  /** Constructor. */
  trainer(lbann_comm *comm,
        int mini_batch_size);

  /** Copy constructor. */
  trainer(const trainer& other);
  /** Copy assignment operator. */
  trainer& operator=(const trainer& other);
  /** Destructor. */
  virtual ~trainer();
  /** Copy trainer. */
  virtual trainer* copy() const = 0;

  /** Return the trainer's type. */
  virtual std::string get_type() const = 0;

  /** Set the trainer's name; this is an arbitrary string
   *  that may be useful in multi-trainer scenarios, e.g,
   *  LTFB, jag
   */
  void set_name(std::string name);

  /** Return the trainer's name; this is an arbitrary string
   *  that may be useful in multi-trainer scenarios, e.g,
   *  LTFB, jag
   */
  std::string get_name() const {
    return m_name;
  }

  /** Human-readable description. */
  virtual description get_description() const;

  /** Set up the trainer. */
  virtual void setup(std::unique_ptr<thread_pool> io_thread_pool);

  virtual std::pair<observing_ptr<model>, execution_mode>
  check_and_build_execution_context(observing_ptr<training_algorithm> alg,
                                    observing_ptr<model> model,
                                    execution_mode mode);

  virtual void apply(observing_ptr<training_algorithm> alg,
                     observing_ptr<model> model,
                     execution_mode mode,
                     termination_criteria const& term_criteria);

  virtual void train(observing_ptr<model> model, El::Int num_epochs, El::Int num_batches=0);

  virtual void evaluate(observing_ptr<model> model, execution_mode mode, El::Int num_batche=0);

  /** Register a new callback for the trainer. */
  void add_callback(lbann_callback *cb);

  /** Get the list of callbacks for the trainer. */
  virtual std::vector<lbann_callback*>& get_callbacks() {
    return m_callbacks;
  }

  /** Return the I/O thread pool */
  //  observing_ptr<thread_pool> get_io_thread_pool() { return m_io_thread_pool.get(); }

  observing_ptr<thread_pool> get_io_thread_pool() const { return static_cast<observing_ptr<thread_pool>>(m_io_thread_pool.get()); }

  /** Get the trainer's comm. */
  inline lbann_comm *get_comm() const {
    return m_comm;
  }


  /** Run one epoch using only the input layer; this supports
   *  data_store functionality
   */
  //  void collect_indices(execution_mode mode);

  /** Set a flag that can be used to enable / disable the background I/O activities */
  void allow_background_io_activity(bool enable) { m_background_io_allowed = enable; }

  /** Are background I/O activities enabled by the input layers */
  bool background_io_activity_allowed() { return m_background_io_allowed; }

  /** Checkpoint trainer to given file descriptor, return number of bytes written */
  virtual bool save_to_checkpoint_shared(persist& p);
  /** Restore trainer by reading checkpoint from given file descriptor, return number of bytes read */
  virtual bool load_from_checkpoint_shared(persist& p);

  virtual bool save_to_checkpoint_distributed(persist& p);
  virtual bool load_from_checkpoint_distributed(persist& p);

  /** Saves the trainer explicitly if the save_trainer callback is present */
  virtual bool save_trainer();

  /** Write trainer to proto file */
  virtual void write_proto(lbann_data::Trainer* proto);

private:

  /** Give trainer a name. */
  std::string m_name;

  /** Communicator for the trainer. */
  lbann_comm *m_comm;
  /** Current callbacks to process. */
  std::vector<lbann_callback *> m_callbacks;

  /** Threads available for I/O */
  std::unique_ptr<thread_pool> m_io_thread_pool;

  /** Flag that allows input layers to fetch data in the background */
  bool m_background_io_allowed;

  /** @brief Map from model and execution mode to its execution context */
  //  std::map<execution_mode, std::unordered_map<observing_ptr<model>, std::unique_ptr<execution_context>>> m_execution_context;
  std::unordered_map<std::pair<observing_ptr<model>, execution_mode>,
                     std::unique_ptr<execution_context>,
                     pair_hash> m_model_execution_context;

  std::unique_ptr<training_algorithm> m_default_training_alg;
};

}  // namespace lbann

#endif  // LBANN_TRAINER_HPP
