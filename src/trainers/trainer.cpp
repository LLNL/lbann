////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

// This header
#include "lbann/trainers/trainer.hpp"

// LBANN dependencies
#include "lbann/base.hpp"
#include "lbann/callbacks/callback.hpp"
#include "lbann/data_coordinator/data_coordinator_metadata.hpp"
#include "lbann/execution_algorithms/sgd_training_algorithm.hpp"
#include "lbann/execution_algorithms/training_algorithm.hpp"
#include "lbann/execution_contexts/sgd_execution_context.hpp"
#include "lbann/io/persist_impl.hpp"
#include "lbann/utils/description.hpp"
#include "lbann/utils/memory.hpp"
#include "lbann/utils/serialize.hpp"

// LBANN proto
#include <lbann.pb.h>

// STL
#include <functional>
#include <memory>
#include <string>

namespace lbann {

////////////////////////////////////////////////////////////
// Constructors and destructor
////////////////////////////////////////////////////////////

trainer::trainer(lbann_comm* comm,
                 std::unique_ptr<data_coordinator> dc,
                 size_t mini_batch_size,
                 std::unique_ptr<training_algorithm> alg)
  : m_data_coordinator{std::move(dc)},
    m_training_alg{std::move(alg)},
    m_comm{comm},
    m_max_mini_batch_size{mini_batch_size},
    m_background_io_allowed{true}
{
  // Default trainer name
  m_name = "trainer" + std::to_string(m_comm->get_trainer_rank());
  m_data_coordinator->set_trainer(*this);
}

trainer::~trainer() {}

template <class Archive> void trainer::serialize(Archive& ar)
{
  ar(CEREAL_NVP(m_persist),
     CEREAL_NVP(m_max_mini_batch_size),
     CEREAL_NVP(m_root_random_seed),
     CEREAL_NVP(m_random_seed),
     CEREAL_NVP(m_data_seq_random_seed));
}

////////////////////////////////////////////////////////////
// Trainer specification
////////////////////////////////////////////////////////////

void trainer::set_name(std::string const& name)
{
  if (name.empty()) {
    LBANN_ERROR("attempted to rename trainer \"",
                get_name(),
                "\" with empty string");
  }
  m_name = name;
}

description trainer::get_description() const
{

  // Construct description object
  description desc(get_name());
  desc.add("Background I/O", m_background_io_allowed);

  // Result
  return desc;
}

////////////////////////////////////////////////////////////
// Setup
////////////////////////////////////////////////////////////

void trainer::setup(std::unique_ptr<thread_pool> io_thread_pool,
                    std::map<execution_mode, generic_data_reader*> data_readers)
{
  // Setup I/O threads - set up before setting up the layers (input
  // layer depends on having a properly initialized thread pool)
  m_io_thread_pool = std::move(io_thread_pool);

  m_data_coordinator.get()->setup(*m_io_thread_pool.get(),
                                  get_max_mini_batch_size(),
                                  data_readers);

  // Set up callbacks first - allow checkpoint / restart to reload state
  for (auto& cb : m_callbacks) {
    cb->setup(this);
  }
}

/// Check if there is already an execution context for the model in this mode,
/// if not create one
trainer::execution_context_key_pair_t
trainer::check_and_build_execution_context(training_algorithm& alg,
                                           observer_ptr<model> model,
                                           execution_mode mode)
{
  auto key = std::make_pair(model, mode);
  if (m_model_execution_context.count(key) == 0) {
    /// Create a execution context for each model and execution mode
    std::unique_ptr<execution_context> context;
    if (dynamic_cast<observer_ptr<sgd_training_algorithm>>(&alg) != nullptr) {
      /// @todo BVE FIXME Figure out how to get a good mini-batch size
      /// in here
      context =
        make_unique<sgd_execution_context>(mode, get_max_mini_batch_size());
    }
    else {
      LBANN_ERROR("Unknown execution algorithm type.");
    }
    m_model_execution_context.emplace(key, std::move(context));
  }
  return key;
}

/// Check if there is already an execution context for the model in this mode,
/// if not create one
trainer::execution_context_key_pair_t
trainer::check_and_build_execution_context(execution_context& c,
                                           model& model,
                                           execution_mode mode)
{
  auto key = std::make_pair(&model, mode);
  if (m_model_execution_context.count(key) == 0) {
    std::unique_ptr<execution_context> context;
    //    observer_ptr<training_algorithm> alg = const_cast
    if (dynamic_cast<observer_ptr</*const */ sgd_execution_context>>(&c) !=
        nullptr) {
      context =
        make_unique<sgd_execution_context>(mode, get_max_mini_batch_size());
    }
    else {
      LBANN_ERROR("Unknown execution context type");
    }
    m_model_execution_context.emplace(key, std::move(context));
  }
  return key;
}

execution_context& trainer::get_execution_context(observer_ptr<model> model,
                                                  execution_mode mode)
{
  auto key = std::make_pair(model, mode);
  return get_execution_context(key);
}

execution_context&
trainer::get_execution_context(execution_context_key_pair_t key)
{
  if (m_model_execution_context.count(key) == 0) {
    LBANN_ERROR("No execution context for this model / mode pair");
  }
  return static_cast<sgd_execution_context&>(
    *(m_model_execution_context[key].get()));
}

void trainer::delete_execution_context(execution_context_key_pair_t key)
{
  if (m_model_execution_context.count(key) == 0) {
    LBANN_WARNING(
      "Attempting to delete an invalid execution context for model=" +
      (key.first)->get_name() + " / " + to_string(key.second));
  }
  m_model_execution_context.erase(key);
}

/// @todo BVE FIXME seems like there is a bug here about mapping
/// execution contexts to the right model
void trainer::for_each_execution_context(
  std::function<void(observer_ptr<execution_context>)> fn)
{
  for (auto&& c : m_model_execution_context) {
    // auto&& model = c.first.first;
    // auto&& mode = c.first.second;
    auto&& context = c.second;
    fn(context.get());
  }
}

////////////////////////////////////////////////////////////
// Evaluation and training
////////////////////////////////////////////////////////////

void trainer::train(observer_ptr<model> model,
                    El::Int num_epochs,
                    El::Int num_batches)
{
  // FIXME (trb 04/22/21): This is a temporary fix to support old PFE
  // model descriptions.
  if (!m_training_alg) {
    std::unique_ptr<sgd_termination_criteria> stopping;
    if (num_epochs)
      stopping = make_unique<epoch_termination_criteria>(num_epochs);
    else
      stopping = make_unique<batch_termination_criteria>(num_batches);

    m_training_alg = std::make_unique<sgd_training_algorithm>(
      "sgd_train", std::move(stopping));
  }
  DataReaderMetaData dr_metadata = get_data_coordinator().get_dr_metadata();
  m_training_alg->setup_models({model}, get_max_mini_batch_size(), dr_metadata);

  // FIXME (trb 04/27/2021): This is a hack to support the current
  // checkpoint/restart mechanisms. This needs to be refactored to be
  // agnostic to the training algorithm. At this time, only SGD is
  // properly C/R-able.
  if (m_training_alg->get_type() == "sgd") {
    auto key = check_and_build_execution_context(*m_training_alg,
                                                 model,
                                                 execution_mode::training);
    m_training_alg->apply(*(m_model_execution_context[key]),
                          *model,
                          get_data_coordinator(),
                          execution_mode::training);
  }
  else {
    m_training_alg->apply(*model, get_data_coordinator());
  }
}

// NOTE (trb 04/19/2021): Currently, "evaluate" is just defined as
// "run forward prop and look at objective
// functions/metrics/etc". This is currently implemented in
// `sgd_training_algorithm`, so we just exploit that for now. This
// could should be refactored in the future.
void trainer::evaluate(observer_ptr<model> model,
                       execution_mode mode,
                       El::Int num_batches)
{
  auto sgd = make_unique<sgd_training_algorithm>(
    "sgd_evaluate",
    make_unique<epoch_termination_criteria>(/*num_epochs=*/1UL));
  auto ctxt = sgd->get_new_execution_context();
  ctxt->set_execution_mode(mode);
  model->reset_mode(*ctxt, execution_mode::invalid);

  DataReaderMetaData dr_metadata = get_data_coordinator().get_dr_metadata();
  sgd->setup_models({model}, get_max_mini_batch_size(), dr_metadata);

  if (m_comm->get_grid_type() == GridType::NO_GRID or
      m_comm->get_grid_type() == GridType::PRIMARY_GRID) {
    sgd->evaluate(*ctxt, *model, get_data_coordinator(), mode,
                  epoch_termination_criteria(/*num_epochs=*/1UL));
  }
}

// =============================================
// Checkpointing
// =============================================

bool trainer::save_to_checkpoint_shared()
{
  for_each_execution_context([this](observer_ptr<execution_context> ctx) {
    ctx->save_to_checkpoint_shared(this->get_persist_obj());
  });
  save_rng_to_checkpoint_shared(get_persist_obj(), m_comm);

  if (m_comm->am_trainer_master()) {
    write_cereal_archive(*this,
                         get_persist_obj(),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                         "trainer.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                         "trainer.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
    );
  }

  return get_data_coordinator().save_to_checkpoint_shared(get_persist_obj());
}

bool trainer::load_from_checkpoint_shared(persist& p)
{
  try {
    load_from_shared_cereal_archive(*this,
                                    p,
                                    *get_comm(),
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                                    "trainer.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                                    "trainer.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
    );
  }
  catch (NonexistentArchiveFile const& e) {
    LBANN_MSG(e.what());
    return false;
  }

  return get_data_coordinator().load_from_checkpoint_shared(p);
}

bool trainer::load_from_checkpoint_shared(model& m, execution_context& c)
{
  // Reload the RNG once the trainer and all of the  models are setup
  // to avoid spurious turns of the RNGs
  load_rng_from_checkpoint(get_persist_obj(), m_comm);

  execution_mode current_mode = c.get_execution_mode();
  for (execution_mode mode : execution_mode_iterator()) {
    /// Restart should optionally load any other valid contexts
    if (mode == execution_mode::invalid) {
      continue;
    }
    trainer::execution_context_key_pair_t key;
    try {
      if (current_mode == mode) {
        /// Restart has to be able to load the currently running execution
        /// context
        c.load_from_checkpoint_shared(get_persist_obj());
      }
      else {
        key = check_and_build_execution_context(c, m, mode);
        auto& evaluation_context =
          static_cast<sgd_execution_context&>(get_execution_context(key));
        evaluation_context.load_from_checkpoint_shared(get_persist_obj());
      }
    }
    catch (NonexistentArchiveFile const& e) {
      // Ignore the exception if the file is not for the current execution mode
      if (current_mode == mode) {
        LBANN_ERROR("Failed to restart model, invalid execution mode: ",
                    to_string(current_mode),
                    "\n\n  e.what(): ",
                    e.what(),
                    "\n");
      }
      else {
        delete_execution_context(key);
      }
    }
  }

  return get_data_coordinator().load_from_checkpoint_shared(get_persist_obj());
}

bool trainer::save_to_checkpoint_distributed()
{
  for_each_execution_context([this](observer_ptr<execution_context> ctx) {
    ctx->save_to_checkpoint_distributed(this->get_persist_obj());
  });
  save_rng_to_checkpoint_distributed(get_persist_obj(), m_comm);
  return get_data_coordinator().save_to_checkpoint_shared(get_persist_obj());
}

bool trainer::load_from_checkpoint_distributed(persist& p)
{
  read_cereal_archive(*this,
                      p,
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
                      "trainer.xml"
#else  // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
                      "trainer.bin"
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  );
  return get_data_coordinator().load_from_checkpoint_distributed(p);
}

bool trainer::load_from_checkpoint_distributed(model& m, execution_context& c)
{
  load_rng_from_checkpoint(get_persist_obj(), m_comm);

  execution_mode current_mode = c.get_execution_mode();

  for (execution_mode mode : execution_mode_iterator()) {
    /// Restart should optionally load any other valid contexts
    if (mode == execution_mode::invalid) {
      continue;
    }
    trainer::execution_context_key_pair_t key;
    try {
      if (current_mode == mode) {
        /// Restart has to be able to load the currently running  execution
        /// context
        c.load_from_checkpoint_distributed(get_persist_obj());
      }
      else {
        key = check_and_build_execution_context(c, m, mode);
        auto& evaluation_context =
          static_cast<sgd_execution_context&>(get_execution_context(key));
        evaluation_context.load_from_checkpoint_distributed(get_persist_obj());
      }
    }
    catch (NonexistentArchiveFile const&) {
      // Ignore the exception if the file is not for the current execution mode
      if (current_mode == mode) {
        LBANN_ERROR("Failed to restart model, invalid execution mode: " +
                    to_string(current_mode));
      }
      else {
        delete_execution_context(key);
      }
    }
  }
  return get_data_coordinator().load_from_checkpoint_distributed(
    get_persist_obj());
}

void trainer::write_proto(lbann_data::Trainer& proto)
{
  proto.Clear();
  if (m_comm->am_world_master()) {
    proto.set_mini_batch_size(m_max_mini_batch_size);
  }
}

} // namespace lbann
