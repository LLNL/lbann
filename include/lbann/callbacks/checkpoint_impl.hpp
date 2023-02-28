//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
//
// checkpoint .hpp .cpp - Callback hooks to checkpoint model
////////////////////////////////////////////////////////////////////////////////
#ifndef LBANN_CALLBACKS_CALLBACK_CHECKPOINT_IMPL_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_CHECKPOINT_IMPL_HPP_INCLUDED

#include "lbann/callbacks/checkpoint.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/io/file_io.hpp"

namespace lbann {
namespace callback {

inline trainer& checkpoint::get_active_trainer(){
  if(m_active_trainer == nullptr) {
    LBANN_ERROR("No active trainer for the checkpoint callback");
  }
  return *m_active_trainer;
}

inline TrainingAlgorithm& checkpoint::get_active_training_algorithm()
{
  if (m_active_training_algorithm == nullptr) {
    LBANN_ERROR("No active training algorithm for the checkpoint callback");
  }
  return *m_active_training_algorithm;
}

inline std::string get_trainer_checkpoint_dirname(const std::string& trainer_name, const std::string& dir) {
  return build_string(dir, '/', trainer_name, '/');
}

inline std::string get_last_shared_checkpoint_filename(const std::string& alg_name, const std::string& dir) {
  return build_string(dir, '/', alg_name, ".last.shared.checkpoint");
}

inline std::string get_last_shared_checkpoint_filename(const std::string& trainer_name, const std::string& alg_name, const std::string& dir) {
  return get_last_shared_checkpoint_filename(alg_name, get_trainer_checkpoint_dirname(trainer_name, dir));
}

  inline std::string get_shared_checkpoint_dirname(const std::string& alg_name, const std::string& dir, visitor_hook hook, execution_mode mode, size_t epoch, size_t step) {
  return build_string(dir, '/', alg_name, ".shared.", (is_execution_mode_hook(hook) ? to_string(hook, mode) : to_string(hook)), ".epoch.", epoch, ".step.", step, '/');
}

inline std::string get_shared_checkpoint_dirname(const std::string& trainer_name, const std::string& alg_name, const std::string& dir, visitor_hook hook, execution_mode mode, size_t epoch, size_t step) {
  return get_shared_checkpoint_dirname(alg_name, get_trainer_checkpoint_dirname(trainer_name, dir), hook, mode, epoch, step);
}

inline std::string get_last_distributed_checkpoint_filename(const std::string& alg_name, const std::string& dir) {
  return build_string(dir, '/', alg_name, ".last.distributed.checkpoint");
}

inline std::string get_last_distributed_checkpoint_filename(const std::string& trainer_name, const std::string& alg_name, const std::string& dir) {
  return get_last_distributed_checkpoint_filename(alg_name, get_trainer_checkpoint_dirname(trainer_name, dir));
}

inline std::string get_distributed_checkpoint_dirname(const std::string& alg_name, const int rank_in_trainer, const std::string& dir, visitor_hook hook, execution_mode mode, size_t epoch, size_t step) {
  return build_string(dir, '/',
     alg_name,
    ".rank.", rank_in_trainer,
    ".distributed.", (is_execution_mode_hook(hook) ? to_string(hook, mode) : to_string(hook)),
    ".epoch.", epoch,
    ".step.", step, '/');
}

inline std::string get_distributed_checkpoint_dirname(const std::string& trainer_name, const std::string& alg_name, const int rank_in_trainer, const std::string& dir, visitor_hook hook, execution_mode mode, size_t epoch, size_t step) {
  return get_distributed_checkpoint_dirname(alg_name, rank_in_trainer, get_trainer_checkpoint_dirname(trainer_name, dir), hook, mode, epoch, step);
}

// Print last checkpoint to file, used to determine which checkpoint to load from.
inline bool write_latest(std::string filename, visitor_hook hook, execution_mode mode, size_t epoch, size_t train) {
  // open the file for writing
  int fd = openwrite(filename.c_str());
  if (fd != -1) {
    char field[256];
    std::string hookStr = is_execution_mode_hook(hook) ? to_string(hook, mode) : to_string(hook);
    sprintf(field, "hook=%s epoch=%ld step=%ld\n", hookStr.c_str(), epoch, train);
    write_string(fd, filename.c_str(), field, strlen(field));
    // close our file
    closewrite(fd, filename.c_str());
  }
  return true;
}

/** \brief Reads the "latest" file and returns the epoch number and
 *        sample offset for most recent checkpoint
 */
inline bool read_latest(std::string filename, visitor_hook *hook, execution_mode *mode, size_t *epochLast, size_t *trainLast) {
  // assume we don't have a file, we'll return -1 in that case
  *epochLast = -1;
  *trainLast = -1;
  *mode = execution_mode::invalid;
  *hook = visitor_hook::invalid;
  // open the file for reading
  int fd = openread(filename.c_str());
  if (fd != -1) {
    // read epoch from file
    char field[256];
    read_string(fd, filename.c_str(), field, sizeof(field));
    char hookStr[64];
    int ret = sscanf(field, "hook=%s epoch=%ld step=%ld\n", hookStr, epochLast, trainLast);
    visitor_hook_from_string(hookStr, *hook, *mode);
    // close our file
    closeread(fd, filename.c_str());
    if(ret != 3) { return false; }
    return true;
  }
  return false;
}

// Builder function
std::unique_ptr<callback_base>
build_checkpoint_callback_from_pbuf(
  const google::protobuf::Message&);

} // namespace callback
} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_CHECKPOINT_IMPL_HPP_INCLUDED
