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

#include "lbann/utils/dataset.hpp"
#include "lbann/trainers/trainer.hpp"
#include "lbann/utils/serialize.hpp"

namespace lbann {

void dataset::setup(uint64_t total_samples, std::string role)
{
  m_base_offset = 0;
  m_sample_stride = 1;
  m_stride_to_next_mini_batch = 0;
  m_stride_to_last_mini_batch = 0;
  m_current_mini_batch_idx = 0;
  m_num_iterations_per_epoch = 0;

  m_total_samples = total_samples;
  m_role = role;

  set_initial_position();
  m_initialized = (total_samples > 0);
  return;
}

template <class Archive>
void dataset::serialize(Archive& ar)
{
  ar(CEREAL_NVP(m_role),
     CEREAL_NVP(m_num_samples_processed),
     CEREAL_NVP(m_total_samples),
     CEREAL_NVP(m_current_mini_batch_idx),

     CEREAL_NVP(m_mini_batch_size),

     CEREAL_NVP(m_current_pos),

     CEREAL_NVP(m_stride_to_next_mini_batch),
     CEREAL_NVP(m_base_offset),
     CEREAL_NVP(m_sample_stride),

     CEREAL_NVP(m_last_mini_batch_size),
     CEREAL_NVP(m_stride_to_last_mini_batch),
     CEREAL_NVP(m_current_mini_batch_idx),

     CEREAL_NVP(m_num_iterations_per_epoch),

     CEREAL_NVP(m_initialized)

  );

  // Ensure that the restored archives have the proper offset for the
  // ranks specific fields
  if constexpr (utils::IsLoad<Archive>) {
    auto& trainer = get_trainer();
    auto* lbann_comm = trainer.get_comm();
    m_base_offset = lbann_comm->get_rank_in_trainer();
    m_current_pos += lbann_comm->get_rank_in_trainer();
  }
}

uint64_t dataset::get_next_mini_batch_size() const
{
  if (m_current_mini_batch_idx + 1 > (m_num_iterations_per_epoch - 1)) {
    return 0;
  }
  else if (m_current_mini_batch_idx + 1 == (m_num_iterations_per_epoch - 1)) {
    return m_last_mini_batch_size;
  }
  else {
    return m_mini_batch_size;
  }
}

uint64_t dataset::get_current_mini_batch_size() const
{
  if (m_current_mini_batch_idx > (m_num_iterations_per_epoch - 1)) {
    return 0;
  }
  else if (m_current_mini_batch_idx == (m_num_iterations_per_epoch - 1)) {
    return m_last_mini_batch_size;
  }
  else {
    return m_mini_batch_size;
  }
}

uint64_t dataset::get_next_position() const
{
  /// If the next mini-batch for this rank is going to be the last
  /// mini-batch, take the proper (possibly reduced) step to
  /// setup for the last mini-batch
  if (m_current_mini_batch_idx == (m_num_iterations_per_epoch - 1)) {
    return m_current_pos + m_stride_to_last_mini_batch;
  }
  else {
    return m_current_pos + m_stride_to_next_mini_batch;
  }
}

void dataset::set_mini_batch_size(const uint64_t s) { m_mini_batch_size = s; }

void dataset::print_config()
{
  LBANN_MSG("Configuration for dataset\n",
            " role                       = ",
            m_role,
            "\n",
            " current position           = ",
            m_current_pos,
            "\n",
            " mini_batch_size            = ",
            m_mini_batch_size,
            "\n",
            " stride_to_next_mini_batch  = ",
            m_stride_to_next_mini_batch,
            "\n",
            " base_offset                = ",
            m_base_offset,
            "\n",
            " sample_stride              = ",
            m_sample_stride,
            "\n",
            " last_mini_batch_size       = ",
            m_last_mini_batch_size,
            "\n",
            " stride_to_last_mini_batch  = ",
            m_stride_to_last_mini_batch,
            "\n",
            " current_mini_batch_idx     = ",
            m_current_mini_batch_idx,
            "\n",
            " num_iterations_per_epoch   = ",
            m_num_iterations_per_epoch,
            "\n",
            " initialized                = ",
            std::to_string(m_initialized),
            "\n",
            " total num. samples         = ",
            m_total_samples,
            "\n",
            " num. samples processed     = ",
            m_num_samples_processed,
            "\n");
}

bool dataset::update()
{
  m_current_mini_batch_idx++;
  m_current_pos = get_next_position();

  if (m_current_mini_batch_idx == m_num_iterations_per_epoch) {
    // for working with 1B jag samples, we may not process all the data
    if (m_current_pos < get_num_data()) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " :: dataset update error: the epoch is complete," +
        " but not all of the data has been used -- current pos = " +
        std::to_string(m_current_pos) + " and there are " +
        std::to_string(get_num_data()) + " indices" +
        " : iteration=" + std::to_string(m_current_mini_batch_idx) + "C of" +
        std::to_string(m_num_iterations_per_epoch) + "+" +
        " index stride=" + std::to_string(m_stride_to_next_mini_batch) + "/" +
        std::to_string(m_stride_to_last_mini_batch));
    }

    set_initial_position();
    return true;
  }
  else {
    return false;
  }
}
} // namespace lbann

#define LBANN_SKIP_CEREAL_REGISTRATION
#define LBANN_CLASS_NAME dataset
#include <lbann/macros/register_class_with_cereal.hpp>
