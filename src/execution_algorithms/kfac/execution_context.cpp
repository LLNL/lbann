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

#include "lbann/execution_algorithms/kfac/execution_context.hpp"

namespace lbann {
namespace kfac {

// =============================================
// Life cycle
// =============================================

ExecutionContext::ExecutionContext(size_t mini_batch_size)
  : m_sgd_execution_context(execution_mode::training, mini_batch_size)
{}

std::unique_ptr<execution_context> ExecutionContext::get_new() const
{
    return std::make_unique<ExecutionContext>(0UL);
}

// =============================================
// Accessors
// =============================================

std::string ExecutionContext::get_type() const
{
  return "KFAC";
}

std::string ExecutionContext::get_state_string() const noexcept
{
  return build_string(this->get_type(), ".step.", m_sgd_execution_context.get_step());
}

// =============================================
// Checkpointing and serialization
// =============================================

void ExecutionContext::save_to_checkpoint_shared(persist& p)
{
  LBANN_ERROR("TODO: Not yet implemented.");
}
void ExecutionContext::load_from_checkpoint_shared(persist& p)
{
  LBANN_ERROR("TODO: Not yet implemented.");
}
void ExecutionContext::save_to_checkpoint_distributed(persist& p)
{
  LBANN_ERROR("TODO: Not yet implemented.");
}
void ExecutionContext::load_from_checkpoint_distributed(persist& p)
{
  LBANN_ERROR("TODO: Not yet implemented.");
}

} // namespace kfac
} // namespace lbann
