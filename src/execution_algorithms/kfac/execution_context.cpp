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

ExecutionContext::ExecutionContext(
  size_t mini_batch_size,
  double damping_act,
  double damping_err,
  double damping_bn_act,
  double damping_bn_err)
  : m_sgd_execution_context(execution_mode::training, mini_batch_size),
    m_damping_act{damping_act},
    m_damping_err{damping_err},
    m_damping_bn_act{damping_bn_act},
    m_damping_bn_err{damping_bn_err}
{}

std::unique_ptr<lbann::ExecutionContext> ExecutionContext::get_new() const
{
    return std::make_unique<ExecutionContext>(0UL, 0.0, 0.0, 0.0, 0.0);
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

El::Matrix<DataType,Device>& ExecutionContext::get_workspace_matrix(
  const std::string& key,
  const size_t height,
  const size_t width) {
  if(m_workspace.find(key) == m_workspace.end()) {
    // std::ostringstream oss;
    // oss << "K-FAC workspace allocation (rank=" << m_rank
    //     << "): " << key << " (" << height << "x" << width << ")" << std::endl;
    // std::cout << oss.str();
    m_workspace.emplace(
        key, El::Matrix<DataType, Device>(height, width));
#ifdef HYDROGEN_HAVE_CUB
    m_workspace[key].SetMemoryMode(1); // Use CUB GPU memory pool if possible
#endif // HYDROGEN_HAVE_CUB
  }
  auto& ret = m_workspace[key];
  if((size_t) ret.Height() != height || (size_t) ret.Width() != width) {
    // Make sure that no kernels are using this workspace.
    El::Synchronize(El::SyncInfoFromMatrix(ret));
    ret.Resize(height, width);
  }
  return ret;
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
