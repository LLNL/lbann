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

#include "lbann/comm_impl.hpp"
#include "lbann/optimizers/optimizer_impl.hpp"
#include "lbann/utils/serialize.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

void optimizer::clear_gradient()
{
  for (auto& g : m_local_gradient_contributions) {
    if (g.second->get_status() == optimizer_gradient_status::sync_started) {
      g.second->complete_sync(*m_comm);
    }
    g.second->clear();
  }
  this->get_gradient_sources().clear();
}

void optimizer::start_gradient_sync()
{
  for (auto& grad_mgr : m_local_gradient_contributions) {
    grad_mgr.second->start_sync(*m_comm);
  }
}

void optimizer::finish_gradient_sync()
{
  for (auto& grad_mgr : m_local_gradient_contributions) {
    grad_mgr.second->complete_sync(*m_comm);
  }
}

std::vector<std::reference_wrapper<El::BaseDistMatrix>>
optimizer::get_raw_gradients() {
  this->start_gradient_sync();
  this->finish_gradient_sync();

  std::vector<std::reference_wrapper<El::BaseDistMatrix>> grads;
  for (auto& grad_mgr_v : m_local_gradient_contributions) {
    auto& grad_mgr = *(grad_mgr_v.second);
    if (grad_mgr.get_status() != optimizer_gradient_status::ready) {
      LBANN_ERROR("Optimizer gradient not ready");
    }
    auto& grad = grad_mgr.global_gradient();
    grads.push_back(grad);
  }
  return grads;
}

std::string to_string(optimizer_gradient_status status)
{
  switch (status) {
  case optimizer_gradient_status::ready:
    return "ready";
  case optimizer_gradient_status::cleared:
    return "cleared";
  case optimizer_gradient_status::sync_needed:
    return "sync needed";
  case optimizer_gradient_status::sync_started:
    return "sync started";
  default:
    return "unknown";
  }
}

optimizer::optimizer() : m_comm(nullptr) {}

optimizer::optimizer(const optimizer& other)
  : m_comm(other.m_comm),
    m_gradient_sources(other.m_gradient_sources),
    m_gradient_status(other.m_gradient_status),
    m_step_time(other.m_step_time)
{
  if (m_gradient_status == optimizer_gradient_status::sync_started) {
    LBANN_ERROR("attempted to copy optimizer while a "
                "gradient sync is in progress");
  }
}

optimizer& optimizer::operator=(const optimizer& other)
{
  m_comm = other.m_comm;
  m_gradient_sources = other.m_gradient_sources;
  m_gradient_status = other.m_gradient_status;
  m_step_time = other.m_step_time;
  if (m_gradient_status == optimizer_gradient_status::sync_started) {
    LBANN_ERROR("attempted to copy optimizer while a "
                "gradient sync is in progress");
  }
  return *this;
}

template <class Archive>
void optimizer::serialize(Archive& ar)
{
  // Do not save the optimizer's step time
}

description optimizer::get_description() const
{
  description desc(get_type() + " optimizer");
  return desc;
}

El::Int optimizer::get_num_gradient_sources() const
{
  return m_gradient_sources.size();
}

void optimizer::add_gradient_source(const void* source)
{
  if (source != nullptr) {
    m_gradient_sources.insert(source);
  }
}

void optimizer::remove_gradient_source(const void* source)
{
  m_gradient_sources.erase(nullptr);
  m_gradient_sources.erase(source);
  if (get_gradient_sources().empty()) {
    start_gradient_sync();
  }
}

} // namespace lbann

// Instantiate methods
#undef PROTO
#define PROTO(T)                                                               \
  template El::AbstractDistMatrix<T>& lbann::optimizer::get_gradient_buffer(   \
    T& buf_scale,                                                              \
    T& in_scale,                                                               \
    bool sync_needed);                                                         \
  template void lbann::optimizer::add_to_gradient(                             \
    El::AbstractDistMatrix<T> const& contrib,                                  \
    T scale,                                                                   \
    bool sync_needed);                                                         \
  template void lbann::optimizer::accumulate_all_gradient_contributions(       \
    El::AbstractDistMatrix<T>& gradient);

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

#define LBANN_CLASS_NAME optimizer
#include <lbann/macros/register_class_with_cereal.hpp>
