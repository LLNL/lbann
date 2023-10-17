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
//
////////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/gpu_memory_usage.hpp"
#include "lbann/comm_impl.hpp"
#include "lbann/models/model.hpp"
#include "lbann/utils/gpu/helpers.hpp"
#include "lbann/utils/serialize.hpp"
#include <iomanip>
#include <sstream>

#ifdef LBANN_HAS_GPU
#include <h2/gpu/memory_utils.hpp>
#endif

#include "lbann/proto/callbacks.pb.h"

namespace {
template <typename T>
T get_mean(const std::vector<T>& v)
{
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}
template <typename T>
T get_median(const std::vector<T>& v)
{
  std::vector<T> tmp = v;
  int median_idx = tmp.size() / 2 - 1 + tmp.size() % 2;
  std::nth_element(tmp.begin(), tmp.begin() + median_idx, tmp.end());
  return tmp[median_idx];
}
template <typename T>
T get_max(const std::vector<T>& v)
{
  return *std::max_element(v.begin(), v.end());
}
template <typename T>
T get_min(const std::vector<T>& v)
{
  return *std::min_element(v.begin(), v.end());
}
} // namespace

namespace lbann {
namespace callback {

template <class Archive>
void gpu_memory_usage::serialize(Archive& ar)
{
  ar(::cereal::make_nvp("BaseCallback",
                        ::cereal::base_class<callback_base>(this)));
}

void gpu_memory_usage::write_specific_proto(lbann_data::Callback& proto) const
{
  proto.mutable_gpu_memory_usage();
}

void gpu_memory_usage::on_epoch_begin(model* m)
{
#ifdef LBANN_HAS_GPU
  auto const [available, total] = h2::gpu::mem_info();
  size_t used = total - available;
  auto comm = m->get_comm();
  if (comm->am_trainer_master()) {
    auto num_procs = comm->get_procs_per_trainer();
    std::vector<size_t> used_list(num_procs);
    comm->trainer_gather(used, used_list.data());
    double used_mean = get_mean(used_list) / 1024.0 / 1024.0 / 1024.0;
    double used_median = get_median(used_list) / 1024.0 / 1024.0 / 1024.0;
    double used_max = get_max(used_list) / 1024.0 / 1024.0 / 1024.0;
    double used_min = get_min(used_list) / 1024.0 / 1024.0 / 1024.0;
    std::cout << "Model " << m->get_comm()->get_trainer_rank()
              << " GPU memory usage statistics : " << std::setprecision(3)
              << used_mean << " GiB mean, " << std::setprecision(3)
              << used_median << " GiB median, " << std::setprecision(3)
              << used_max << " GiB max, " << std::setprecision(3) << used_min
              << " GiB min "
              << "(" << std::setprecision(3)
              << (total / 1024.0 / 1024.0 / 1024.0) << " GiB total)"
              << std::endl;
  }
  else {
    comm->trainer_gather(used, comm->get_trainer_master());
  }
#endif
}

} // namespace callback
} // namespace lbann

#define LBANN_CLASS_NAME callback::gpu_memory_usage
#define LBANN_CLASS_LIBNAME callback_gpu_memory_usage
#include <lbann/macros/register_class_with_cereal.hpp>
