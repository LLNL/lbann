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
#include "lbann/utils/summary_impl.hpp"

#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/exception.hpp"
#ifdef LBANN_HAS_OPENCV
#include "lbann/utils/image.hpp"
#endif // LBANN_HAS_OPENCV

namespace lbann {

#ifdef LBANN_HAS_TBINF

lbann_summary::lbann_summary(std::string logdir, lbann_comm* comm)
  : m_comm(comm)
{
  if (m_comm->am_world_master()) {
    m_sw = new TBinf::SummaryWriter(logdir);
  }
  else {
    m_sw = nullptr;
  }
  m_histogram_buckets = TBinf::SummaryWriter::get_default_histogram_buckets();
}

lbann_summary::~lbann_summary()
{
  flush();
  if (m_sw != nullptr) {
    delete m_sw;
  }
}

#ifdef LBANN_HAS_OPENCV
void lbann_summary::report_image(std::string const& tag,
                                 std::string const& img_format,
                                 CPUMat const& image,
                                 std::vector<El::Int> const& dims_in,
                                 int step)
{
  std::vector<size_t> dims(dims_in.begin(), dims_in.end());

  if (static_cast<size_t>(image.Height()) != get_linear_size(dims)) {
    LBANN_ERROR("Image height \"",
                image.Height(),
                "\" does not match expected value \"",
                get_linear_size(dims),
                "\".");
  }

  auto uint8_img = get_uint8_t_image(image, dims);
  auto img_str = encode_image(uint8_img, dims, img_format);
  m_sw->add_image(tag, img_str, dims, step);
}
#endif // LBANN_HAS_OPENCV

void lbann_summary::flush()
{
  flush_means();
  flush_mins();
  flush_maxes();
  flush_stdevs();
  flush_scalars();
  flush_sum_scalars();
  flush_scalar_alls();
  flush_histograms();
  if (m_sw != nullptr) {
    m_sw->flush();
  }
}

void lbann_summary::flush_means()
{
  if (m_pending_means.empty()) {
    return;
  }
  std::vector<float> local_sums;
  for (const auto& op : m_pending_means) {
    local_sums.push_back(op.local);
  }
  if (m_comm->am_trainer_master()) {
    std::vector<float> global_sums(local_sums.size());
    m_comm->trainer_reduce(local_sums.data(),
                           local_sums.size(),
                           global_sums.data(),
                           El::mpi::SUM);
    // Compute the means in-place.
    for (unsigned i = 0; i < global_sums.size(); ++i) {
      global_sums[i] /= m_pending_means[i].num;
    }
    gather_scalar_summary(m_pending_means, global_sums);
  }
  else {
    m_comm->trainer_reduce(local_sums.data(),
                           local_sums.size(),
                           m_comm->get_trainer_master(),
                           El::mpi::SUM);
  }
  m_pending_means.clear();
}

void lbann_summary::flush_mins()
{
  if (m_pending_mins.empty()) {
    return;
  }
  std::vector<float> local_mins;
  for (const auto& op : m_pending_mins) {
    local_mins.push_back(op.local);
  }
  if (m_comm->am_trainer_master()) {
    std::vector<float> global_mins(local_mins.size());
    m_comm->trainer_reduce(local_mins.data(),
                           local_mins.size(),
                           global_mins.data(),
                           El::mpi::MIN);
    gather_scalar_summary(m_pending_mins, global_mins);
  }
  else {
    m_comm->trainer_reduce(local_mins.data(),
                           local_mins.size(),
                           m_comm->get_trainer_master(),
                           El::mpi::MIN);
  }
  m_pending_mins.clear();
}

void lbann_summary::flush_maxes()
{
  if (m_pending_maxes.empty()) {
    return;
  }
  std::vector<float> local_maxes;
  for (const auto& op : m_pending_maxes) {
    local_maxes.push_back(op.local);
  }
  if (m_comm->am_trainer_master()) {
    std::vector<float> global_maxes(local_maxes.size());
    m_comm->trainer_reduce(local_maxes.data(),
                           local_maxes.size(),
                           global_maxes.data(),
                           El::mpi::MAX);
    gather_scalar_summary(m_pending_maxes, global_maxes);
  }
  else {
    m_comm->trainer_reduce(local_maxes.data(),
                           local_maxes.size(),
                           m_comm->get_trainer_master(),
                           El::mpi::MAX);
  }
  m_pending_maxes.clear();
}

void lbann_summary::flush_stdevs()
{
  if (m_pending_stdevs.empty()) {
    return;
  }
  std::vector<float> local_sums;
  std::vector<float> local_sqsums;
  for (const auto& op : m_pending_stdevs) {
    local_sums.push_back(op.local);
    local_sqsums.push_back(op.local2);
  }
  if (m_comm->am_trainer_master()) {
    // Compute the model sample standard deviation as:
    // sqrt[1/(n-1) (sqsum - (1/n)*sum^2)]
    // The n-1 is to use an unbiased variance estimate.
    // This unrolls the usual formulation of standard deviation some, to avoid
    // global operations when pushing the operation.
    std::vector<float> global_sums(local_sums.size());
    std::vector<float> global_sqsums(local_sqsums.size());
    m_comm->trainer_reduce(local_sums.data(),
                           local_sums.size(),
                           global_sums.data(),
                           El::mpi::SUM);
    m_comm->trainer_reduce(local_sqsums.data(),
                           local_sqsums.size(),
                           global_sqsums.data(),
                           El::mpi::SUM);
    // Re-use the global_sums vector for the standard deviation.
    for (unsigned i = 0; i < global_sums.size(); ++i) {
      global_sums[i] =
        El::Sqrt((global_sqsums[i] -
                  global_sums[i] * global_sums[i] / m_pending_stdevs[i].num) /
                 (m_pending_stdevs[i].num - 1));
    }
    gather_scalar_summary(m_pending_stdevs, global_sums);
  }
  else {
    m_comm->trainer_reduce(local_sums.data(),
                           local_sums.size(),
                           m_comm->get_trainer_master(),
                           El::mpi::SUM);
    m_comm->trainer_reduce(local_sqsums.data(),
                           local_sqsums.size(),
                           m_comm->get_trainer_master(),
                           El::mpi::SUM);
  }
  m_pending_stdevs.clear();
}

void lbann_summary::flush_scalars()
{
  if (m_pending_scalars.empty()) {
    return;
  }
  if (m_comm->am_trainer_master()) {
    std::vector<float> local_scalars;
    for (const auto& op : m_pending_scalars) {
      local_scalars.push_back(op.local);
    }
    gather_scalar_summary(m_pending_scalars, local_scalars);
  }
  m_pending_scalars.clear();
}

void lbann_summary::flush_sum_scalars()
{
  if (m_pending_sum_scalars.empty()) {
    return;
  }
  std::vector<float> local_sums;
  for (const auto& op : m_pending_sum_scalars) {
    local_sums.push_back(op.local);
  }
  if (m_comm->am_trainer_master()) {
    std::vector<float> global_sums(local_sums.size());
    m_comm->trainer_reduce(local_sums.data(),
                           local_sums.size(),
                           global_sums.data(),
                           El::mpi::SUM);
    gather_scalar_summary(m_pending_sum_scalars, global_sums);
  }
  else {
    m_comm->trainer_reduce(local_sums.data(),
                           local_sums.size(),
                           m_comm->get_trainer_master(),
                           El::mpi::SUM);
  }
  m_pending_sum_scalars.clear();
}

void lbann_summary::flush_scalar_alls()
{
  if (m_pending_scalar_alls.empty()) {
    return;
  }
  // Gather from every process to world master.
  std::vector<float> local_scalars;
  for (const auto& op : m_pending_scalar_alls) {
    local_scalars.push_back(op.local);
  }
  if (m_comm->am_world_master()) {
    std::vector<float> scalars(m_comm->get_procs_in_world() *
                               local_scalars.size());
    m_comm->gather(local_scalars.data(),
                   local_scalars.size(),
                   scalars.data(),
                   m_comm->get_world_comm());
    for (size_t i = 0; i < scalars.size(); ++i) {
      int rank = i / local_scalars.size();
      int model = rank / m_comm->get_procs_per_trainer();
      int pos = i % local_scalars.size();
      m_sw->add_scalar(prepend_model("rank" + std::to_string(rank) + "/" +
                                       m_pending_scalar_alls[pos].tag,
                                     model),
                       scalars[i],
                       m_pending_scalar_alls[pos].step);
    }
  }
  else {
    m_comm->gather(local_scalars.data(),
                   local_scalars.size(),
                   m_comm->get_world_master(),
                   m_comm->get_world_comm());
  }
  m_pending_scalar_alls.clear();
}

void lbann_summary::flush_histograms()
{
  if (m_pending_histograms.empty()) {
    return;
  }
  std::vector<float> local_mins;
  std::vector<float> local_maxes;
  std::vector<float> local_sums;
  std::vector<float> local_sqsums;
  std::vector<float> buckets;
  for (const auto& op : m_pending_histograms) {
    local_mins.push_back(op.min);
    local_maxes.push_back(op.max);
    local_sums.push_back(op.sum);
    local_sqsums.push_back(op.sqsum);
    buckets.insert(buckets.end(), op.buckets.begin(), op.buckets.end());
  }
  if (m_comm->am_trainer_master()) {
    std::vector<float> model_mins(local_mins.size());
    std::vector<float> model_maxes(local_maxes.size());
    std::vector<float> model_sums(local_sums.size());
    std::vector<float> model_sqsums(local_sqsums.size());
    std::vector<float> model_buckets(buckets.size());
    m_comm->trainer_reduce(local_mins.data(),
                           local_mins.size(),
                           model_mins.data(),
                           El::mpi::MIN);
    m_comm->trainer_reduce(local_maxes.data(),
                           local_maxes.size(),
                           model_maxes.data(),
                           El::mpi::MAX);
    m_comm->trainer_reduce(local_sums.data(),
                           model_sums.size(),
                           model_sums.data());
    m_comm->trainer_reduce(local_sqsums.data(),
                           local_sqsums.size(),
                           model_sqsums.data());
    m_comm->trainer_reduce(buckets.data(),
                           buckets.size(),
                           model_buckets.data());
    // Gather to the world master for writing out.
    if (m_comm->am_world_master()) {
      std::vector<float> global_mins(m_comm->get_num_trainers() *
                                     model_mins.size());
      std::vector<float> global_maxes(m_comm->get_num_trainers() *
                                      model_maxes.size());
      std::vector<float> global_sums(m_comm->get_num_trainers() *
                                     model_sums.size());
      std::vector<float> global_sqsums(m_comm->get_num_trainers() *
                                       model_sqsums.size());
      std::vector<float> global_buckets(m_comm->get_num_trainers() *
                                        model_buckets.size());
      m_comm->intertrainer_gather(model_mins.data(),
                                  model_mins.size(),
                                  global_mins.data());
      m_comm->intertrainer_gather(model_maxes.data(),
                                  model_maxes.size(),
                                  global_maxes.data());
      m_comm->intertrainer_gather(model_sums.data(),
                                  model_sums.size(),
                                  global_sums.data());
      m_comm->intertrainer_gather(model_sqsums.data(),
                                  model_sqsums.size(),
                                  global_sqsums.data());
      m_comm->intertrainer_gather(model_buckets.data(),
                                  model_buckets.size(),
                                  global_buckets.data());
      for (unsigned i = 0; i < global_mins.size(); ++i) {
        int model = i / m_pending_histograms.size();
        unsigned ops_pos = i % m_pending_histograms.size();
        std::vector<float> tmp_buckets(
          global_buckets.begin() + i * m_histogram_buckets.size(),
          global_buckets.begin() + (i + 1) * m_histogram_buckets.size());
        m_sw->add_histogram(
          prepend_model(m_pending_histograms[ops_pos].tag, model),
          tmp_buckets,
          global_mins[i],
          global_maxes[i],
          m_pending_histograms[ops_pos].num,
          global_sums[i],
          global_sqsums[i],
          m_pending_histograms[ops_pos].step);
      }
    }
    else {
      m_comm->intertrainer_gather(model_mins.data(),
                                  model_mins.size(),
                                  m_comm->get_intertrainer_master());
      m_comm->intertrainer_gather(model_maxes.data(),
                                  model_maxes.size(),
                                  m_comm->get_intertrainer_master());
      m_comm->intertrainer_gather(model_sums.data(),
                                  model_sums.size(),
                                  m_comm->get_intertrainer_master());
      m_comm->intertrainer_gather(model_sqsums.data(),
                                  model_sqsums.size(),
                                  m_comm->get_intertrainer_master());
      m_comm->intertrainer_gather(model_buckets.data(),
                                  model_buckets.size(),
                                  m_comm->get_intertrainer_master());
    }
  }
  else {
    m_comm->trainer_reduce(local_mins.data(),
                           local_mins.size(),
                           m_comm->get_trainer_master(),
                           El::mpi::MIN);
    m_comm->trainer_reduce(local_maxes.data(),
                           local_maxes.size(),
                           m_comm->get_trainer_master(),
                           El::mpi::MAX);
    m_comm->trainer_reduce(local_sums.data(),
                           local_sums.size(),
                           m_comm->get_trainer_master(),
                           El::mpi::SUM);
    m_comm->trainer_reduce(local_sqsums.data(),
                           local_sqsums.size(),
                           m_comm->get_trainer_master(),
                           El::mpi::SUM);
    m_comm->trainer_reduce(buckets.data(),
                           buckets.size(),
                           m_comm->get_trainer_master(),
                           El::mpi::SUM);
  }
  m_pending_histograms.clear();
}

std::string lbann_summary::prepend_model(const std::string tag, int model) const
{
  return "model" + std::to_string(model) + "/" + tag;
}

void lbann_summary::gather_scalar_summary(const std::vector<pending_op>& ops,
                                          std::vector<float>& scalars)
{
  if (m_comm->am_world_master()) {
    std::vector<float> data(m_comm->get_num_trainers() * scalars.size());
    m_comm->intertrainer_gather(scalars.data(), scalars.size(), data.data());
    for (unsigned i = 0; i < data.size(); ++i) {
      int model = i / ops.size();
      unsigned ops_pos = i % ops.size();
      m_sw->add_scalar(prepend_model(ops[ops_pos].tag, model),
                       data[i],
                       ops[ops_pos].step);
    }
  }
  else {
    m_comm->intertrainer_gather(scalars.data(),
                                scalars.size(),
                                m_comm->get_intertrainer_master());
  }
}

void lbann_summary::gather_scalar_summary(const std::string tag,
                                          float s,
                                          int step)
{
  if (m_comm->am_world_master()) {
    std::vector<float> data(m_comm->get_num_trainers());
    m_comm->intertrainer_gather(s, data);
    for (size_t model = 0; model < data.size(); ++model) {
      m_sw->add_scalar(prepend_model(tag, model), data[model], step);
    }
  }
  else {
    m_comm->intertrainer_gather(s, m_comm->get_intertrainer_master());
  }
}

#endif // LBANN_HAS_TBINF

} // namespace lbann
