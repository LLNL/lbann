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
//
// lbann_summary .hpp .cpp - Write summary statistics to Tensorboard
////////////////////////////////////////////////////////////////////////////////

#include "lbann/utils/lbann_summary.hpp"

namespace lbann {

#if __HAVE_TBINF

lbann_summary::lbann_summary(std::string logdir, lbann_comm* comm)
  : comm(comm) {
  if (comm->am_world_master()) {
    sw = new TBinf::SummaryWriter(logdir);
  } else {
    sw = nullptr;
  }
}

lbann_summary::~lbann_summary() {
  flush();
  if (sw != nullptr) {
    delete sw;
  }
}

void lbann_summary::reduce_mean(const std::string tag, const ElMat& mat,
                                int64_t step) {
  // Local sum
  DataType sum = 0.0;

  // Check distributed matrix format
  El::DistData mat_format(mat);
  if(mat_format.colDist == El::STAR && mat_format.rowDist == El::STAR) {
    // Compute local sum on master process if matrix is Star,Star
    if(comm->am_model_master()) {
      sum = local_sum(mat.LockedMatrix());
    }
  }
  else {
    // Compute local sum on all processes if matrix is in MC,MR;
    // Star,VC; or similar format
    // TODO: implement for matrices in Circ,Circ; MC,Star; or similar
    // formats
    sum = local_sum(mat.LockedMatrix());
  }

  // Add local sum to list of pending means
  pending_means.emplace_back(tag, step, sum, 0.0f, mat.Height() * mat.Width());
}

void lbann_summary::reduce_min(const std::string tag, const ElMat& mat,
                               int64_t step) {
  DataType local_min = El::Min(mat.LockedMatrix());
  pending_mins.emplace_back(tag, step, local_min);
}

void lbann_summary::reduce_max(const std::string tag, const ElMat& mat,
                                      int64_t step) {
  DataType local_max = El::Max(mat.LockedMatrix());
  pending_maxes.emplace_back(tag, step, local_max);
}

void lbann_summary::reduce_stdev(const std::string tag, const ElMat& mat,
                                 int64_t step) {
  // Local sum and squared sum
  DataType sum = 0.0;
  DataType sqsum = 0.0;

  // Check distributed matrix format
  El::DistData mat_format(mat);
  if(mat_format.colDist == El::STAR && mat_format.rowDist == El::STAR) {
    // Compute local sums on master process if matrix is Star,Star
    if(comm->am_model_master()) {
      sum = local_sum(mat.LockedMatrix());
      sqsum = local_sqsum(mat.LockedMatrix());
    }
  }
  else {
    // Compute local sums on all processes if matrix is in MC,MR;
    // Star,VC; or similar format
    // TODO: implement for matrices in Circ,Circ; MC,Star; or similar
    // formats
    sum = local_sum(mat.LockedMatrix());
    sqsum = local_sqsum(mat.LockedMatrix());
  }

  // Add local sums to list of pending means
  pending_stdevs.emplace_back(tag, step, sum, sqsum, mat.Height() * mat.Width());
}

void lbann_summary::reduce_scalar(const std::string tag, DataType s,
                                  int64_t step) {
  if (comm->am_model_master()) {
    pending_scalars.emplace_back(tag, step, s);
  }
}

void lbann_summary::sum_reduce_scalar(const std::string tag, DataType s,
                                      int64_t step) {
  pending_sum_scalars.emplace_back(tag, step, s);
}

void lbann_summary::reduce_histogram(const std::string tag, const ElMat& mat,
                                     int64_t step) {
  
}

void lbann_summary::flush() {
  flush_means();
  flush_mins();
  flush_maxes();
  flush_stdevs();
  flush_scalars();
  flush_sum_scalars();
  if (sw != nullptr) {
    sw->flush();
  }
}

void lbann_summary::flush_means() {
  std::vector<DataType> local_sums;
  for (const auto& op : pending_means) {
    local_sums.push_back(op.local);
  }
  if (comm->am_model_master()) {
    std::vector<DataType> global_sums(pending_means.size());
    comm->model_reduce(local_sums.data(), local_sums.size(),
                       global_sums.data());
    // Compute the means in-place.
    for (unsigned i = 0; i < global_sums.size(); ++i) {
      global_sums[i] /= pending_means[i].num;
    }
    gather_scalar_summary(pending_means, global_sums);
  } else {
    comm->model_reduce(local_sums.data(), local_sums.size(),
                       comm->get_model_master());
  }
  pending_means.clear();
}

void lbann_summary::flush_mins() {
  std::vector<DataType> local_mins;
  for (const auto& op : pending_mins) {
    local_mins.push_back(op.local);
  }
  if (comm->am_model_master()) {
    std::vector<DataType> global_mins(pending_mins.size());
    comm->model_reduce(local_mins.data(), local_mins.size(),
                       global_mins.data(), El::mpi::MIN);
    gather_scalar_summary(pending_mins, global_mins);
  } else {
    comm->model_reduce(local_mins.data(), local_mins.size(),
                       comm->get_model_master(), El::mpi::MIN);
  }
  pending_mins.clear();
}

void lbann_summary::flush_maxes() {
  std::vector<DataType> local_maxes;
  for (const auto& op : pending_maxes) {
    local_maxes.push_back(op.local);
  }
  if (comm->am_model_master()) {
    std::vector<DataType> global_maxes(pending_maxes.size());
    comm->model_reduce(local_maxes.data(), local_maxes.size(),
                       global_maxes.data(), El::mpi::MAX);
    gather_scalar_summary(pending_maxes, global_maxes);
  } else {
    comm->model_reduce(local_maxes.data(), local_maxes.size(),
                       comm->get_model_master(), El::mpi::MAX);
  }
  pending_maxes.clear();
}

void lbann_summary::flush_stdevs() {
  std::vector<DataType> local_sums;
  std::vector<DataType> local_sqsums;
  for (const auto& op : pending_stdevs) {
    local_sums.push_back(op.local);
    local_sqsums.push_back(op.local2);
  }
  if (comm->am_model_master()) {
    // Compute the model sample standard deviation as:
    // sqrt[1/(n-1) (sqsum - (1/n)*sum^2)]
    // The n-1 is to use an unbiased variance estimate.
    // This unrolls the usual formulation of standard deviation some, to avoid
    // global operations when pushing the operation.
    std::vector<DataType> global_sums(pending_stdevs.size());
    std::vector<DataType> global_sqsums(pending_stdevs.size());
    comm->model_reduce(local_sums.data(), local_sums.size(),
                       global_sums.data());
    comm->model_reduce(local_sqsums.data(), local_sqsums.size(),
                       global_sqsums.data());
    // Re-use the global_sums vector for the standard deviation.
    for (unsigned i = 0; i < global_sums.size(); ++i) {
      global_sums[i] = std::sqrt(
        (global_sqsums[i] -
         global_sums[i] * global_sums[i] / pending_stdevs[i].num) /
        (pending_stdevs[i].num - 1));
    }
    gather_scalar_summary(pending_stdevs, global_sums);
  } else {
    comm->model_reduce(local_sums.data(), local_sums.size(),
                       comm->get_model_master());
    comm->model_reduce(local_sqsums.data(), local_sqsums.size(),
                       comm->get_model_master());
  }
  pending_stdevs.clear();
}

void lbann_summary::flush_scalars() {
  if (comm->am_model_master()) {
    std::vector<DataType> local_scalars;
    for (const auto& op : pending_scalars) {
      local_scalars.push_back(op.local);
    }
    gather_scalar_summary(pending_scalars, local_scalars);
  }
  pending_scalars.clear();
}

void lbann_summary::flush_sum_scalars() {
  std::vector<DataType> local_sums;
  for (const auto& op : pending_sum_scalars) {
    local_sums.push_back(op.local);
  }
  if (comm->am_model_master()) {
    std::vector<DataType> global_sums(pending_sum_scalars.size());
    comm->model_reduce(local_sums.data(), local_sums.size(),
                       global_sums.data());
    gather_scalar_summary(pending_sum_scalars, global_sums);
  } else {
    comm->model_reduce(local_sums.data(), local_sums.size(),
                       comm->get_model_master());
  }
  pending_sum_scalars.clear();
}

DataType lbann_summary::local_sum(const Mat& mat) const {
  // Note there are more numerically stable ways to compute a sum.
  DataType sum = 0.0;
  const Int height = mat.Height();
  const Int width = mat.Width();
  const Int ldim = mat.LDim();
  const DataType* __restrict__ mat_buf = mat.LockedBuffer();
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      sum += mat_buf[row + col * ldim];
    }
  }
  return sum;
}

DataType lbann_summary::local_sqsum(const Mat& mat) const {
  // Note there are more numerically stable ways to compute a sum.
  DataType sqsum = 0.0;
  const Int height = mat.Height();
  const Int width = mat.Width();
  const Int ldim = mat.LDim();
  const DataType* __restrict__ mat_buf = mat.LockedBuffer();
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int pos = row + col * ldim;
      sqsum += mat_buf[pos] * mat_buf[pos];
    }
  }
  return sqsum;
}

std::string lbann_summary::prepend_model(const std::string tag,
                                         int model) const {
  return "model" + std::to_string(model) + "/" + tag;
}

void lbann_summary::gather_scalar_summary(
  const std::vector<pending_op>& ops, std::vector<DataType>& scalars) {
  if (comm->am_world_master()) {
    std::vector<DataType> data(comm->get_num_models() * scalars.size());
    comm->intermodel_gather(scalars.data(), scalars.size(), data.data());
    for (unsigned i = 0; i < data.size(); ++i) {
      int model = i / ops.size();
      unsigned ops_pos = i % ops.size();
      sw->add_scalar(prepend_model(ops[ops_pos].tag, model),
                     data[i], ops[ops_pos].step);
    }
  } else {
    comm->intermodel_gather(scalars.data(), scalars.size(),
                            comm->get_intermodel_master());
  }
}

void lbann_summary::gather_scalar_summary(const std::string tag, DataType s,
                                          int64_t step) {
  if (comm->am_world_master()) {
    std::vector<DataType> data(comm->get_num_models());
    comm->intermodel_gather(s, data);
    for (size_t model = 0; model < data.size(); ++model) {
      sw->add_scalar(prepend_model(tag, model), data[model], step);
    }
  } else {
    comm->intermodel_gather(s, comm->get_intermodel_master());
  }
}

#endif  // __HAVE_TBINF

}  // namespace lbann
