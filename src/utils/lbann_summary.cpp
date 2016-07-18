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

#if __HAVE_TBINF

lbann::lbann_summary::lbann_summary(std::string logdir, lbann_comm* comm)
  : comm(comm) {
  if (comm->am_world_master()) {
    sw = new TBinf::SummaryWriter(logdir);
  } else {
    sw = nullptr;
  }
}

lbann::lbann_summary::~lbann_summary() {
  flush();
  if (sw != nullptr) {
    delete sw;
  }
}

void lbann::lbann_summary::reduce_mean(std::string tag, DistMat& mat,
                                       int64_t step) {
  DataType sum = local_sum(mat);
  pending_means.emplace_back(tag, step, sum, 0.0f, mat.Height() * mat.Width());
}

void lbann::lbann_summary::reduce_min(std::string tag, DistMat& mat,
                                      int64_t step) {
  DataType local_min = El::Min(mat.Matrix());
  pending_mins.emplace_back(tag, step, local_min);
}

void lbann::lbann_summary::reduce_max(std::string tag, DistMat& mat,
                                      int64_t step) {
  DataType local_max = El::Max(mat.Matrix());
  pending_maxes.emplace_back(tag, step, local_max);
}

void lbann::lbann_summary::reduce_stdev(std::string tag, DistMat& mat,
                                        int64_t step) {
  // Compute the local sum and squared sum.
  DataType sum = 0.0f;
  DataType sqsum = 0.0f;
  Mat& local_mat = mat.Matrix();
  for (int row = 0; row < local_mat.Height(); ++row) {
    for (int col = 0; col < local_mat.Width(); ++col) {
      DataType v = local_mat.Get(row, col);
      sum += v;
      sqsum += v * v;
    }
  }
  pending_stdevs.emplace_back(tag, step, sum, sqsum, mat.Height() * mat.Width());
}

void lbann::lbann_summary::reduce_scalar(std::string tag, DataType s,
                                         int64_t step) {
  if (comm->am_model_master()) {
    pending_scalars.emplace_back(tag, step, s);
  }
}

void lbann::lbann_summary::sum_reduce_scalar(std::string tag, DataType s,
                                             int64_t step) {
  pending_sum_scalars.emplace_back(tag, step, s);
}

void lbann::lbann_summary::reduce_histogram(std::string tag, DistMat& mat,
                                            int64_t step) {
  
}

void lbann::lbann_summary::flush() {
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

void lbann::lbann_summary::flush_means() {
  for (const auto& op : pending_means) {
    if (comm->am_model_master()) {
      DataType model_sum = comm->model_reduce(op.local);
      DataType model_mean = model_sum / op.num;
      gather_scalar_summary(op.tag, model_mean, op.step);
    } else {
      comm->model_reduce(op.local, comm->get_model_master());
    }
  }
  pending_means.clear();
}

void lbann::lbann_summary::flush_mins() {
  for (const auto& op : pending_mins) {
    if (comm->am_model_master()) {
      DataType model_min = comm->model_reduce(op.local, El::mpi::MIN);
      gather_scalar_summary(op.tag, model_min, op.step);
    } else {
      comm->model_reduce(op.local, comm->get_model_master(), El::mpi::MIN);
    }
  }
  pending_mins.clear();
}

void lbann::lbann_summary::flush_maxes() {
  for (const auto& op : pending_maxes) {
    if (comm->am_model_master()) {
      DataType model_max = comm->model_reduce(op.local, El::mpi::MAX);
      gather_scalar_summary(op.tag, model_max, op.step);
    } else {
      comm->model_reduce(op.local, comm->get_model_master(), El::mpi::MAX);
    }
  }
  pending_maxes.clear();
}

void lbann::lbann_summary::flush_stdevs() {
  for (const auto& op : pending_stdevs) {
    // Compute the model sample standard deviation as:
    // sqrt[1/(n-1) (sqsum - (1/n)*sum^2)]
    // The n-1 is to use an unbiased variance estimate.
    // This unrolls the usual formulation of standard deviation some, to avoid
    // global operations when pushing the operation.
    if (comm->am_model_master()) {
      DataType model_sum = comm->model_reduce(op.local);
      DataType model_sqsum = comm->model_reduce(op.local2);
      DataType model_stdev = std::sqrt((model_sqsum -
                                        model_sum * model_sum / op.num) /
                                       (op.num - 1));
      gather_scalar_summary(op.tag, model_stdev, op.step);
    } else {
      comm->model_reduce(op.local, comm->get_model_master());
      comm->model_reduce(op.local2, comm->get_model_master());
    }
  }
  pending_stdevs.clear();
}

void lbann::lbann_summary::flush_scalars() {
  if (comm->am_model_master()) {
    for (const auto& op : pending_scalars) {
      gather_scalar_summary(op.tag, op.local, op.step);
    }
    pending_scalars.clear();
  }
}

void lbann::lbann_summary::flush_sum_scalars() {
  for (const auto& op : pending_sum_scalars) {
    if (comm->am_model_master()) {
      DataType model_sum = comm->model_reduce(op.local);
      gather_scalar_summary(op.tag, model_sum, op.step);
    } else {
      comm->model_reduce(op.local, comm->get_model_master());
    }
  }
  pending_sum_scalars.clear();
}

DataType lbann::lbann_summary::local_sum(DistMat& _mat) {
  Mat& mat = _mat.Matrix();
  // Note there are more numerically stable ways to compute a sum.
  DataType sum = 0.0;
  for (int row = 0; row < mat.Height(); ++row) {
    for (int col = 0; col < mat.Width(); ++col) {
      sum += mat.Get(row, col);
    }
  }
  return sum;
}

std::string lbann::lbann_summary::prepend_model(std::string tag, int model) {
  return "model" + std::to_string(model) + "/" + tag;
}

void lbann::lbann_summary::gather_scalar_summary(std::string tag, DataType s,
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
