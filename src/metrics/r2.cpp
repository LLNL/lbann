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
////////////////////////////////////////////////////////////////////////////////

#include "lbann/metrics/r2.hpp"
#include "lbann/utils/statistics.hpp"

namespace lbann {

EvalType r2_metric::evaluate_compute(const AbsDistMat& prediction,
                                                     const AbsDistMat& ground_truth) {

  // Get matrix dimensions
  const int local_height = prediction.LocalHeight();
  const int local_width = prediction.LocalWidth();
  const int width = prediction.Width();

  // Get local matrices
  const Mat& prediction_local = prediction.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();

  DataType gt_mean, gt_std;
  // Entry-wise mean of ground truth
  //@todo fix stat class not to compute stdev if not needed
  entrywise_mean_and_stdev(ground_truth, gt_mean, gt_std);

  // Compute residual sum of squares ss_res
  // and sum of squares ss_tot as sum(square(ground_truth - mean(ground_truth)))
  EvalType ss_res = 0;
  EvalType ss_tot = 0;
  int nthreads = omp_get_num_threads();
  std::vector<EvalType> local_ss_res(nthreads, EvalType(0));
  std::vector<EvalType> local_ss_tot(nthreads, EvalType(0));
  LBANN_OMP_TASKLOOP_COLLAPSE2
  for(El::Int col = 0; col < local_width; ++col) {
    for(El::Int row = 0; row < local_height; ++row) {
      const EvalType true_val = ground_truth_local(row, col);
      const EvalType pred_val = prediction_local(row, col);
      const EvalType val1 = true_val - pred_val;
      const EvalType val2 = true_val - gt_mean;
      const int tid = omp_get_thread_num();
      local_ss_res[tid] += val1 * val1;
      local_ss_tot[tid] += val2 * val2;
    }
  }
  for (int i = 0; i < nthreads; ++i) {
    ss_res += local_ss_res[i];
    ss_tot += local_ss_tot[i];
  }

  EvalType res_tot[2] = {ss_res, ss_tot};  // Pack to do one allreduce.
  El::mpi::AllReduce(res_tot, 2, prediction.DistComm());
  //Keras and TF add epsilon (1e-07) to denominator to avoid inf score
  //We might actually need to do this here and other places too
  EvalType ss_tot_eps = res_tot[1] + 0.0000001;
  //Multiply by width because base class divide by mini-batch size
  return ((1-(res_tot[0]/ss_tot_eps))*width);
}

}  // namespace lbann
