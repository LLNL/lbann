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
// lbann_quantizer_bm.cpp - Benchmark's LBANN's quantizer
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/utils/lbann_quantizer.hpp"
#include "lbann/utils/lbann_timer.hpp"

/** Number of times to run the quantization. */
const int num_trials = 10;

using namespace lbann;

// Summarizers for different approaches.
lbann_summary* normal_summarizer;
lbann_summary* onebit_summarizer;
lbann_summary* thresh_summarizer;
lbann_summary* comp_thresh_summarizer;
lbann_summary* adaptive_summarizer;
lbann_summary* comp_adaptive_summarizer;

std::vector<double> test_normal(lbann_comm* comm, DistMat& mat) {
  std::vector<double> times;
  for (int trial = 0; trial < num_trials; ++trial) {
    double start = get_time();
    comm->intermodel_sum_matrix(mat);
    double tot = get_time() - start;
    times.push_back(tot);
    normal_summarizer->reduce_scalar("time", tot, trial);
  }
  return times;
}

void quantize_summary(lbann_summary* summarizer, lbann_quantizer& quantizer,
                      int trial) {  
  summarizer->reduce_scalar("bytes_sent",
                            quantizer.get_bytes_sent(),
                            trial);
  summarizer->reduce_scalar("bytes_received",
                            quantizer.get_bytes_received(),
                            trial);
  summarizer->reduce_scalar("rs_bytes_sent",
                            quantizer.get_rs_bytes_sent(),
                            trial);
  summarizer->reduce_scalar("ag_bytes_sent",
                            quantizer.get_ag_bytes_sent(),
                            trial);
  summarizer->reduce_scalar("rs_bytes_received",
                            quantizer.get_rs_bytes_received(),
                            trial);
  summarizer->reduce_scalar("ag_bytes_received",
                            quantizer.get_ag_bytes_received(),
                            trial);
  summarizer->reduce_scalar("rs_time",
                            quantizer.get_rs_time(),
                            trial);
  summarizer->reduce_scalar("ag_time",
                            quantizer.get_ag_time(),
                            trial);
  summarizer->reduce_scalar("rs_send_trans_time",
                            quantizer.get_rs_send_trans_time(),
                            trial);
  summarizer->reduce_scalar("rs_recv_buf_time",
                            quantizer.get_rs_recv_buf_time(),
                            trial);
  summarizer->reduce_scalar("rs_recv_trans_time",
                            quantizer.get_rs_recv_trans_time(),
                            trial);
  summarizer->reduce_scalar("ag_reduced_trans_time",
                            quantizer.get_ag_reduced_trans_time(),
                            trial);
  summarizer->reduce_scalar("ag_recv_buf_time",
                            quantizer.get_ag_recv_buf_time(),
                            trial);
  summarizer->reduce_scalar("ag_recv_trans_time",
                            quantizer.get_ag_recv_trans_time(),
                            trial);
  summarizer->reduce_scalar("pta_time",
                            quantizer.get_pta_time(),
                            trial);
  summarizer->reduce_scalar("pta_pos_time",
                            quantizer.get_pta_pos_time(),
                            trial);
  quantizer.reset_bytes_counters();
  quantizer.reset_time_counters();
}

std::vector<double> test_onebit(lbann_comm* comm, DistMat& mat) {
  std::vector<double> times;
  lbann_quantizer quantizer;
  Mat qerror;
  Mat im_qerror;
  // Allocate here, prevents messing with timing.
  Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  for (int trial = 0; trial < num_trials; ++trial) {
    double start = get_time();
    quantizer.intermodel_sum_quantized(comm, mat, qerror, im_qerror,
                                       false);
    double tot = get_time() - start;
    times.push_back(tot);
    onebit_summarizer->reduce_scalar("time", tot, trial);
    quantize_summary(onebit_summarizer, quantizer, trial);
  }
  return times;
}

std::vector<double> test_thresh(lbann_comm* comm, DistMat& mat,
                                float thresh) {
  std::vector<double> times;
  lbann_quantizer quantizer;
  Mat qerror;
  Mat im_qerror;
  // Allocate here, prevents messing with timing.
  Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  for (int trial = 0; trial < num_trials; ++trial) {
    double start = get_time();
    quantizer.intermodel_sum_threshold_quantized(
      comm, mat, qerror, thresh, -thresh, im_qerror, false);
    double tot = get_time() - start;
    times.push_back(tot);
    thresh_summarizer->reduce_scalar("time", tot, trial);
    quantize_summary(thresh_summarizer, quantizer, trial);
  }
  return times;
}

std::vector<double> test_comp_thresh(lbann_comm* comm, DistMat& mat,
                                     float thresh) {
  std::vector<double> times;
  lbann_quantizer quantizer;
  Mat qerror;
  Mat im_qerror;
  // Allocate here, prevents messing with timing.
  Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  for (int trial = 0; trial < num_trials; ++trial) {
    double start = get_time();
    quantizer.intermodel_sum_threshold_quantized(
      comm, mat, qerror, thresh, -thresh, im_qerror, true);
    double tot = get_time() - start;
    times.push_back(tot);
    comp_thresh_summarizer->reduce_scalar("time", tot, trial);
    quantize_summary(comp_thresh_summarizer, quantizer, trial);
  }
  return times;
}

std::vector<double> test_adaptive(lbann_comm* comm, DistMat& mat,
                                  int proportion) {
  std::vector<double> times;
  lbann_quantizer quantizer;
  Mat qerror;
  Mat im_qerror;
  // Allocate here, prevents messing with timing.
  Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  for (int trial = 0; trial < num_trials; ++trial) {
    double start = get_time();
    quantizer.intermodel_sum_adaptive_threshold_quantized(
      comm, mat, qerror, proportion, im_qerror, false);
    double tot = get_time() - start;
    times.push_back(tot);
    adaptive_summarizer->reduce_scalar("time", tot, trial);
    quantize_summary(adaptive_summarizer, quantizer, trial);
  }
  return times;
}

std::vector<double> test_comp_adaptive(lbann_comm* comm, DistMat& mat,
                                       int proportion) {
  std::vector<double> times;
  lbann_quantizer quantizer;
  Mat qerror;
  Mat im_qerror;
  // Allocate here, prevents messing with timing.
  Zeros(qerror, mat.LocalHeight(), mat.LocalWidth());
  for (int trial = 0; trial < num_trials; ++trial) {
    double start = get_time();
    quantizer.intermodel_sum_adaptive_threshold_quantized(
      comm, mat, qerror, proportion, im_qerror, true);
    double tot = get_time() - start;
    times.push_back(tot);
    comp_adaptive_summarizer->reduce_scalar("time", tot, trial);
    quantize_summary(comp_adaptive_summarizer, quantizer, trial);
  }
  return times;
}

void print_stats(const std::vector<double>& times) {
  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  double mean = sum / times.size();
  auto minmax = std::minmax_element(times.begin(), times.end());
  double sqsum = 0.0;
  for (const auto& t : times) {
    sqsum += (t - mean) * (t - mean);
  }
  double stdev = std::sqrt(sqsum / (times.size() - 1));
  std::cout << "\tMean: " << mean << std::endl;
  std::cout << "\tMin: " << *(minmax.first) << std::endl;
  std::cout << "\tMax: " << *(minmax.second) << std::endl;
  std::cout << "\tStdev: " << stdev << std::endl;
  std::cout << "\tRaw: ";
  for (const auto& t : times) {
    std::cout << t << " ";
  }
  std::cout << std::endl;
}

void test_mat(lbann_comm* comm, DistMat& mat) {
  DistMat normal_copy(mat);
  auto normal_times = test_normal(comm, normal_copy);
  if (comm->am_world_master()) {
    std::cout << "Normal (" << mat.Height() << "x" << mat.Width() << "):" <<
      std::endl;
    print_stats(normal_times);
  }
  normal_copy.Empty();
  DistMat onebit_copy(mat);
  auto onebit_times = test_onebit(comm, onebit_copy);
  if (comm->am_world_master()) {
    std::cout << "Onebit (" << mat.Height() << "x" << mat.Width() << "):" <<
      std::endl;
    print_stats(onebit_times);
  }
  onebit_copy.Empty();
  DistMat thresh_copy(mat);
  auto thresh_times = test_thresh(comm, thresh_copy, 3.875f);
  if (comm->am_world_master()) {
    std::cout << "Thresh (" << mat.Height() << "x" << mat.Width() << "):" <<
      std::endl;
    print_stats(thresh_times);
  }
  thresh_copy.Empty();
  DistMat comp_thresh_copy(mat);
  auto comp_thresh_times = test_comp_thresh(comm, comp_thresh_copy, 3.875f);
  if (comm->am_world_master()) {
    std::cout << "Compressed thresh (" << mat.Height() << "x" << mat.Width() <<
      "):" << std::endl;
    print_stats(comp_thresh_times);
  }
  comp_thresh_copy.Empty();
  DistMat adaptive_copy(mat);
  auto adaptive_times = test_adaptive(comm, adaptive_copy, 45);
  if (comm->am_world_master()) {
    std::cout << "Adaptive 45 " << "(" << mat.Height() << "x" <<
      mat.Width() << "):" << std::endl;
    print_stats(adaptive_times);
  }
  adaptive_copy.Empty();
  DistMat comp_adaptive_copy(mat);
  auto comp_adaptive_times = test_comp_adaptive(comm, comp_adaptive_copy, 45);
  if (comm->am_world_master()) {
    std::cout << "Compressed adaptive 45 " << "(" << mat.Height() <<
      "x" << mat.Width() << "):" << std::endl;
    print_stats(comp_adaptive_times);
  }
  comp_adaptive_copy.Empty();
}

int main(int argc, char** argv) {
  El::Initialize(argc, argv);
  lbann_comm* comm = new lbann_comm(2);
  normal_summarizer = new lbann_summary("qbm/normal", comm);
  onebit_summarizer = new lbann_summary("qbm/onebit", comm);
  thresh_summarizer = new lbann_summary("qbm/thresh", comm);
  comp_thresh_summarizer = new lbann_summary("qbm/comp_thresh", comm);
  adaptive_summarizer = new lbann_summary("qbm/adaptive", comm);
  comp_adaptive_summarizer = new lbann_summary("qbm/comp_adaptive", comm);

  if (comm->am_world_master()) {
    std::cout << "Models: " << comm->get_num_models() << std::endl;
    std::cout << "Procs per model: " << comm->get_procs_per_model() << std::endl;
  }
  for (int mat_size = 64; mat_size <= 32768; mat_size *= 2) {
    DistMat mat(comm->get_model_grid());
    El::Uniform(mat, mat_size, mat_size, 0.0f, 4.0f);
    test_mat(comm, mat);
  }

  delete normal_summarizer;
  delete onebit_summarizer;
  delete thresh_summarizer;
  delete comp_thresh_summarizer;
  delete adaptive_summarizer;
  delete comp_adaptive_summarizer;
  delete comm;
  El::Finalize();
}
