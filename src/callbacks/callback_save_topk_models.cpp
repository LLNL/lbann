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
// lbann_callback_save_topk_models .hpp .cpp - Callback hooks to save_topk_models information
////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include "lbann/callbacks/callback_save_topk_models.hpp"

namespace lbann {
void lbann_callback_save_topk_models::on_test_end(model *m) {
  bool am_in_topk = false;
  if(m->get_comm()->am_trainer_master()) {
    am_in_topk = compute_stats(m);
  }
  m->get_comm()->trainer_broadcast(0, am_in_topk); 
  if(am_in_topk) save_model(m);
}

bool lbann_callback_save_topk_models::compute_stats(model *m) {
  lbann_comm *comm = m->get_comm();
  const int num_trainers = comm->get_num_trainers();
  std::string mode_string = "test";
  bool found_metric = false;
  EvalType score = 0;
  for (const auto& met : m->get_metrics()) {
    if (met->name() == m_metric_name) {
      found_metric = true;
      score = met->get_mean_value(m->get_execution_mode());
      break;
    }
  }
  //sanity check
  if (!found_metric) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "could not find metric \"" << m_metric_name << "\""
        << "in model \"" << m->get_name() << "\"";
    LBANN_ERROR(err.str());
  }

  if (m_k > num_trainers) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__ << " :: "
        << "k ( " << m_k << ") " 
        << " can not be greater than number of trainers (" 
        << num_trainers << ") " ;
    LBANN_ERROR(err.str());
  }

  std::vector<EvalType> score_list(comm->get_num_trainers());
  comm->all_gather<EvalType>(score, score_list,comm->get_intertrainer_comm());
  std::vector<EvalType> score_v = score_list;
  //top-k in an ascending order
  if(m_ascending_ordering) std::sort(score_v.begin(), score_v.end(),std::less<EvalType>());
  //top-k in an descending order
  else  std::sort(score_v.begin(), score_v.end(),std::greater<EvalType>());
  score_v.resize(m_k);
  
  if (comm->am_world_master()) {
    std::cout << "Top " << m_k << " " << m_metric_name << " average "
              << std::accumulate(score_v.begin(), score_v.end(), EvalType(0))/m_k << std::endl;
  } 
  for(int i =0; i < num_trainers; ++i) {
    if(std::find(score_v.begin(), score_v.end(), 
                 score_list[i]) != score_v.end()) { 
      if( i == comm->get_trainer_rank()) {
        std::cout << "Trainer [ " << comm->get_trainer_rank() << "] in top list with score " << score_list[i] << std::endl;;
          return true;
      }
    }
  } 
  return false;
}

}  // namespace lbann
