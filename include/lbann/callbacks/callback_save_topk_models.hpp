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
// lbann_callback_save_topk_models .hpp .cpp - Callback to save top k models
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_CALLBACKS_CALLBACK_SAVE_TOPK_MODELS_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_SAVE_TOPK_MODELS_HPP_INCLUDED

#include "lbann/callbacks/callback_save_model.hpp"

namespace lbann {

/** Save_topk_models for (e.g., inference and other analysis).
   * @param dir directory to save model
   * @param k number of models to save, should be less than number of trainers
   * @param metric_namei, evaluation metric
   * @ordering for the topk, descending order is default
 */
class lbann_callback_save_topk_models : public lbann_callback_save_model {
 public:
  lbann_callback_save_topk_models(std::string dir, int k, std::string metric_name, bool ascending_ordering=false) : 
  lbann_callback_save_model(dir,true), m_k(k),m_metric_name(metric_name),m_ascending_ordering(ascending_ordering) {}
  lbann_callback_save_topk_models(const lbann_callback_save_topk_models&) = default;
  lbann_callback_save_topk_models& operator=(const lbann_callback_save_topk_models&) = default;
  lbann_callback_save_topk_models* copy() const override { return new lbann_callback_save_topk_models(*this); }
  void on_test_end(model *m) override;
  std::string name() const override { return "save_topk_models"; }

 private:
  bool compute_stats(model *m);
  int m_k ;  
  std::string m_metric_name; 
  bool m_ascending_ordering; 

};

}  // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_SAVE_TOPK_MODELS_HPP_INCLUDED
