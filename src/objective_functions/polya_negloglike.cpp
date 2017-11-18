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

#include "lbann/objective_functions/polya_negloglike.hpp"
#include "lbann/layers/activations/softmax.hpp"
#include <typeinfo>
#include <typeindex>
#include <limits>

namespace lbann {

namespace objective_functions {

polya_negloglike::polya_negloglike(bool categorical_ground_truth)
  : objective_function(),
    m_categorical_ground_truth(categorical_ground_truth) {}

void polya_negloglike::setup(const Layer& prev_layer) {

  /*
  // Activate softmax-cross-entropy shortcut if possible
  if(m_categorical_ground_truth) {
    const std::type_info& prev_layer_type = typeid(prev_layer);
    const std::type_info& data_parallel_softmax_type
      = typeid(softmax_layer<data_layout::DATA_PARALLEL>);
    const std::type_info& model_parallel_softmax_type
      = typeid(softmax_layer<data_layout::MODEL_PARALLEL>);
    if((std::type_index(prev_layer_type)
        == std::type_index(data_parallel_softmax_type))
       || (std::type_index(prev_layer_type)
           == std::type_index(model_parallel_softmax_type))) {
      m_shortcut_softmax_layer = &prev_layer;
    }
  }
  */

}

void polya_negloglike::compute_value(const AbsDistMat& predictions,
                                  const AbsDistMat& ground_truth) {

  // Get local matrices and matrix parameters
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  const int width = predictions.Width();
  const int local_height = predictions_local.Height();
  const int local_width = predictions_local.Width();

  AbsDistMat *counts;
  AbsDistMat *alphaSums;
  AbsDistMat *lgammaAlphaSums;
  AbsDistMat *lgammaAlphaLevelCountSums;
  
  if (predictions.DistData().colDist == El::MC) {  
    //model parallel
    counts = new StarMRMat(ground_truth.Grid());
    alphaSums = new StarMRMat(predictions.Grid());
    lgammaAlphaSums = new StarMRMat(predictions.Grid());
    lgammaAlphaLevelCountSums = new StarMRMat(predictions.Grid());
  } else {
    //data parallel 
    counts = new StarVCMat(ground_truth.Grid());
    alphaSums = new StarVCMat(predictions.Grid());
    lgammaAlphaSums = new StarVCMat(predictions.Grid());
    lgammaAlphaLevelCountSums = new StarVCMat(predictions.Grid());
  } 
  counts->Resize(1, width); 
  alphaSums->Resize(1, width);
  lgammaAlphaSums->Resize(1, width);
  lgammaAlphaLevelCountSums->Resize(1, width); 

  Mat& local_counts = counts->Matrix();
  Mat& local_alphaSums = alphaSums->Matrix();
  Mat& local_lgammaAlphaSums = lgammaAlphaSums->Matrix();
  Mat& local_lgammaAlphaLevelCountSums = lgammaAlphaLevelCountSums->Matrix();

  for (int col = 0;  col < local_width; ++col) {
    double count = 0;
    double alphaSum = 0;
    double lgammaAlphaSum = 0;
    double lgammaAlphaLevelCountSum = 0;
    for (int row = 0; row < local_height; ++row) {
      count += ground_truth_local(row, col);
      alphaSum += predictions_local(row, col);
      lgammaAlphaSum += std::lgamma(predictions_local(row, col));
      lgammaAlphaLevelCountSum += std::lgamma(predictions_local(row, col) + ground_truth_local(row, col));
    } 
    local_counts(0, col) = count; 
    local_alphaSums(0, col) = alphaSum;
    local_lgammaAlphaSums(0, col) = lgammaAlphaSum;
    local_lgammaAlphaLevelCountSums(0, col) = lgammaAlphaLevelCountSum;
  }
  El::AllReduce(*counts, counts->RedundantComm(), El::mpi::SUM); 
  El::AllReduce(*alphaSums, alphaSums->RedundantComm(), El::mpi::SUM);
  El::AllReduce(*lgammaAlphaSums, lgammaAlphaSums->RedundantComm(), El::mpi::SUM); 
  El::AllReduce(*lgammaAlphaLevelCountSums, lgammaAlphaLevelCountSums->RedundantComm(), El::mpi::SUM); 
 
  // Compute total negative log-likelihood of Polya distribution across mini-batch.
  double sum_polya_negloglike = 0;
  for (int i = 0; i < local_width; i++) {
    sum_polya_negloglike +=
      -std::lgamma(local_alphaSums(0, i)) + std::lgamma(local_counts(0, i) + local_alphaSums(0, i)) - local_lgammaAlphaLevelCountSums(0, i) + local_lgammaAlphaSums(0, i);
  }

  double mean_polya_negloglike = sum_polya_negloglike / width;
  mean_polya_negloglike = El::mpi::AllReduce(mean_polya_negloglike,
                                          predictions.DistComm());

  // Update objective function value
  add_to_value(mean_polya_negloglike);

}

double digamma(double x) {
  double result = 0.0;
  double z = x;
  while (z < 8.0) {
    result -= 1.0/z;
    z += 1.0;
  }
  double s = 1.0/z;
  double s2 = s*s;
  double s4 = s2*s2;
  double s6 = s2*s4;
  double s8 = s4*s4;
  result += std::log(z) - s/2.0 - s2/12.0 + s4/120.0 - s6/252.0 + s8/240.0 - (5.0/660.0)*s4*s6 +(691.0/32760.0)*s6*s6 - s6*s8/12.0;
  return result;
}

void polya_negloglike::compute_gradient(const AbsDistMat& predictions,
                                     const AbsDistMat& ground_truth,
                                     AbsDistMat& gradient) {

  /*
  // Apply softmax-cross-entropy shortcut if activated
  if(m_shortcut_softmax_layer != nullptr) {
    El::LockedView(gradient, ground_truth);
    return;
  }
  */

  // Get local matrices
  const Mat& predictions_local = predictions.LockedMatrix();
  const Mat& ground_truth_local = ground_truth.LockedMatrix();
  const int width = predictions.Width();
  const int local_height = predictions_local.Height();
  const int local_width = predictions_local.Width();
  Mat& gradient_local = gradient.Matrix();

  AbsDistMat *counts;
  AbsDistMat *alphaSums;

  if (predictions.DistData().colDist == El::MC) {  
    //model parallel
    counts = new StarMRMat(ground_truth.Grid());
    alphaSums = new StarMRMat(predictions.Grid());
  } else {
    //data parallel 
    counts = new StarVCMat(ground_truth.Grid());
    alphaSums = new StarVCMat(predictions.Grid());
  } 
  counts->Resize(1, width); 
  alphaSums->Resize(1, width);

  Mat& local_counts = counts->Matrix();
  Mat& local_alphaSums = alphaSums->Matrix();

  for (int col = 0;  col < local_width; ++col) {
    double count = 0;
    double alphaSum = 0;
     for (int row = 0; row < local_height; ++row) {
      count += ground_truth_local(row, col);
      alphaSum += predictions_local(row, col);
     } 
    local_counts(0, col) = count; 
    local_alphaSums(0, col) = alphaSum;
  }
  El::AllReduce(*counts, counts->RedundantComm(), El::mpi::SUM); 
  El::AllReduce(*alphaSums, alphaSums->RedundantComm(), El::mpi::SUM);

  // Compute gradient
  El::IndexDependentFill(gradient_local,
                         (std::function<DataType(El::Int,El::Int)>)
                         ([&predictions_local, &ground_truth_local, &local_counts, &local_alphaSums]
                          (El::Int r, El::Int c) -> DataType {                        
                           return DataType(
                             -digamma(local_alphaSums(0, c)) 
                               + digamma(local_counts(0, c) + local_alphaSums(0, c)) 
                               - digamma(ground_truth_local(r, c) + predictions_local(r, c)) 
  			       + digamma(predictions_local(r, c))
                           );
                         }));

}

}  // namespace objective_functions

}  // namespace lbann
