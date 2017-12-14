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

#include "lbann/metrics/pearson_correlation.hpp"
#include "lbann/utils/statistics.hpp"

namespace lbann {

double pearson_correlation_metric::evaluate_compute(const AbsDistMat& prediction,
                                                    const AbsDistMat& ground_truth) {

    double corr = 0.0;

    // Compute mean and stdev
    DataType pred_mean = 0;
    DataType pred_std = 0;
    DataType true_mean = 0;
    DataType true_std = 0;
    DataType corr_mean = 0;
    DataType corr_std = 0;

    entrywise_mean_and_stdev(prediction,pred_mean, pred_std);
    entrywise_mean_and_stdev(ground_truth,true_mean, true_std);
    
    //Compute covariance 
    auto sub_pred_mean = [&](const DataType& z) {return z - pred_mean;};
    auto sub_true_mean = [&](const DataType& z) {return z - true_mean;};
     
    AbsDistMat* fsp = prediction.Construct(prediction.Grid(),
                                               prediction.Root());
    AbsDistMat* fst = ground_truth.Construct(ground_truth.Grid(),
                                               ground_truth.Root());
    
    Copy(prediction,*fsp);
    Copy(ground_truth,*fst);
    
    El::EntrywiseMap(*fsp, El::MakeFunction(sub_pred_mean));
    El::EntrywiseMap(*fst, El::MakeFunction(sub_true_mean));
    
    AbsDistMat* covariance_mat = ground_truth.Construct(ground_truth.Grid(),
                                               ground_truth.Root());

    El::Hadamard(*fsp,*fst, *covariance_mat);

    entrywise_mean_and_stdev(*covariance_mat, corr_mean, corr_std);
    //Compute correlation
    corr = corr_mean/(pred_std*true_std);

    return corr;

}

}  // namespace lbann
