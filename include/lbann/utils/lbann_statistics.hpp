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

#ifndef LBANN_UTILS_STATISTICS_HPP
#define LBANN_UTILS_STATISTICS_HPP

#include "lbann/lbann_base.hpp"

namespace lbann {

/// Compute mean and standard deviation over matrix entries
/** @param data    Input matrix.
 *  @param means   Mean value.
 *  @param stdevs  Standard deviation.
 */
void mean_and_stdev(const Mat& data, DataType& mean, DataType& stdev);

/// Compute column-wise means and standard deviations
/** @param data    Input matrix.
 *  @param means   Mean vector. Output as a row vector with same number
 *                 of columns as 'data'.
 *  @param stdevs  Standard deviation vector. Output as a row vector
 *                 with same number of columns as 'data'.
 */
void columnwise_mean_and_stdev(const Mat& data, Mat& means, Mat& stdevs);

/// Compute column-wise means and standard deviations
/** @param data    Input matrix.
 *  @param means   Mean vector. Output as a row vector with same number
 *                 of columns as 'data'.
 *  @param stdevs  Standard deviation vector. Output as a row vector
 *                 with same number of columns as 'data'.
 */
void columnwise_mean_and_stdev(const Mat& data, Mat& means, Mat& stdevs);

/// Compute column-wise means and standard deviations
/** @param data    Input matrix in U,V format.
 *  @param means   Mean vector in STAR,V format. Output as a row vector
 *                 with same number of columns as 'data'.
 *  @param stdevs  Standard deviation vector in STAR,V format. Output as
 *                 a row vector with same number of columns as 'data'.
 */
void columnwise_mean_and_stdev(const AbsDistMat& data,
                               AbsDistMat& means,
                               AbsDistMat& stdevs);

/// Compute row-wise means and standard deviations
/** @param data    Input matrix.
 *  @param means   Mean vector. Output as a column vector with same
 *                 number of rows as 'data'.
 *  @param stdevs  Standard deviation vector. Output as a column vector
 *                 with same number of rows as 'data'.
 */
void rowwise_mean_and_stdev(const Mat& data, Mat& means, Mat& stdevs);

/// Compute row-wise means and standard deviations
/** @param data    Input matrix in U,V format.
 *  @param means   Mean vector in U,STAR format. Output as a column
 *                 vector with same number of rows as 'data'.
 *  @param stdevs  Standard deviation vector in U,STAR format. Output as
 *                 a column vector with same number of rows as 'data'.
 */
void rowwise_mean_and_stdev(const AbsDistMat& data,
                            AbsDistMat& means,
                            AbsDistMat& stdevs);

} // end namespace
#endif // LBANN_UTILS_STATISTICS_HPP
