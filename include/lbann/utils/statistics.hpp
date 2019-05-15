////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include "lbann/base.hpp"

namespace lbann {

/** @brief Compute mean and standard deviation over matrix entries
 *  @param data    Input matrix.
 *  @param mean    Mean value (output).
 *  @param stdev   Standard deviation (output).
 */
void entrywise_mean_and_stdev(const Mat& data, DataType& mean, DataType& stdev);

/** @brief Compute mean and standard deviation over matrix entries
 *  @param data    Input matrix.
 *  @param mean    Mean value (output).
 *  @param stdev   Standard deviation (output).
 */
void entrywise_mean_and_stdev(const AbsDistMat& data, DataType& mean, DataType& stdev);

/** @brief Compute column-wise means and standard deviations
 *  @param data    Input matrix.
 *  @param means   Mean vector. Output as a row vector with same number
 *                 of columns as 'data'.
 *  @param stdevs  Standard deviation vector. Output as a row vector
 *                 with same number of columns as 'data'.
 */
void columnwise_mean_and_stdev(const Mat& data, Mat& means, Mat& stdevs);

/** @brief Compute column-wise means and standard deviations
 *  @param data    Input matrix.
 *  @param means   Mean vector. Output as a row vector with same number
 *                 of columns as 'data'.
 *  @param stdevs  Standard deviation vector. Output as a row vector
 *                 with same number of columns as 'data'.
 */
void columnwise_mean_and_stdev(const Mat& data, Mat& means, Mat& stdevs);

//  Wraps around column-wise sum and sqsum
/** @brief Compute column-wise means and standard deviations
 *  @param data    Input matrix in U,V format.
 *  @param means   Mean vector in STAR,V format. Output as a row vector
 *                 with same number of columns as 'data'.
 *  @param stdevs  Standard deviation vector in STAR,V format. Output as
 *                 a row vector with same number of columns as 'data'.
 */
void columnwise_mean_and_stdev(const AbsDistMat& data,
                               AbsDistMat& means,
                               AbsDistMat& stdevs);

/** @brief Compute column-wise sum and sqsum
 *  @param data    Input matrix in U,V format.
 *  @param sums    Sum vector in STAR,V format. Output as a row vector
 *                 with same number of columns as 'data'.
 *  @param sqsums  Sum of squared vector in STAR,V format. Output as
 *                 a row vector with same number of columns as 'data'.
 */
void columnwise_sums_and_sqsums(const AbsDistMat& data,
                               AbsDistMat& sums,
                               AbsDistMat& sqsums);

/** @brief Compute row-wise means and standard deviations
 *  @param data    Input matrix.
 *  @param means   Mean vector. Output as a column vector with same
 *                 number of rows as 'data'.
 *  @param stdevs  Standard deviation vector. Output as a column vector
 *                 with same number of rows as 'data'.
 */
void rowwise_mean_and_stdev(const Mat& data, Mat& means, Mat& stdevs);

/** @brief Compute row-wise sum and sum of squares
 *  @param data    Input matrix in U,V format.
 *  @param sums    Sum vector in U,STAR format. Output as a column
 *                 vector with same number of rows as 'data'.
 *  @param sqsums  Sum of squared in U,STAR format. Output as
 *                 a column vector with same number of rows as 'data'.
 */
void rowwise_sums_and_sqsums(const AbsDistMat& data,
                            AbsDistMat& sums,
                            AbsDistMat& sqsums);

//Wraps around rowwise_sum_and_sqsum
/** @brief Compute row-wise means and standard deviations
 *  @param data    Input matrix in U,V format.
 *  @param means   Mean vector in U,STAR format. Output as a column
 *                 vector with same number of rows as 'data'.
 *  @param stdevs  Standard deviation vector in U,STAR format. Output as
 *                 a column vector with same number of rows as 'data'.
 */
void rowwise_mean_and_stdev(const AbsDistMat& data,
                            AbsDistMat& means,
                            AbsDistMat& stdevs);

/** @brief Compute column-wise covariances
 *  @param data1   Input matrix in U,V format.
 *  @param data2   Input matrix in U,V format.
 *  @param means1  Column-wise mean vector for data1 in STAR,V format.
 *  @param means2  Column-wise mean vector for data2 in STAR,V format.
 *  @param cov     Covariance vector in STAR,V format. Output as a row
 *                 vector with same number of columns as 'data1'.
 */
void columnwise_covariance(const AbsDistMat& data1,
                           const AbsDistMat& data2,
                           const AbsDistMat& means1,
                           const AbsDistMat& means2,
                           AbsDistMat& cov);

} // end namespace
#endif // LBANN_UTILS_STATISTICS_HPP
