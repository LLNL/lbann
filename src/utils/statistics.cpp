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

#include "lbann/utils/statistics.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

void entrywise_mean_and_stdev(const Mat& data,
                              DataType& mean,
                              DataType& stdev) {
  // Note: This routine is primarily called in an OpenMP-parallelized
  // loop in data_readers/image_preprocessor.hpp. If a more
  // significant use-case is found, it may be worthwhile parallelizing
  // the loop.

  // Matrix dimensions
  const El::Int height = data.Height();
  const El::Int width = data.Width();
  const El::Int size = height * width;

  // Compute sums over matrix entries
  const DataType shift = data(0, 0);
  DataType shifted_sum = 0;
  DataType shifted_sqsum = 0;
  for(El::Int col = 0; col < width; ++col) {
    for(El::Int row = 0; row < height; ++row) {
      const DataType shifted_val = data(row, col) - shift;
      shifted_sum += shifted_val;
      shifted_sqsum += shifted_val * shifted_val;
    }
  }

  // Compute mean and standard deviation
  const DataType shifted_mean = shifted_sum / size;
  const DataType shifted_sqmean = shifted_sqsum / size;
  mean = shifted_mean + shift;
  const DataType var = std::max(shifted_sqmean - shifted_mean * shifted_mean,
                                DataType(0));
  stdev = std::sqrt(var);

}

void entrywise_mean_and_stdev(const AbsDistMat& data,
                              DataType& mean,
                              DataType& stdev) {

  // Matrix dimensions
  const El::Int size = data.Height() * data.Width();
  const El::Int local_height = data.LocalHeight();
  const El::Int local_width = data.LocalWidth();

  // Local matrices
  const Mat& local_data = data.LockedMatrix();

  // Compute sums over matrix entries
  DataType sum = 0;
  DataType sqsum = 0;
  LBANN_OMP_PARALLEL_FOR_ARGS(reduction(+:sum,sqsum) collapse(2))
  for(El::Int col = 0; col < local_width; ++col) {
    for(El::Int row = 0; row < local_height; ++row) {
      const DataType val = local_data(row, col);
      sum += val;
      sqsum += val * val;
    }
  }
  DataType sum_sqsum[2] = {sum, sqsum};  // Pack to do one allreduce.
  El::mpi::AllReduce(sum_sqsum, 2, data.DistComm(),
                     El::SyncInfo<El::Device::CPU>{});

  // Compute mean and standard deviation
  mean = sum_sqsum[0] / size;
  const DataType var = std::max(sum_sqsum[1] / size - mean * mean, DataType(0));
  stdev = std::sqrt(var);

}

void columnwise_mean_and_stdev(const Mat& data,
                               Mat& means,
                               Mat& stdevs) {

  // Matrix dimensions
  const El::Int height = data.Height();
  const El::Int width = data.Width();

  // Initialize outputs
  means.Resize(1, width);
  stdevs.Resize(1, width);

  // Compute mean and standard deviation of each matrix column
  LBANN_OMP_PARALLEL_FOR
  for(El::Int col = 0; col < width; ++col) {
    const DataType shift = data(0, col);
    DataType shifted_sum = 0;
    DataType shifted_sqsum = 0;
    for(El::Int row = 0; row < height; ++row) {
      const DataType shifted_val = data(row, col) - shift;
      shifted_sum += shifted_val;
      shifted_sqsum += shifted_val * shifted_val;
    }
    const DataType shifted_mean = shifted_sum / height;
    const DataType shifted_sqmean = shifted_sqsum / height;
    const DataType mean = shifted_mean + shift;
    const DataType var = std::max(shifted_sqmean - shifted_mean * shifted_mean,
                                  DataType(0));
    const DataType stdev = std::sqrt(var);
    means(0, col) = mean;
    stdevs(0, col) = stdev;
  }

}

/// @todo Numerically stable implementation
void columnwise_sums_and_sqsums(const AbsDistMat& data,
                               AbsDistMat& sums,
                               AbsDistMat& sqsums) {

#ifdef LBANN_DEBUG
  El::DistData data_dist(data), sum_dist(sums), sqsum_dist(sqsums);
  if(sum_dist.colDist != El::STAR
      || sum_dist.rowDist != data_dist.rowDist
      || sqsum_dist.colDist != El::STAR
      || sqsum_dist.rowDist != data_dist.rowDist) {
    throw lbann_exception("columnwise_sum_and_sqsum: invalid matrix format");
  }
#endif // #ifdef LBANN_DEBUG

  // Matrix dimensions
  const El::Int width = data.Width();
  const El::Int local_height = data.LocalHeight();
  const El::Int local_width = data.LocalWidth();

  // Initialize outputs
  sums.Resize(1, width);
  sqsums.Resize(1, width);

  // Local matrices
  const Mat& local_data = data.LockedMatrix();
  Mat& local_sum = sums.Matrix();
  Mat& local_sqsum = sqsums.Matrix();

  // Compute sum and sum of squares of each matrix column
  LBANN_OMP_PARALLEL_FOR
  for(El::Int col = 0; col < local_width; ++col) {
    DataType sum_val = 0;
    DataType sqsum_val = 0;
    for(El::Int row = 0; row < local_height; ++row) {
      const DataType val = local_data(row, col);
      sum_val += val;
      sqsum_val += val * val;
    }
    local_sum(0, col) = sum_val;
    local_sqsum(0, col) = sqsum_val;
  }

  // Allreduce sums and sums of squares
  AllReduce(sums, sums.RedundantComm(), El::mpi::SUM);
  AllReduce(sqsums, sqsums.RedundantComm(), El::mpi::SUM);

}
/// @todo Numerically stable implementation
void columnwise_mean_and_stdev(const AbsDistMat& data,
                               AbsDistMat& means,
                               AbsDistMat& stdevs) {

#ifdef LBANN_DEBUG
  El::DistData data_dist(data), means_dist(means), stdevs_dist(stdevs);
  if(means_dist.colDist != El::STAR
      || means_dist.rowDist != data_dist.rowDist
      || stdevs_dist.colDist != El::STAR
      || stdevs_dist.rowDist != data_dist.rowDist) {
    throw lbann_exception("columnwise_mean_and_stdev: invalid matrix format");
  }
#endif // #ifdef LBANN_DEBUG

  // Matrix dimensions
  const El::Int height = data.Height();
  const El::Int local_width = data.LocalWidth();

  columnwise_sums_and_sqsums(data, means, stdevs);
  // Local matrices
  Mat& local_means = means.Matrix();
  Mat& local_stdevs = stdevs.Matrix();

  for(El::Int col = 0; col < local_width; ++col) {
    const DataType mean = local_means(0, col) / height;
    const DataType sqmean = local_stdevs(0, col) / height;
    const DataType var = std::max(sqmean - mean * mean, DataType(0));
    const DataType stdev = std::sqrt(var);
    local_means(0, col) = mean;
    local_stdevs(0, col) = stdev;
  }

}

void rowwise_mean_and_stdev(const Mat& data,
                            Mat& means,
                            Mat& stdevs) {

  // Matrix dimensions
  const El::Int height = data.Height();
  const El::Int width = data.Width();

  // Initialize outputs
  means.Resize(height, 1);
  stdevs.Resize(height, 1);

  // Iterate through row blocks
  const El::Int block_size = 16;
  LBANN_OMP_PARALLEL_FOR
  for(El::Int row_start = 0; row_start < height; row_start += block_size) {
    const El::Int row_end = std::min(row_start + block_size, height);

    // Initialize shift and sums for each row
    auto *shifts = new DataType[block_size];
    for(El::Int row = row_start; row < row_end; ++row) {
      means(row, 0) = 0;
      stdevs(row, 0) = 0;
      shifts[row-row_start] = data(row, 0);
    }

    // Iterate through blocks in row block
    for(El::Int col_start = 0; col_start < width; col_start += block_size) {
      const El::Int col_end = std::min(col_start + block_size, width);

      // Compute sums by iterating through block entries
      for(El::Int col = col_start; col < col_end; ++col) {
        for(El::Int row = row_start; row < row_end; ++row) {
          const DataType shift = shifts[row - row_start];
          const DataType shifted_val = data(row, col) - shift;
          means(row, 0) += shifted_val;
          stdevs(row, 0) += shifted_val * shifted_val;
        }
      }

    }

    // Compute mean and standard deviation of each row
    for(El::Int row = row_start; row < row_end; ++row) {
      const DataType shifted_mean = means(row, 0) / width;
      const DataType shifted_sqmean = stdevs(row, 0) / width;
      const DataType mean = shifted_mean + shifts[row - row_start];
      const DataType var = std::max(shifted_sqmean - shifted_mean * shifted_mean,
                                    DataType(0));
      const DataType stdev = std::sqrt(var);
      means(row, 0) = mean;
      stdevs(row, 0) = stdev;
    }

    // Deallocate shifts
    delete[] shifts;

  }

}

/// @todo Numerically stable implementation
void rowwise_sums_and_sqsums(const AbsDistMat& data,
                            AbsDistMat& sums,
                            AbsDistMat& sqsums) {

#ifdef LBANN_DEBUG
  El::DistData data_dist(data), sum_dist(sums), sqsum_dist(sqsums);
  if(sum_dist.colDist != data_dist.colDist
      || sum_dist.rowDist != El::STAR
      || sqsum_dist.colDist != data_dist.colDist
      || sqsum_dist.rowDist != El::STAR) {
    throw lbann_exception("rowwise_sums_and_sqsums: invalid matrix format");
  }
#endif // #ifdef LBANN_DEBUG

  // Matrix dimensions
  const El::Int height = data.Height();
  const El::Int local_height = data.LocalHeight();
  const El::Int local_width = data.LocalWidth();

  // Initialize outputs
  sums.Resize(height, 1);
  sqsums.Resize(height, 1);

  // Local matrices
  const Mat& local_data = data.LockedMatrix();
  Mat& local_sum = sums.Matrix();
  Mat& local_sqsum = sqsums.Matrix();

  // Iterate through row blocks
  const El::Int block_size = 16;
  LBANN_OMP_PARALLEL_FOR
  for(El::Int row_start = 0; row_start < local_height; row_start += block_size) {
    const El::Int row_end = std::min(row_start + block_size, local_height);

    // Iterate through blocks in row block
    for(El::Int col_start = 0; col_start < local_width; col_start += block_size) {
      const El::Int col_end = std::min(col_start + block_size, local_width);

      // Compute sums by iterating through block entries
      for(El::Int col = col_start; col < col_end; ++col) {
        for(El::Int row = row_start; row < row_end; ++row) {
          const DataType val = local_data(row, col);
          local_sum(row, 0) += val;
          local_sqsum(row, 0) += val * val;
        }
      }

    }

  }

  // Allreduce sums and sums of squares
  AllReduce(sums, sums.RedundantComm(), El::mpi::SUM);
  AllReduce(sqsums, sqsums.RedundantComm(), El::mpi::SUM);

}

/// @todo Numerically stable implementation
void rowwise_mean_and_stdev(const AbsDistMat& data,
                            AbsDistMat& means,
                            AbsDistMat& stdevs) {


  const El::Int width = data.Width();
  const El::Int local_height = data.LocalHeight();

  rowwise_sums_and_sqsums(data, means, stdevs);

  // Local matrices
  Mat& local_means = means.Matrix();
  Mat& local_stdevs = stdevs.Matrix();

  // Compute mean and standard deviation of each matrix row
  LBANN_OMP_PARALLEL_FOR
  for(El::Int row = 0; row < local_height; ++row) {
    const DataType mean = local_means(row, 0) / width;
    const DataType sqmean = local_stdevs(row, 0) / width;
    const DataType var = std::max(sqmean - mean * mean, DataType(0));
    const DataType stdev = std::sqrt(var);
    local_means(row, 0) = mean;
    local_stdevs(row, 0) = stdev;
  }

}

void columnwise_covariance(const AbsDistMat& data1,
                           const AbsDistMat& data2,
                           const AbsDistMat& means1,
                           const AbsDistMat& means2,
                           AbsDistMat& covs) {

  // Check matrix formats and dimensions are valid
  El::DistData data1_dist(data1), data2_dist(data2),
    means1_dist(means1), means2_dist(means2), covs_dist(covs);
  if(data1_dist != data2_dist
     || means1_dist.colDist != El::STAR
     || means1_dist.rowDist != data1_dist.rowDist
     || means1_dist != means2_dist
     || means1_dist != covs_dist) {
    throw lbann_exception("columnwise_covariance: invalid matrix format");
  }
  if(data1.Height() != data2.Height() || data1.Width() != data2.Width()) {
    throw lbann_exception("columnwise_covariance: data matrix dimensions don't match");
  }

  // Matrix dimensions
  const El::Int height = data1.Height();
  const El::Int width = data1.Width();
  const El::Int local_height = data1.LocalHeight();
  const El::Int local_width = data1.LocalWidth();

  // Initialize covariance
  covs.Resize(1, width);

  // Local matrices
  const AbsMat& local_data1 = data1.LockedMatrix();
  const AbsMat& local_data2 = data2.LockedMatrix();
  const AbsMat& local_means1 = means1.LockedMatrix();
  const AbsMat& local_means2 = means2.LockedMatrix();
  CPUMat& local_covs = static_cast<CPUMat&>(covs.Matrix());

  // Accumulate sum and divide to get covariance
  LBANN_OMP_PARALLEL_FOR
  for(El::Int col = 0; col < local_width; ++col) {
    DataType sum = 0;
    const DataType mean1 = local_means1(0, col);
    const DataType mean2 = local_means2(0, col);
    for(El::Int row = 0; row < local_height; ++row) {
      const DataType val1 = local_data1(row, col);
      const DataType val2 = local_data2(row, col);
      sum += (val1 - mean1) * (val2 - mean2);
    }
    local_covs(0, col) = sum;
  }
  El::AllReduce(covs, covs.RedundantComm(), El::mpi::SUM);
  El::Scale(DataType(1) / height, local_covs);

}


}
