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

#include "lbann/utils/lbann_statistics.hpp"

using namespace El;

namespace lbann {

void mean_and_stdev(const Mat& data,
                    DataType& mean,
                    DataType& stdev) {

  // Matrix dimensions
  const Int height = data.Height();
  const Int width = data.Width();
  const Int size = height * width;

  // Compute sums over matrix entries
  const DataType shift = data(0, 0);
  DataType shifted_sum = 0;
  DataType shifted_sqsum = 0;
  for(Int col = 0; col < width; ++col) {
    for(Int row = 0; row < height; ++row) {
      const DataType shifted_val = data(row, col) - shift;
      shifted_sum += shifted_val;
      shifted_sqsum += shifted_val * shifted_val;
    }
  }

  // Compute mean and standard deviation
  const DataType shifted_mean = shifted_sum / size;
  const DataType shifted_sqmean = shifted_sqsum / size;
  mean = shifted_mean + shift;
  const DataType var = Max(shifted_sqmean - shifted_mean * shifted_mean, 0);
  stdev = Sqrt(var);

}

void columnwise_mean_and_stdev(const Mat& data,
                               Mat& means,
                               Mat& stdevs) {

  // Matrix dimensions
  const Int height = data.Height();
  const Int width = data.Width();

  // Initialize outputs
  means.Resize(1, width);
  stdevs.Resize(1, width);

  // Compute mean and standard deviation of each matrix column
#pragma omp parallel for
  for(Int col = 0; col < width; ++col) {
    const DataType shift = data(0, col);
    DataType shifted_sum = 0;
    DataType shifted_sqsum = 0;
    for(Int row = 0; row < height; ++row) {
      const DataType shifted_val = data(row, col) - shift;
      shifted_sum += shifted_val;
      shifted_sqsum += shifted_val * shifted_val;
    }
    const DataType shifted_mean = shifted_sum / height;
    const DataType shifted_sqmean = shifted_sqsum / height;
    const DataType mean = shifted_mean + shift;
    const DataType var = Max(shifted_sqmean - shifted_mean * shifted_mean, 0);
    const DataType stdev = Sqrt(var);
    means(0, col) = mean;
    stdevs(0, col) = stdev;
  }

}

/// @todo Numerically stable implementation
void columnwise_mean_and_stdev(const AbsDistMat& data,
                               AbsDistMat& means,
                               AbsDistMat& stdevs) {

#ifdef LBANN_DEBUG
  DistData data_dist(data), means_dist(means), stdevs_dist(stdevs);
  if(means_dist.colDist != STAR
     || means_dist.rowDist != data_dist.rowDist
     || stdevs_dist.colDist != STAR
     || stdevs_dist.rowDist != data_dist.rowDist) {
    throw lbann_exception("columnwise_mean_and_stdev: invalid matrix format");
  }
#endif // #ifdef LBANN_DEBUG

  // Matrix dimensions
  const Int height = data.Height();
  const Int width = data.Width();
  const Int local_height = data.LocalHeight();
  const Int local_width = data.LocalWidth();

  // Initialize outputs
  means.Resize(1, width);
  stdevs.Resize(1, width);

  // Local matrices
  const Mat& local_data = data.LockedMatrix();
  Mat& local_means = means.Matrix();
  Mat& local_stdevs = stdevs.Matrix();

  // Compute sum and sum of squares of each matrix column
#pragma omp parallel for
  for(Int col = 0; col < local_width; ++col) {
    DataType sum = 0;
    DataType sqsum = 0;
    for(Int row = 0; row < local_height; ++row) {
      const DataType val = local_data(row, col);
      sum += val;
      sqsum += val * val;
    }
    local_means(0, col) = sum;
    local_stdevs(0, col) = sqsum;
  }

  // Allreduce sums and sums of squares
  AllReduce(means, means.RedundantComm(), mpi::SUM);
  AllReduce(stdevs, stdevs.RedundantComm(), mpi::SUM);

  // Compute mean and standard deviation of each matrix column
#pragma omp parallel for
  for(Int col = 0; col < local_width; ++col) {
    const DataType mean = local_means(0, col) / height;
    const DataType sqmean = local_stdevs(0, col) / height;
    const DataType var = Max(sqmean - mean * mean, 0);
    const DataType stdev = Sqrt(var);
    local_means(0, col) = mean;
    local_stdevs(0, col) = stdev;
  }  

}

void rowwise_mean_and_stdev(const Mat& data,
                            Mat& means,
                            Mat& stdevs) {

  // Matrix dimensions
  const Int height = data.Height();
  const Int width = data.Width();

  // Initialize outputs
  means.Resize(height, 1);
  stdevs.Resize(height, 1);

  // Iterate through row blocks
  const Int block_size = 16;
#pragma omp parallel for
  for(Int row_start = 0; row_start < height; row_start += block_size) {
    const Int row_end = Min(row_start + block_size, height);
    
    // Initialize shift and sums for each row
    DataType* shifts = new DataType[block_size];
    for(Int row = row_start; row < row_end; ++row) {
      means(row, 0) = 0;
      stdevs(row, 0) = 0;
      shifts[row-row_start] = data(row, 0);
    }

    // Iterate through blocks in row block
    for(Int col_start = 0; col_start < width; col_start += block_size) {
      const Int col_end = Min(col_start + block_size, width);

      // Compute sums by iterating through block entries
      for(Int col = col_start; col < col_end; ++col) {
        for(Int row = row_start; row < row_end; ++row) {
          const DataType shift = shifts[row - row_start];
          const DataType shifted_val = data(row, col) - shift;
          means(row, 0) += shifted_val;
          stdevs(row, 0) += shifted_val * shifted_val;
        }
      }

    }

    // Compute mean and standard deviation of each row
    for(Int row = row_start; row < row_end; ++row) {
      const DataType shifted_mean = means(row, 0) / width;
      const DataType shifted_sqmean = stdevs(row, 0) / width;
      const DataType mean = shifted_mean + shifts[row - row_start];
      const DataType var = Max(shifted_sqmean - shifted_mean * shifted_mean, 0);
      const DataType stdev = Sqrt(var);
      means(row, 0) = mean;
      stdevs(row, 0) = stdev;
    }
    
    // Deallocate shifts
    delete[] shifts;

  }

}

/// @todo Numerically stable implementation
void rowwise_mean_and_stdev(const AbsDistMat& data,
                            AbsDistMat& means,
                            AbsDistMat& stdevs) {

#ifdef LBANN_DEBUG
  DistData data_dist(data), means_dist(means), stdevs_dist(stdevs);
  if(means_dist.colDist != data_dist.colDist
     || means_dist.rowDist != STAR
     || stdevs_dist.colDist != data_dist.colDist
     || stdevs_dist.rowDist != STAR) {
    throw lbann_exception("rowwise_mean_and_stdev: invalid matrix format");
  }
#endif // #ifdef LBANN_DEBUG

  // Matrix dimensions
  const Int height = data.Height();
  const Int width = data.Width();
  const Int local_height = data.LocalHeight();
  const Int local_width = data.LocalWidth();

  // Initialize outputs
  means.Resize(height, 1);
  stdevs.Resize(height, 1);

  // Local matrices
  const Mat& local_data = data.LockedMatrix();
  Mat& local_means = means.Matrix();
  Mat& local_stdevs = stdevs.Matrix();

  // Iterate through row blocks
  const Int block_size = 16;
#pragma omp parallel for
  for(Int row_start = 0; row_start < local_height; row_start += block_size) {
    const Int row_end = Min(row_start + block_size, local_height);
    
    // Iterate through blocks in row block
    for(Int col_start = 0; col_start < local_width; col_start += block_size) {
      const Int col_end = Min(col_start + block_size, local_width);

      // Compute sums by iterating through block entries
      for(Int col = col_start; col < col_end; ++col) {
        for(Int row = row_start; row < row_end; ++row) {
          const DataType val = local_data(row, col);
          local_means(row, 0) += val;
          local_stdevs(row, 0) += val * val;
        }
      }

    }

  }

  // Allreduce sums and sums of squares
  AllReduce(means, means.RedundantComm(), mpi::SUM);
  AllReduce(stdevs, stdevs.RedundantComm(), mpi::SUM);

  // Compute mean and standard deviation of each matrix row
#pragma omp parallel for
  for(Int row = 0; row < local_height; ++row) {
    const DataType mean = local_means(row, 0) / height;
    const DataType sqmean = local_stdevs(row, 0) / height;
    const DataType var = Max(sqmean - mean * mean, 0);
    const DataType stdev = Sqrt(var);
    local_means(row, 0) = mean;
    local_stdevs(row, 0) = stdev;
  }  

}

}
