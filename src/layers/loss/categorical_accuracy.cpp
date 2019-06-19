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

#include "lbann/layers/loss/categorical_accuracy.hpp"
#include <limits>


namespace lbann {

namespace {

/** CPU implementation of categorical accuracy layer forward prop. */
void fp_cpu(lbann_comm& comm,
            const AbsDistMat& predictions,
            const AbsDistMat& labels,
            AbsDistMat& loss) {

  // Local matrices
  const auto& local_predictions = predictions.LockedMatrix();
  const auto& local_labels = labels.LockedMatrix();
  auto& local_loss = loss.Matrix();

  // Dimensions
  const auto& height = predictions.Height();
  const auto& local_height = local_predictions.Height();
  const auto& local_width = local_predictions.Width();
  if (local_width < 1) { return; }

  // Column communicator
  auto&& col_comm = predictions.ColComm();
  const auto& col_comm_rank = El::mpi::Rank(col_comm);
  const auto& col_comm_size = El::mpi::Size(col_comm);
  const auto& col_comm_root = loss.RowOwner(0);

  // Find largest prediction entries in local data
  std::vector<DataType> prediction_vals(local_width);
  std::vector<El::Int> prediction_inds(local_width);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    DataType max_val = -std::numeric_limits<DataType>::infinity();
    El::Int max_ind = height;
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& val = local_predictions(row, col);
      if (val > max_val) {
        max_val = val;
        max_ind = predictions.GlobalRow(row);
      }
    }
    prediction_vals[col] = max_val;
    prediction_inds[col] = max_ind;
  }

  // Gather large prediction entries
  /// @todo Non-blocking gather
  Al::request prediction_vals_req, prediction_inds_req;
  std::vector<DataType> gathered_prediction_vals;
  std::vector<El::Int> gathered_prediction_inds;
  if (col_comm_size > 1) {
    if (col_comm_rank != col_comm_root) {
      comm.gather(prediction_vals.data(), prediction_vals.size(),
                  col_comm_root, col_comm,
                  El::SyncInfo<El::Device::CPU>{});
      comm.gather(prediction_inds.data(), prediction_inds.size(),
                  col_comm_root, col_comm,
                  El::SyncInfo<El::Device::CPU>{});
    } else {
      gathered_prediction_vals.resize(prediction_vals.size() * col_comm_size);
      gathered_prediction_inds.resize(prediction_inds.size() * col_comm_size);
      comm.gather(prediction_vals.data(), prediction_vals.size(),
                  gathered_prediction_vals.data(),
                  col_comm, El::SyncInfo<El::Device::CPU>{});
      comm.gather(prediction_inds.data(), prediction_inds.size(),
                  gathered_prediction_inds.data(),
                  col_comm, El::SyncInfo<El::Device::CPU>{});
    }
  }

  // Find largest label entries in local data
  std::vector<DataType> label_vals(local_width);
  std::vector<El::Int> label_inds(local_width);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    DataType max_val = -std::numeric_limits<DataType>::infinity();
    El::Int max_ind = height;
    for (El::Int row = 0; row < local_height; ++row) {
      const auto& val = local_labels(row, col);
      if (val > max_val) {
        max_val = val;
        max_ind = labels.GlobalRow(row);
      }
    }
    label_vals[col] = max_val;
    label_inds[col] = max_ind;
  }

  // Gather large label entries
  /// @todo Non-blocking gather
  Al::request label_vals_req, label_inds_req;
  std::vector<DataType> gathered_label_vals;
  std::vector<El::Int> gathered_label_inds;
  if (col_comm_size > 1) {
    if (col_comm_rank != col_comm_root) {
      comm.gather(label_vals.data(), label_vals.size(),
                  col_comm_root, col_comm,
                  El::SyncInfo<El::Device::CPU>{});
      comm.gather(label_inds.data(), label_inds.size(),
                  col_comm_root, col_comm,
                  El::SyncInfo<El::Device::CPU>{});
    } else {
      gathered_label_vals.resize(label_vals.size() * col_comm_size);
      gathered_label_inds.resize(label_inds.size() * col_comm_size);
      comm.gather(label_vals.data(), label_vals.size(),
                  gathered_label_vals.data(),
                  col_comm, El::SyncInfo<El::Device::CPU>{});
      comm.gather(label_inds.data(), label_inds.size(),
                  gathered_label_inds.data(),
                  col_comm, El::SyncInfo<El::Device::CPU>{});
    }
  }

  // Find largest prediction entry in global data
  comm.wait(prediction_vals_req);
  comm.wait(prediction_inds_req);
  if (col_comm_size > 1 && col_comm_rank == col_comm_root) {
    LBANN_OMP_PARALLEL_FOR
    for (El::Int col = 0; col < local_width; ++col) {
      DataType max_val = -std::numeric_limits<DataType>::infinity();
      El::Int max_ind = height;
      for (El::Int rank = 0; rank < col_comm_size; ++rank) {
        const auto& val = gathered_prediction_vals[col + rank * local_width];
        const auto& ind = gathered_prediction_inds[col + rank * local_width];
        if (val > max_val || (val == max_val && ind < max_ind)) {
          max_val = val;
          max_ind = ind;
        }
      }
      label_vals[col] = max_val;
      label_inds[col] = max_ind;
    }
  }

  // Find largest label entry in global data
  comm.wait(label_vals_req);
  comm.wait(label_inds_req);
  if (col_comm_size > 1 && col_comm_rank == col_comm_root) {
    LBANN_OMP_PARALLEL_FOR
    for (El::Int col = 0; col < local_width; ++col) {
      DataType max_val = -std::numeric_limits<DataType>::infinity();
      El::Int max_ind = height;
      for (El::Int rank = 0; rank < col_comm_size; ++rank) {
        const auto& val = gathered_label_vals[col + rank * local_width];
        const auto& ind = gathered_label_inds[col + rank * local_width];
        if (val > max_val || (val == max_val && ind < max_ind)) {
          max_val = val;
          max_ind = ind;
        }
      }
      label_vals[col] = max_val;
      label_inds[col] = max_ind;
    }
  }

  // Compute categorical accuracy
  if (col_comm_rank == col_comm_root) {
    LBANN_OMP_PARALLEL_FOR
    for (El::Int col = 0; col < local_width; ++col) {
      local_loss(0, col) = (prediction_inds[col] == label_inds[col] ?
                            DataType(1) : DataType(0));
    }
  }

}

} // namespace

template <>
void categorical_accuracy_layer<data_layout::MODEL_PARALLEL, El::Device::CPU>
     ::fp_compute() {
  fp_cpu(*get_comm(),
         get_prev_activations(0),
         get_prev_activations(1),
         get_activations());
}
template <>
void categorical_accuracy_layer<data_layout::DATA_PARALLEL, El::Device::CPU>
     ::fp_compute() {
  fp_cpu(*get_comm(),
         get_prev_activations(0),
         get_prev_activations(1),
         get_activations());
}

} // namespace lbann
