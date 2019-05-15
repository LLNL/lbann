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
///////////////////////////////////////////////////////////////////////////////

#include "lbann/callbacks/callback_confusion_matrix.hpp"

namespace lbann {

// ---------------------------------------------------------
// Constructors
// ---------------------------------------------------------

lbann_callback_confusion_matrix::lbann_callback_confusion_matrix(std::string prediction_layer,
                                                                 std::string label_layer,
                                                                 std::string prefix)
  : lbann_callback(1, nullptr),
    m_prediction_layer(std::move(prediction_layer)),
    m_label_layer(std::move(label_layer)),
    m_prefix(std::move(prefix)) {}

lbann_callback_confusion_matrix::lbann_callback_confusion_matrix(const lbann_callback_confusion_matrix& other)
  : lbann_callback(other),
    m_prediction_layer(other.m_prediction_layer),
    m_label_layer(other.m_label_layer),
    m_prefix(other.m_prefix),
    m_counts(other.m_counts),
    m_predictions_v(other.m_predictions_v ? other.m_predictions_v->Copy() : nullptr),
    m_labels_v(other.m_labels_v ? other.m_labels_v->Copy() : nullptr) {}

lbann_callback_confusion_matrix& lbann_callback_confusion_matrix::operator=(const lbann_callback_confusion_matrix& other) {
  lbann_callback::operator=(other);
  m_prediction_layer = other.m_prediction_layer;
  m_label_layer = other.m_label_layer;
  m_prefix = other.m_prefix;
  m_counts = other.m_counts;
  m_predictions_v.reset(other.m_predictions_v ? other.m_predictions_v->Copy() : nullptr);
  m_labels_v.reset(other.m_labels_v ? other.m_labels_v->Copy() : nullptr);
  return *this;
}

// ---------------------------------------------------------
// Setup
// ---------------------------------------------------------

void lbann_callback_confusion_matrix::setup(model* m) {
  lbann_callback::setup(m);

  // Initialize matrix views/copies
  const auto& predictions = get_predictions(*m);
  const auto& labels = get_labels(*m);
  auto dist_data = predictions.DistData();
  dist_data.device = El::Device::CPU;
  m_predictions_v.reset(AbsDistMat::Instantiate(dist_data));
  m_labels_v.reset(AbsDistMat::Instantiate(dist_data));

  // Check output dimensions of prediction and label layers
  if (predictions.Height() != labels.Height()) {
    std::stringstream err;
    err << "callback \"" << name() << "\" "
        << "has prediction and label layers with different dimensions "
        << "(prediction layer \"" << m_prediction_layer << "\" "
        << "outputs " << predictions.Height() << " entries, "
        << "label layer \"" << m_label_layer << "\" "
        << "outputs " << labels.Height() << " entries)";
    LBANN_ERROR(err.str());
  }

}

// ---------------------------------------------------------
// Matrix access functions
// ---------------------------------------------------------

const AbsDistMat& lbann_callback_confusion_matrix::get_predictions(const model& m) const {
  for (const auto* l : m.get_layers()) {
    if (l->get_name() == m_prediction_layer) {
      return l->get_activations();
    }
  }
  std::stringstream err;
  err << "callback \"" << name() << "\" could not find "
      << "prediction layer \"" << m_prediction_layer << "\"";
  LBANN_ERROR(err.str());
  return m.get_layers()[0]->get_activations();
}

const AbsDistMat& lbann_callback_confusion_matrix::get_labels(const model& m) const {
  for (const auto* l : m.get_layers()) {
    if (l->get_name() == m_label_layer) {
      return l->get_activations();
    }
  }
  std::stringstream err;
  err << "callback \"" << name() << "\" could not find "
      << "label layer \"" << m_prediction_layer << "\"";
  LBANN_ERROR(err.str());
  return m.get_layers()[0]->get_activations();
}

// ---------------------------------------------------------
// Count management functions
// ---------------------------------------------------------

void lbann_callback_confusion_matrix::reset_counts(const model& m) {
  auto& counts = m_counts[m.get_execution_mode()];
  const auto& num_classes = get_predictions(m).Height();
  counts.assign(num_classes * num_classes, 0);
}

void lbann_callback_confusion_matrix::update_counts(const model& m) {
  constexpr DataType zero = 0;

  // Get predictions
  const auto& predictions = get_predictions(m);
  const auto& num_classes = predictions.Height();
  m_predictions_v->Empty(false);
  m_predictions_v->AlignWith(predictions);
  if (m_predictions_v->DistData() == predictions.DistData()) {
    El::LockedView(*m_predictions_v, predictions);
  } else {
    El::Copy(predictions, *m_predictions_v);
  }
  const auto& local_predictions = m_predictions_v->LockedMatrix();

  // Get labels
  const auto& labels = get_labels(m);
  m_labels_v->Empty(false);
  m_labels_v->AlignWith(predictions);
  if (m_labels_v->DistData() == labels.DistData()) {
    El::LockedView(*m_labels_v, labels);
  } else {
    El::Copy(labels, *m_labels_v);
  }
  const auto& local_labels = m_labels_v->LockedMatrix();

  // Update counts
  auto& counts = m_counts[m.get_execution_mode()];
  const auto& local_height = local_predictions.Height();
  const auto& local_width = local_predictions.Width();
  for (El::Int local_col = 0; local_col < local_width; ++local_col) {
    El::Int prediction_index = -1, label_index = -1;
    LBANN_OMP_PARALLEL_FOR
    for (El::Int local_row = 0; local_row < local_height; ++local_row) {
      if (local_predictions(local_row, local_col) != zero) {
        prediction_index = m_predictions_v->GlobalRow(local_row);
      }
      if (local_labels(local_row, local_col) != zero) {
        label_index = m_labels_v->GlobalRow(local_row);
      }
    }
    if (prediction_index >= 0 && label_index >= 0) {
      counts[label_index + prediction_index * num_classes]++;
    }
  }

}

void lbann_callback_confusion_matrix::save_confusion_matrix(const model& m) {

  // Get counts
  const auto& mode = m.get_execution_mode();
  auto& counts = m_counts[mode];

  // Accumulate counts in master process
  // Note: Counts in non-root processes are set to zero, so this can
  // be called multiple times without affecting correctness.
  auto&& comm = *m.get_comm();
  if (comm.am_trainer_master()) {
    comm.trainer_reduce(static_cast<El::Int*>(MPI_IN_PLACE),
                      counts.size(),
                      counts.data());
  } else {
    comm.trainer_reduce(counts.data(), counts.size(),
                      comm.get_trainer_master());
    counts.assign(counts.size(), 0);
  }

  // Save confusion matrix on master process
  if (comm.am_trainer_master()) {
    const auto& num_classes = get_predictions(m).Height();
    const auto& total_count = std::accumulate(counts.begin(), counts.end(), 0);
    const auto& scale = DataType(1) / total_count;

    // Construct output file name
    std::string mode_string;
    switch (mode) {
    case execution_mode::training:
      mode_string = "train-epoch" + std::to_string(m.get_epoch());
      break;
    case execution_mode::validation:
      mode_string = "validation-epoch" + std::to_string(m.get_epoch());
      break;
    case execution_mode::testing:
      mode_string = "test";
      break;
    default: return; // Exit immediately if execution mode is unknown
    }

    // Write to file
    std::ofstream fs(m_prefix + mode_string + ".csv");
    for (El::Int i = 0; i < num_classes; ++i) {
      for (El::Int j = 0; j < num_classes; ++j) {
        fs << (j > 0 ? "," : "") << counts[j + i * num_classes] * scale;
      }
      fs << "\n";
    }
    fs.close();

  }

}

}  // namespace lbann
