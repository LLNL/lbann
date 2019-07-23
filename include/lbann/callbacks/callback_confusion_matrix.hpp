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

#ifndef LBANN_CALLBACKS_CALLBACK_CONFUSION_MATRIX_HPP_INCLUDED
#define LBANN_CALLBACKS_CALLBACK_CONFUSION_MATRIX_HPP_INCLUDED

#include "lbann/callbacks/callback.hpp"

namespace lbann {

/** Compute confusion matrix.
 *  Confusion matrices are saved in CSV files of the form
 *  "<prefix><mode>.csv". The (i,j)-entry is the proportion of samples
 *  with prediction i and label j. The prediction and label layers are
 *  assumed to output one-hot vectors for each mini-batch sample.
 */
class lbann_callback_confusion_matrix : public lbann_callback {
public:

  lbann_callback_confusion_matrix(std::string prediction_layer,
                                  std::string label_layer,
                                  std::string prefix);
  lbann_callback_confusion_matrix(const lbann_callback_confusion_matrix&);
  lbann_callback_confusion_matrix& operator=(const lbann_callback_confusion_matrix&);
  lbann_callback_confusion_matrix* copy() const override {
    return new lbann_callback_confusion_matrix(*this);
  }
  std::string name() const override { return "confusion matrix"; }

  void setup(model *m) override;

  void on_epoch_begin(model *m) override      { reset_counts(*m); }
  void on_epoch_end(model *m) override        { save_confusion_matrix(*m); }
  void on_validation_begin(model *m) override { reset_counts(*m); }
  void on_validation_end(model *m) override   { save_confusion_matrix(*m); }
  void on_test_begin(model *m) override       { reset_counts(*m); }
  void on_test_end(model *m) override         { save_confusion_matrix(*m); }
  void on_batch_end(model *m) override          { update_counts(*m); }
  void on_batch_evaluate_end(model *m) override { update_counts(*m); }

private:

  /** Name of prediction layer.
   *  This layer is assumed to output one-hot vectors.
   */
  std::string m_prediction_layer;
  /** Name of label layer.
   *  This layer is assumed to output one-hot vectors.
   */
  std::string m_label_layer;
  /** Prefix for output files. */
  std::string m_prefix;

  /** Confusion matrix counts.
   *  Each vector should be interpreted as a num_classes x num_classes
   *  matrix in row-major order. The (i,j)-entry is the number of
   *  samples with prediction i and label j.
   */
  std::map<execution_mode,std::vector<El::Int>> m_counts;

  /** "View" into prediction matrix.
   *  This is a CPU matrix. If the prediction layer keeps data on GPU,
   *  then this will be a matrix copy rather than a matrix view.
   */
  std::unique_ptr<AbsDistMat> m_predictions_v;
  /** "View" into label matrix.
   *  This is a CPU matrix. If the label layer keeps data on GPU or in
   *  a different distribution than the prediction layer, then this
   *  will be a matrix copy rather than a matrix view.
   */
  std::unique_ptr<AbsDistMat> m_labels_v;

  /** Get prediction matrix. */
  const AbsDistMat& get_predictions(const model& m) const;
  /** Get label matrix. */
  const AbsDistMat& get_labels(const model& m) const;

  /** Reset confusion matrix counts. */
  void reset_counts(const model& m);
  /** Update confusion matrix counts.
   *  Counts are updated with current mini-batch predictions and
   *  labels.
   */
  void update_counts(const model& m);
  /** Output confusion matrix to file. */
  void save_confusion_matrix(const model& m);

};

// Builder function
std::unique_ptr<lbann_callback>
build_callback_confusion_matrix_from_pbuf(
  const google::protobuf::Message&, lbann_summary*);

} // namespace lbann

#endif  // LBANN_CALLBACKS_CALLBACK_CONFUSION_MATRIX_HPP_INCLUDED
