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
// lbann_numpy_reader .hpp .cpp
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_numpy.hpp"
#include <stdio.h>
#include <string>
#include <unordered_set>
#include <cnpy.h>

namespace lbann {

numpy_reader::numpy_reader(int batch_size, bool shuffle)
  : generic_data_reader(batch_size, shuffle), m_num_samples(0),
    m_num_features(0) {}

numpy_reader::numpy_reader(const numpy_reader& other) :
  generic_data_reader(other),
  m_num_samples(other.m_num_samples),
  m_num_features(other.m_num_features),
  m_num_labels(other.m_num_labels),
  m_has_labels(other.m_has_labels),
  m_has_responses(other.m_has_responses),
  m_data(other.m_data) {}

numpy_reader& numpy_reader::operator=(const numpy_reader& other) {
  generic_data_reader::operator=(other);
  m_num_samples = other.m_num_samples;
  m_num_features = other.m_num_features;
  m_num_labels = other.m_num_labels;
  m_has_labels = other.m_has_labels;
  m_has_responses = other.m_has_responses;
  m_data = other.m_data;
  return *this;
}

void numpy_reader::load() {
  std::string infile = get_data_filename();
  // Ensure the file exists.
  std::ifstream ifs(infile);
  if (!ifs) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " numpy_reader::load() - can't open file : " + infile);
  }
  ifs.close();

  m_data = cnpy::npy_load(infile);
  m_num_samples = m_data.shape[0];
  m_num_features = std::accumulate(
    m_data.shape.begin() + 1, m_data.shape.end(), (unsigned) 1,
    std::multiplies<unsigned>());

  // Ensure we understand the word size.
  if (!(m_data.word_size == 4 || m_data.word_size == 8)) {
    throw lbann_exception(
      "numpy_reader: word size " + std::to_string(m_data.word_size) +
      " not supported");
  }
  // Fortran order not yet supported.
  if (m_data.fortran_order) {
    throw lbann_exception(
      "numpy_reader: fortran order not supported");
  }
  // Don't currently support both labels and responses.
  if (m_has_labels && m_has_responses) {
    throw lbann_exception(
      "numpy_reader: labels and responses not supported at same time");
  }

  if (m_has_labels) {
    // Shift feature count because the last becomes the label.
    m_num_features -= 1;
    // Determine number of label classes.
    std::unordered_set<int> label_classes;
    for (int i = 0; i < m_num_samples; ++i) {
      if (m_data.word_size == 4) {
        float *data = m_data.data<float>() + i*(m_num_features+1);
        label_classes.insert((int) data[m_num_features+1]);
      } else if (m_data.word_size == 8) {
        double *data = m_data.data<double>() + i*(m_num_features+1);
        label_classes.insert((int) data[m_num_features+1]);
      }
    }
    // Sanity checks.
    auto minmax = std::minmax_element(label_classes.begin(), label_classes.end());
    if (*minmax.first != 0) {
      throw lbann_exception(
        "numpy_reader: classes are not indexed from 0");
    }
    if (*minmax.second != (int) label_classes.size() - 1) {
      throw lbann_exception(
        "numpy_reader: label classes are not contiguous");
    }
    m_num_labels = label_classes.size();
  }
  if (m_has_responses) {
    // Last feature becomes the response.
    m_num_features -= 1;
  }

  // Reset indices.
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

bool numpy_reader::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {
  int features_size = m_num_features;
  if (m_has_labels || m_has_responses) {
    features_size += 1;
  }
  if (m_data.word_size == 4) {
    float *data = m_data.data<float>() + data_id * features_size;
    for (int j = 0; j < m_num_features; ++j) {
      X(j, mb_idx) = data[j];
    }
  } else if (m_data.word_size == 8) {
    double *data = m_data.data<double>() + data_id * features_size;
    for (int j = 0; j < m_num_features; ++j) {
      X(j, mb_idx) = data[j];
    }
  }
  return true;
}

bool numpy_reader::fetch_label(Mat& Y, int data_id, int mb_idx, int tid) {
  if (!m_has_labels) {
    throw lbann_exception("numpy_reader: do not have labels");
  }
  int label = 0;
  if (m_data.word_size == 4) {
    float *data = m_data.data<float>() + data_id*(m_num_features+1);
    label = (int) data[m_num_features+1];
  } else if (m_data.word_size == 8) {
    double *data = m_data.data<double>() + data_id*(m_num_features+1);
    label = (int) data[m_num_features+1];
  }
  Y(label, mb_idx) = 1;
  return true;
}

bool numpy_reader::fetch_response(Mat& Y, int data_id, int mb_idx, int tid) {
  if (!m_has_responses) {
    throw lbann_exception("numpy_reader: do not have responses");
  }
  DataType response = DataType(0);
  if (m_data.word_size == 4) {
    float *data = m_data.data<float>() + data_id*(m_num_features+1);
    response = (DataType) data[m_num_features+1];
  } else if (m_data.word_size == 8) {
    double *data = m_data.data<double>() + data_id*(m_num_features+1);
    response = (DataType) data[m_num_features+1];
  }
  Y(0, mb_idx) = response;
  return true;
}

}  // namespace lbann
