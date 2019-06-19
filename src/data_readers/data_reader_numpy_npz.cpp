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
//
// data_reader_numpy_npz .hpp .cpp - generic_data_reader class for numpy .npz dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_numpy_npz.hpp"
#include <cstdio>
#include <string>
#include <unordered_set>
#include <cnpy.h>

namespace lbann {
  const std::string numpy_npz_reader::NPZ_KEY_DATA = "data";
  const std::string numpy_npz_reader::NPZ_KEY_LABELS = "labels";
  const std::string numpy_npz_reader::NPZ_KEY_RESPONSES = "responses";

  numpy_npz_reader::numpy_npz_reader(const bool shuffle)
    : generic_data_reader(shuffle),
      m_num_samples(0),
      m_num_features(0),
      m_num_response_features(0) {}

  numpy_npz_reader::numpy_npz_reader(const numpy_npz_reader& other) :
    generic_data_reader(other),
    m_num_samples(other.m_num_samples),
    m_num_features(other.m_num_features),
    m_num_labels(other.m_num_labels),
    m_num_response_features(other.m_num_response_features),
    m_has_labels(other.m_has_labels),
    m_has_responses(other.m_has_responses),
    m_data(other.m_data),
    m_labels(other.m_labels),
    m_responses(other.m_responses),
    m_scaling_factor_int16(other.m_scaling_factor_int16) {}

  numpy_npz_reader& numpy_npz_reader::operator=(const numpy_npz_reader& other) {
    generic_data_reader::operator=(other);
    m_num_samples = other.m_num_samples;
    m_num_features = other.m_num_features;
    m_num_labels = other.m_num_labels;
    m_num_response_features = other.m_num_response_features;
    m_has_labels = other.m_has_labels;
    m_has_responses = other.m_has_responses;
    m_data = other.m_data;
    m_labels = other.m_labels;
    m_responses = other.m_responses;
    m_scaling_factor_int16 = other.m_scaling_factor_int16;
    return *this;
  }

  void numpy_npz_reader::load() {
    std::string infile = get_data_filename();
    // Ensure the file exists.
    std::ifstream ifs(infile);
    if (!ifs) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                            " numpy_npz_reader::load() - can't open file : " + infile);
    }
    ifs.close();

    const cnpy::npz_t npz = cnpy::npz_load(infile);

    std::vector<std::tuple<const bool, const std::string, cnpy::NpyArray &> > npyLoadList;
    npyLoadList.push_back(std::forward_as_tuple(true,            NPZ_KEY_DATA,      m_data));
    npyLoadList.push_back(std::forward_as_tuple(m_has_labels,    NPZ_KEY_LABELS,    m_labels));
    npyLoadList.push_back(std::forward_as_tuple(m_has_responses, NPZ_KEY_RESPONSES, m_responses));
    for(const auto npyLoad : npyLoadList) {
      // Check whether the tensor have to be loaded.
      if(!std::get<0>(npyLoad)) {
        continue;
      }

      // Load the tensor.
      const std::string key = std::get<1>(npyLoad);
      cnpy::NpyArray &ary = std::get<2>(npyLoad);
      const auto i = npz.find(key);
      if(i != npz.end()) {
        ary = i->second;
      } else {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                              " numpy_npz_reader::load() - can't find npz key : " + key);
      }

      // Check whether the labels/responses has the same number of samples.
      if(key == NPZ_KEY_DATA) {
        m_num_samples = m_data.shape[0];
      } else if(m_num_samples != (int) ary.shape[0]) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                              " numpy_npz_reader::load() - the number of samples of data and " + key + " do not match : "
                              + std::to_string(m_num_samples) + " vs. " + std::to_string(ary.shape[0]));
      }
    }

    m_num_features = std::accumulate(m_data.shape.begin() + 1,
                                     m_data.shape.end(),
                                     (unsigned) 1,
                                     std::multiplies<unsigned>());
    if(m_has_responses) {
      m_num_response_features = std::accumulate(m_responses.shape.begin() + 1,
                                                m_responses.shape.end(),
                                                (unsigned) 1,
                                                std::multiplies<unsigned>());
    }

    // Ensure we understand the word size.
    if (!(m_data.word_size == 2 || m_data.word_size == 4 || m_data.word_size == 8)) {
      throw lbann_exception("numpy_npz_reader: word size " + std::to_string(m_data.word_size) +
                            " not supported");
    }

    if (m_has_labels) {
      // Determine number of label classes.
      std::unordered_set<int> label_classes;
      if (m_labels.word_size != 4) {
        throw lbann_exception("numpy_npz_reader: label numpy array should be in int32");
      }
      int *data = m_labels.data<int>();
      for (int i = 0; i < m_num_samples; ++i) {
        label_classes.insert((int) data[i]);
      }

      // Sanity checks.
      auto minmax = std::minmax_element(label_classes.begin(), label_classes.end());
      if (*minmax.first != 0) {
        throw lbann_exception("numpy_reader: classes are not indexed from 0");
      }
      if (*minmax.second != (int) label_classes.size() - 1) {
        throw lbann_exception("numpy_reader: label classes are not contiguous");
      }
      m_num_labels = label_classes.size();
    }

    // Reset indices.
    m_shuffled_indices.clear();
    m_shuffled_indices.resize(m_num_samples);
    std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
    select_subset_of_data();
  }

  bool numpy_npz_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
    Mat X_v = El::View(X, El::IR(0, X.Height()), El::IR(mb_idx, mb_idx+1));

    if (m_data.word_size == 2) {
      // Convert int16 to DataType.
      const short *data = m_data.data<short>() + data_id * m_num_features;
      DataType *dest = X_v.Buffer();

      // OPTIMIZE
      LBANN_OMP_PARALLEL_FOR
        for(int j = 0; j < m_num_features; j++)
          dest[j] = data[j] * m_scaling_factor_int16;

    } else {
      void *data = NULL;
      if (m_data.word_size == 4) {
        data = (void *) (m_data.data<float>() + data_id * m_num_features);
      } else if (m_data.word_size == 8) {
        data = (void *) (m_data.data<double>() + data_id * m_num_features);
      }
      std::memcpy(X_v.Buffer(), data, m_num_features * m_data.word_size);
    }
    return true;
  }

  bool numpy_npz_reader::fetch_label(Mat& Y, int data_id, int mb_idx) {
    if (!m_has_labels) {
      throw lbann_exception("numpy_npz_reader: do not have labels");
    }
    const int label = m_labels.data<int>()[data_id];
    Y(label, mb_idx) = 1;
    return true;
  }

  bool numpy_npz_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
    if (!m_has_responses) {
      throw lbann_exception("numpy_npz_reader: do not have responses");
    }
    void *responses = NULL;
    if (m_responses.word_size == 4) {
      responses = (void *) (m_responses.data<float>()
                            + data_id * m_num_response_features);
    } else if (m_responses.word_size == 8) {
      responses = (void *) (m_responses.data<double>()
                            + data_id * m_num_response_features);
    }
    Mat Y_v = El::View(Y, El::IR(0, Y.Height()), El::IR(mb_idx, mb_idx + 1));
    std::memcpy(Y_v.Buffer(), responses,
                m_num_response_features * m_responses.word_size);
    return true;
  }

}  // namespace lbann
