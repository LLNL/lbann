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
// lbann_data_reader_cosmoflow .hpp .cpp - data_reader class for CosmoFlow
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_cosmoflow.hpp"
#include <cstdio>
#include <string>
#include <unordered_set>
#include <cnpy.h>
#include <cassert>

namespace lbann {
const std::string cosmoflow_reader::NPZ_KEY_DATA = "data";
const std::string cosmoflow_reader::NPZ_KEY_RESPONSES = "responses";

cosmoflow_reader::cosmoflow_reader(const bool shuffle)
    : generic_data_reader(shuffle),
      m_num_samples_total(0),
      m_num_features(0),
      m_num_response_features(0),
      m_current_idx(-1) {}

cosmoflow_reader::cosmoflow_reader(const cosmoflow_reader& other) :
    generic_data_reader(other),
    m_num_samples_total(other.m_num_samples_total),
    m_num_features(other.m_num_features),
    m_num_response_features(other.m_num_response_features),
    m_num_samples(other.m_num_samples),
    m_num_samples_prefix(other.m_num_samples_prefix),
    m_data_dims(other.m_data_dims),
    m_current_data(other.m_current_data),
    m_current_responses(other.m_current_responses),
    m_current_idx(other.m_current_idx),
    m_npz_paths(other.m_npz_paths),
    m_scaling_factor_int16(other.m_scaling_factor_int16) {}

cosmoflow_reader& cosmoflow_reader::operator=(const cosmoflow_reader& other) {
  generic_data_reader::operator=(other);
  m_num_samples_total = other.m_num_samples_total;
  m_num_features = other.m_num_features;
  m_num_response_features = other.m_num_response_features;
  m_num_samples = other.m_num_samples;
  m_num_samples_prefix = other.m_num_samples_prefix;
  m_data_dims = other.m_data_dims;
  m_current_data = other.m_current_data;
  m_current_responses = other.m_current_responses;
  m_current_idx = other.m_current_idx;
  m_npz_paths = other.m_npz_paths;
  m_scaling_factor_int16 = other.m_scaling_factor_int16;
  return *this;
}

void cosmoflow_reader::load() {
  for(const auto infile : m_npz_paths) {
    // Ensure the file exists.
    {
      std::ifstream ifs(infile);
      if (!ifs) {
        throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                              " cosmoflow_reader::load() - can't open file : " + infile);
      }
      ifs.close();
    }

    if(m_num_features == 0) {
      const cnpy::NpyArray data = cnpy::npz_load(infile, NPZ_KEY_DATA);
      m_data_dims = std::vector<int>(data.shape.begin()+1, data.shape.end());
      m_num_features = std::accumulate(m_data_dims.begin(),
                                       m_data_dims.end(),
                                       (size_t) 1,
                                       std::multiplies<size_t>());
    }

    const cnpy::NpyArray responses = cnpy::npz_load(infile, NPZ_KEY_RESPONSES);
    if(m_num_response_features == 0)
      m_num_response_features = responses.shape[1];
    m_num_samples.push_back(responses.shape[0]);
  }

  assert(m_num_samples.size() == m_npz_paths.size());
  m_num_samples_total = std::accumulate(m_num_samples.begin(),
                                        m_num_samples.end(),
                                        0);

  std::string s;
  for(auto i : m_num_samples) {
    s += std::to_string(i) + " ";
  }

  m_num_samples_prefix.resize(m_num_samples.size());
  std::partial_sum(m_num_samples.begin(),
                   m_num_samples.end(),
                   m_num_samples_prefix.begin());

  // reset indices.
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples_total);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();
}

int cosmoflow_reader::prepare_npz_file(const int data_id) {
  // OPTIMIZE
  for(auto i = m_num_samples_prefix.begin(); i != m_num_samples_prefix.end(); i++) {
    if(data_id < *i) {
      const auto position = std::distance(m_num_samples_prefix.begin(), i);
      const auto offset = data_id - (position == 0 ? 0 : *(i-1));
      if(position != m_current_idx) {
        // if(m_comm->get_rank_in_trainer() == 0) {
        //   std::cout << "cosmoflow_reader: Opening file "
        //             << m_npz_paths[position]
        //             << " @ " << position
        //             << std::endl;
        // }

        const cnpy::npz_t npz = cnpy::npz_load(m_npz_paths[position]);

        const auto safe_find =
            [](const cnpy::npz_t z, const std::string key) {
              const auto t = z.find(key);
              if(t != z.end()) {
                return t->second;
              } else {
                throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                                      " numpy_npz_reader::prepare_npz_file() - can't find npz key : " + key);
              }
            };

        m_current_data = safe_find(npz, NPZ_KEY_DATA);
        m_current_responses = safe_find(npz, NPZ_KEY_RESPONSES);
        m_current_idx = position;
      }
      return offset;
    }
  }
  throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
                        " cosmoflow_reader::prepare_npz_file() - invalid data_id : " + std::to_string(data_id));
  return -1;
}

bool cosmoflow_reader::fetch_datum(Mat& X, int data_id, int mb_idx) {
  const auto offset = prepare_npz_file(data_id);
  Mat X_v = El::View(X, El::IR(0, X.Height()), El::IR(mb_idx, mb_idx+1));

  // Convert int16 to DataType.
  const short *data = m_current_data.data<short>() + (size_t) offset * m_num_features;
  DataType *dest = X_v.Buffer();

  // OPTIMIZE
  LBANN_OMP_PARALLEL_FOR
      for(int j = 0; j < m_num_features; j++)
        dest[j] = data[j] * m_scaling_factor_int16;

  return true;
}

bool cosmoflow_reader::fetch_response(Mat& Y, int data_id, int mb_idx) {
  const auto offset = prepare_npz_file(data_id);

  void *responses = NULL;
  if (m_current_responses.word_size == 4) {
    responses = (void *) (m_current_responses.data<float>()
                          + offset * m_num_response_features);
  } else if (m_current_responses.word_size == 8) {
    responses = (void *) (m_current_responses.data<double>()
                          + offset * m_num_response_features);
  }
  Mat Y_v = El::View(Y, El::IR(0, Y.Height()), El::IR(mb_idx, mb_idx + 1));
  std::memcpy(Y_v.Buffer(), responses,
              m_num_response_features * m_current_responses.word_size);

  return true;
}

}  // namespace lbann
