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
// lbann_data_reader_external .hpp .cpp - generic_data_reader class for data readers connected over IPC
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_external/data_reader_external.hpp"
#include <cstdio>
#include <string>

// TODO all response sub-messages can probably be const &

namespace lbann {

external_reader::external_reader() {}

external_reader::~external_reader() {}

external_reader* external_reader::copy() const {
  return new external_reader(*this);
}

void external_reader::load() {
  //m_connection.connect(get_file_dir() + get_data_filename());

  //m_connection = std::make_unique<connection>("socket_test");
  m_connection = std::unique_ptr<connection>(new connection("socket_test"));

  Request request;

  request.mutable_init_request();

  m_connection->message_write(request);
  Response response = m_connection->message_read();

  InitResponse init_response = response.init_response();

  m_num_samples = init_response.num_samples();
  m_data_size = init_response.data_size();

  m_has_labels = init_response.has_labels();
  m_num_labels = init_response.num_labels();
  m_label_size = init_response.label_size();

  m_has_responses = init_response.has_responses();
  m_num_responses = init_response.num_responses();
  m_response_size = init_response.response_size();

  m_data_dims.assign(init_response.data_dims().begin(), init_response.data_dims().end());

  m_reader_type = init_response.reader_type();

  // Reset indices.
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);

  select_subset_of_data();
}

void external_reader::load_data() {

  //std::cout << "m_num_samples: " << m_num_samples << std::endl;
  //std::cout << "m_data_size: " << m_data_size << std::endl;
  //std::cout << "protobuf size: " << response.fetch_data_response().data_size() << std::endl;

}

int external_reader::fetch_data(Mat& X) {
  // TODO split out index construction into its own thing
  // TODO do a similar thing for fetch_labels, fetch_responses
  if(!position_valid()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__)
      + " :: generic data reader load error: !position_valid"
      + " -- current pos = " + std::to_string(m_current_pos)
      + " and there are " + std::to_string(m_shuffled_indices.size()) + " indices");
  }

  int loaded_batch_size = get_loaded_mini_batch_size();
  const int end_pos = std::min(static_cast<size_t>(m_current_pos+loaded_batch_size),
                               m_shuffled_indices.size());
  const int mb_size = std::min(
    El::Int{((end_pos - m_current_pos) + m_sample_stride - 1) / m_sample_stride},
    X.Width());

  El::Zeros(X, X.Height(), X.Width());
  El::Zeros(m_indices_fetched_per_mb, mb_size, 1);

  Request request;
  auto fdrequest = request.mutable_fetch_data_request();
  auto indices = fdrequest->mutable_indices();

  for (int i = m_current_pos; i < end_pos; i++) {
    indices->add_value(m_shuffled_indices[i]);
  }

  m_connection->message_write(request);
  Response response = m_connection->message_read();
  auto samples = response.fetch_data_response().data().samples();

  for (El::Int s = 0; s < mb_size; s++) {
    int n = m_current_pos + s;
    int index = m_shuffled_indices[n];
    m_indices_fetched_per_mb.Set(s, 0, index);

    auto X_view = El::View(X,   El::IR(0, m_data_size), El::IR(s, s+1));
    auto float_values = samples[s].float_values();

    for (El::Int j = 0; j < m_data_size; j++) {
      X_view(j, 0) = float_values.float_samples(j);
    }
  }

  return mb_size;
}
//int fetch_labels(Mat& Y) override;
//int fetch_responses(Mat& Y) override;

}  // namespace lbann
