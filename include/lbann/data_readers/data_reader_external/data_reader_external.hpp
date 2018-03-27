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

#ifndef LBANN_DATA_READER_EXTERNAL_HPP
#define LBANN_DATA_READER_EXTERNAL_HPP

#include <mutex>

#include <data_reader_communication.pb.h>

#include "lbann/data_readers/data_reader.hpp"

namespace lbann {

class external_reader : public generic_data_reader {
  public:
    external_reader(bool shuffle = true);
    external_reader(const external_reader&);
    external_reader& operator=(const external_reader&);
    ~external_reader() override;

    external_reader* copy() const override;

    std::string get_type() const override;

    void set_has_labels(bool);
    void set_has_responses(bool);

    void load() override;

    int get_num_labels() const override;
    int get_linearized_data_size() const override;
    int get_linearized_label_size() const override;

    const std::vector<int> get_data_dims() const override;

  protected:
    bool fetch_datum(Mat& X, int data_id, int mb_idx, int tid) override;
    bool fetch_response(Mat& Y, int data_id, int mb_idx, int tid) override;
    bool fetch_label(Mat& Y, int data_id, int mb_idx, int tid) override;

    int m_num_samples = -1;
    int m_num_features = -1;
    int m_num_labels = -1;
    bool m_has_labels = false;
    bool m_has_responses = false;

    int m_label_count = -1;
    int m_data_size = -1;
    int m_label_size = -1;
    std::vector<int> m_dims;

  private:
    void connect();
    void disconnect();

    void get_config();

    Response message_transaction(const Request& request) const;
    Response message_read() const;
    bool message_write(Request request) const;

    bool m_connected = false;

    std::string m_lbann_comm_dir;

    std::string m_lbann_to_external_file;
    std::string m_external_to_lbann_file;

    int m_lbann_to_external_fd = -1;
    int m_external_to_lbann_fd = -1;

    // read_message/write_message are visibly const, but they resize their
    // buffers to fit the largest message they've seen
    mutable uint8_t *m_write_buffer = nullptr;
    mutable size_t m_write_buffer_size = 0;

    mutable uint8_t *m_read_buffer = nullptr;
    mutable size_t m_read_buffer_size = 0;

    // ensure that writes are paired with a read
    mutable std::mutex m_read_write_completion;
};
}  // namespace lbann

#endif  // LBANN_DATA_READER_EXTERNAL_HPP
