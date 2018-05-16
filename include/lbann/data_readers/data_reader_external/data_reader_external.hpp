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

#include <memory>

#include <data_reader_communication.pb.h>

#include "lbann/data_readers/data_reader_external/connection.hpp"
#include "lbann/data_readers/data_reader.hpp"

namespace lbann {

class external_reader : public generic_data_reader {
  public:
    external_reader();
    ~external_reader() override;

    external_reader(const external_reader &other) :
    generic_data_reader(other) {

      // reinitialize everything
    }

    external_reader* copy() const override;

    void load() override;

    std::string get_type() const override {
      return m_reader_type;
    }

    int fetch_data(Mat& X) override;
    int fetch_labels(Mat& Y) override;
    int fetch_responses(Mat& Y) override;

    int get_linearized_data_size() const override {
      return m_data_size;
    }

    int get_num_labels() const override {
      return m_num_labels;
    }
    int get_linearized_label_size() const override {
      return m_label_size;
    }

    int get_num_responses() const override {
      return m_num_responses;
    }
    int get_linearized_response_size() const override {
      return m_response_size;
    }

    const std::vector<int> get_data_dims() const override {
      return m_data_dims;
    }

  private:
    void load_data();

    std::unique_ptr<connection> m_connection = nullptr;

    int m_num_samples = 0; // total training examples
    int m_data_size = 0; // elements in a sample

    bool m_has_labels = false;
    int m_num_labels = 0;
    int m_label_size = 0;

    bool m_has_responses = false;
    int m_num_responses = 0;
    int m_response_size = 0;

    std::vector<int> m_data_dims{};

    std::string m_reader_type{};
};
}  // namespace lbann

#endif  // LBANN_DATA_READER_EXTERNAL_HPP
