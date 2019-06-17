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
// data_reader_ascii .hpp .cpp - generic_data_reader class for ASCII text files
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READER_ASCII_HPP
#define LBANN_DATA_READER_ASCII_HPP

#include "data_reader.hpp"

namespace lbann {

class ascii_reader : public generic_data_reader {
 public:
  ascii_reader(int sequence_length = 1, bool shuffle = true);
  ascii_reader(const ascii_reader&) = default;
  ascii_reader& operator=(const ascii_reader&) = default;
  ~ascii_reader() override = default;
  ascii_reader* copy() const override { return new ascii_reader(*this); }

  std::string get_type() const override {
    return "ascii_reader";
  }

  void load() override;

  int get_linearized_data_size() const override {
    return 128 * m_sequence_length;
  }
  int get_linearized_label_size() const override {
    return 128 * m_sequence_length;
  }
  const std::vector<int> get_data_dims() const override {
    return {128 * m_sequence_length};
  }

 protected:
  bool fetch_datum(CPUMat& X, int data_id, int mb_idx) override;
  bool fetch_label(CPUMat& Y, int data_id, int mb_idx) override;

  /** Length of text sequence. */
  int m_sequence_length;
  /** Size of data file in bytes. */
  int m_file_size;

};

}  // namespace lbann

#endif  // LBANN_DATA_READER_ASCII_HPP
