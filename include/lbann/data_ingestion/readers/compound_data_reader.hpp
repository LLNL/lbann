////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_GENERIC_COMPOUND_DATA_READER_HPP
#define LBANN_GENERIC_COMPOUND_DATA_READER_HPP

#include "lbann/data_ingestion/data_reader.hpp"

#include <utility>

namespace lbann {

/**
 * Data reader for merging the samples from multiple data readers into a
 * single dataset.
 */
class generic_compound_data_reader : public generic_data_reader
{
public:
  generic_compound_data_reader(std::vector<generic_data_reader*> data_readers,
                               bool shuffle = true)
    : generic_data_reader(shuffle), m_data_readers(std::move(data_readers))
  {
    if (m_data_readers.empty()) {
      throw lbann_exception(
        "generic_compound_data_reader: data reader list empty");
    }
  }

  generic_compound_data_reader(const generic_compound_data_reader& other)
    : generic_data_reader(other)
  {
    for (auto&& reader : other.m_data_readers) {
      m_data_readers.push_back(reader->copy());
    }
  }
  generic_compound_data_reader&
  operator=(const generic_compound_data_reader& other)
  {
    generic_data_reader::operator=(other);
    for (auto&& reader : m_data_readers) {
      delete reader;
    }
    m_data_readers.clear();
    for (auto&& reader : other.m_data_readers) {
      m_data_readers.push_back(reader->copy());
    }
    return *this;
  }
  ~generic_compound_data_reader() override
  {
    for (auto&& reader : m_data_readers) {
      delete reader;
    }
  }
  generic_compound_data_reader* copy() const override = 0;

  //************************************************************************
  /// Apply operations to subsidiary data readers
  //************************************************************************
  void set_execution_mode_split_fraction(execution_mode m, double s) override
  {
    generic_data_reader::set_execution_mode_split_fraction(m, s);
    /// Don't propagate the validation fraction to subsidiary readers
    /// The fraction is applied at the top level
    for (auto&& reader : m_data_readers) {
      reader->set_execution_mode_split_fraction(m, 0);
    }
  }

  void set_role(std::string role) override
  {
    generic_data_reader::set_role(role);
    for (auto&& reader : m_data_readers) {
      reader->set_role(role);
    }
  }

  /// needed to support data_store_merge_samples
  std::vector<generic_data_reader*>& get_data_readers()
  {
    return m_data_readers;
  }

  bool has_labels() const override
  {
    for (auto&& reader : m_data_readers) {
      return reader->has_labels();
    }
    return false;
  }

  bool has_responses() const override
  {
    for (auto&& reader : m_data_readers) {
      return reader->has_responses();
    }
    return false;
  }

  void set_has_labels(const bool b) override
  {
    for (auto&& reader : m_data_readers) {
      reader->set_has_labels(b);
    }
  }
  /// Whether or not a data reader has a response field
  void set_has_responses(const bool b) override
  {
    for (auto&& reader : m_data_readers) {
      reader->set_has_responses(b);
    }
  }

  //************************************************************************

protected:
  /// List of readers providing data.
  std::vector<generic_data_reader*> m_data_readers;
};

} // namespace lbann

#endif // LBANN_GENERIC_COMPOUND_DATA_READER_HPP
