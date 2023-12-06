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

#ifndef LBANN_DATA_READER_SAMPLE_LIST_HPP
#define LBANN_DATA_READER_SAMPLE_LIST_HPP

// #include "lbann_config.hpp"
#include "lbann/data_ingestion/data_reader.hpp"
#include <conduit/conduit.hpp>

namespace lbann {

/**
 * Base class for all readers that employ sample lists
 */
template <typename SampleListT>
class data_reader_sample_list : public generic_data_reader
{
public:
  using sample_name_type = typename SampleListT::name_t;
  using sample_file_id_type = typename SampleListT::sample_file_id_t;
  using sample_type = std::pair<sample_file_id_type, sample_name_type>;
  using file_handle_type = typename SampleListT::file_handle_t;

  data_reader_sample_list(bool shuffle = true);
  data_reader_sample_list(const data_reader_sample_list&);
  data_reader_sample_list& operator=(const data_reader_sample_list&);
  ~data_reader_sample_list() override{};
  data_reader_sample_list* copy() const override
  {
    return new data_reader_sample_list(*this);
  }
  void copy_members(const data_reader_sample_list& rhs);

  std::string get_type() const override { return "data_reader_sample_list"; }

  /** @brief Open the file and get the sample name for the given index.
   *  @returns A pair containing the file handle and the name of the
   *           sample.
   */
  std::pair<file_handle_type, sample_name_type> open_file(size_t index);
  void close_file(size_t index_in);

  /**
   * Override the shuffle indices function to update the sample list's
   * file usage.
   */
  void shuffle_indices(rng_gen& gen) override;

  /** Developer's note: derived classes that override load() should
   * explicitly call data_reader_sample_list::load() at the
   * beginning of their method load() method
   */
  void load() override;

  SampleListT& get_sample_list() { return m_sample_list; }

  sample_type get_sample(size_t index) { return m_sample_list[index]; }

protected:
  SampleListT m_sample_list;

  void load_list_of_samples(const std::string sample_list_file);

  void
  load_list_of_samples_from_archive(const std::string& sample_list_archive);

}; // class data_reader_sample_list

} // end of namespace lbann

#endif // LBANN_DATA_READER_SAMPLE_LIST_HPP
