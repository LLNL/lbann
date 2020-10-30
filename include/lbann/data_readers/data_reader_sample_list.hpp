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

#ifndef _DATA_READER_SAMPLE_LIST_HPP_
#define _DATA_READER_SAMPLE_LIST_HPP_

#include "lbann_config.hpp"
#include "data_reader.hpp"
#include "conduit/conduit.hpp"

#ifdef _USE_IO_HANDLE_
#include "lbann/data_readers/sample_list_conduit_io_handle.hpp"
#else
#include "lbann/data_readers/sample_list_hdf5.hpp"
#endif

namespace lbann {

/**
 * Base class for all readers that employ sample lists
 */
class data_reader_sample_list : public generic_data_reader {
 public:

  using sample_name_t = std::string;
#ifdef _USE_IO_HANDLE_
  using sample_list_t = sample_list_conduit_io_handle<sample_name_t>;
#else
  using sample_list_t = sample_list_hdf5<sample_name_t>;
#endif
  using sample_t = std::pair<sample_list_t::sample_file_id_t, sample_name_t>;

  data_reader_sample_list(bool shuffle = true);
  data_reader_sample_list(const data_reader_sample_list&);
  data_reader_sample_list& operator=(const data_reader_sample_list&);
  ~data_reader_sample_list() override {};
  data_reader_sample_list* copy() const override { return new data_reader_sample_list(*this); }
  void copy_members(const data_reader_sample_list &rhs);

  std::string get_type() const override {
    return "data_reader_sample_list";
  }

  /** Developer's note: derived classes that override load() should
   * explicitly call data_reader_sample_list::load() at the
   * beginning of their method
   */
  void load() override;

 protected:

  sample_list_t m_sample_list;

  void load_list_of_samples(const std::string sample_list_file); 

  void load_list_of_samples_from_archive(const std::string& sample_list_archive);

}; // class data_reader_sample_list

} // end of namespace lbann

#endif // _DATA_READER_SAMPLE_LIST_HPP_
