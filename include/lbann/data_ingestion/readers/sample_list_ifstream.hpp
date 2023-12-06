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
#ifndef __SAMPLE_LIST_IFSTREAM_HPP__
#define __SAMPLE_LIST_IFSTREAM_HPP__

#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include "hdf5.h"
#include "lbann/utils/exception.hpp"
#include "sample_list_open_files.hpp"
#include <fstream>

namespace lbann {

template <typename sample_name_t>
class sample_list_ifstream
  : public sample_list_open_files<sample_name_t, std::ifstream*>
{
public:
  using file_handle_t = std::ifstream*;
  using typename sample_list_open_files<sample_name_t,
                                        std::ifstream*>::sample_file_id_t;
  using
    typename sample_list_open_files<sample_name_t, std::ifstream*>::sample_t;
  using
    typename sample_list_open_files<sample_name_t, std::ifstream*>::samples_t;
  using typename sample_list_open_files<sample_name_t,
                                        std::ifstream*>::file_id_stats_t;
  using typename sample_list_open_files<sample_name_t,
                                        std::ifstream*>::file_id_stats_v_t;
  using typename sample_list_open_files<sample_name_t,
                                        std::ifstream*>::fd_use_map_t;

  sample_list_ifstream();
  ~sample_list_ifstream() override;

  bool is_file_handle_valid(const file_handle_t& h) const override;

protected:
  void
  obtain_sample_names(file_handle_t& h,
                      std::vector<std::string>& sample_names) const override;
  std::ifstream* open_file_handle_for_read(const std::string& path) override;
  void close_file_handle(file_handle_t& h) override;
  void clear_file_handle(file_handle_t& h) override;
};

template <typename sample_name_t>
inline sample_list_ifstream<sample_name_t>::sample_list_ifstream()
  : sample_list_open_files<sample_name_t, std::ifstream*>()
{}

template <typename sample_name_t>
inline sample_list_ifstream<sample_name_t>::~sample_list_ifstream()
{
  // Close the existing open files
  for (auto& f : this->m_file_id_stats_map) {
    file_handle_t& h = std::get<1>(f);
    close_file_handle(h);
    clear_file_handle(h);
    std::get<2>(f).clear();
  }
  this->m_file_id_stats_map.clear();
}

template <typename sample_name_t>
inline void sample_list_ifstream<sample_name_t>::obtain_sample_names(
  file_handle_t& h,
  std::vector<std::string>& sample_names) const
{
  // dah - I can't find anyplace where this method is called, and there's no
  //       easy implementation, so am ignoring; an exception will be thrown
  //       in case it's ever needed
  LBANN_ERROR("obtain_sample_names not implemented for std::ifstream*");
}

template <typename sample_name_t>
inline bool sample_list_ifstream<sample_name_t>::is_file_handle_valid(
  const file_handle_t& h) const
{
  if (h == nullptr) {
    return false;
  }
  return h->is_open();
}

template <typename sample_name_t>
inline std::ifstream*
sample_list_ifstream<sample_name_t>::open_file_handle_for_read(
  const std::string& file_path)
{
  std::ifstream* istrm = new std::ifstream(file_path.c_str());
  return istrm;
}

template <typename sample_name_t>
inline void
sample_list_ifstream<sample_name_t>::close_file_handle(file_handle_t& h)
{
  if (is_file_handle_valid(h)) {

    h->close();
  }
  delete h;
}

template <>
inline std::ifstream* uninitialized_file_handle<std::ifstream*>()
{
  return nullptr;
}

template <typename sample_name_t>
inline void
sample_list_ifstream<sample_name_t>::clear_file_handle(file_handle_t& h)
{
  // dah - don't think anything needs to be done here ...
}

} // end of namespace lbann

#endif // __SAMPLE_LIST_IFSTREAM_HPP__
