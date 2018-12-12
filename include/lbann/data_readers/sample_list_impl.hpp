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
////////////////////////////////////////////////////////////////////////////////

#ifndef __SAMPLE_LIST_IMPL_HPP__
#define __SAMPLE_LIST_IMPL_HPP__

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <deque>
#include <regex>
#include <algorithm>
#include "lbann/utils/exception.hpp"
#include <mpi.h>

namespace lbann {

template <typename SN>
inline std::string sample_list<SN>::to_string(const std::string& s) {
  return s;
}

template <typename SN>
template <typename T>
inline std::string sample_list<SN>::to_string(const T v) {
  return std::to_string(v);
}


template <typename SN>
inline bool sample_list<SN>::set_num_partitions(size_t n) {
  if (n == 0) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: number of partitions must be a positive number ("
                          + std::to_string(n) + ")");
    return false;
  }
  clear();
  m_num_partitions = n;
  return true;
}

template <typename SN>
inline bool sample_list<SN>::load(const std::string& samplelist_file) {
  bool ok = true;
#if 1
  std::ifstream istr(samplelist_file);
  ok = get_samples_per_file(istr);
  istr.close();
#else
  std::ifstream ifs(samplelist_file);
  std::string samplelist((std::istreambuf_iterator<char>(ifs)),
                          std::istreambuf_iterator<char>());

  ok = get_samples_per_file(samplelist);
#endif
  ok = ok && get_sample_range_per_file();
  ok = ok && get_sample_range_per_part();
  return ok;
}

template <typename SN>
inline bool sample_list<SN>::load_from_string(const std::string& samplelist) {
  bool ok = true;
#if 1
  std::istringstream istr(samplelist);
  ok = get_samples_per_file(istr);
#else
  ok = get_samples_per_file(samplelist);
#endif
  ok = ok && get_sample_range_per_file();
  ok = ok && get_sample_range_per_part();
  return ok;
}

template <typename SN>
inline size_t sample_list<SN>::get_samples_per_file(std::istream& ifstr)
{
  if (!ifstr.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: unable to read from the data input stream");
  }

  std::string line;

  size_t total_num_samples = 0u;
  m_samples_per_file.clear();

  // If we know the number of data files, we can reserve the space here for vector
  // but not for list.
  // m_samples_per_file.reserve(num_files);

  while (getline(ifstr, line)) {
    std::stringstream sstr(line);
    std::string filename;

    sstr >> filename;

    m_samples_per_file.emplace_back();
    (m_samples_per_file.back()).first = filename;
    auto& samples_of_current_file = (m_samples_per_file.back()).second;

    sample_name_t sample_name;

    while (sstr >> sample_name) {
      samples_of_current_file.emplace_back(sample_name);
    }

    const size_t num_samples_of_current_file = samples_of_current_file.size();
    total_num_samples += num_samples_of_current_file;
  }

  return total_num_samples;
}


template <typename SN>
inline size_t sample_list<SN>::get_samples_per_file(const std::string& samplelist)
{ // This might be slower than the strean-based on, but require less memory space and copying.
  size_t total_num_samples = 0u;
  m_samples_per_file.clear();

  static const std::regex newline("[\\r|\\n]+[\\s]*");
  static const std::regex space("[^\\S]+");

  std::sregex_token_iterator line(samplelist.cbegin(), samplelist.cend(), newline, -1);
  std::sregex_token_iterator end_of_lines;

  //m_samples_per_file.reserve(line.size());

  for ( ; line != end_of_lines ; ++line ) {
    const std::string& linestr = line->str();
    std::sregex_token_iterator word(linestr.cbegin(), linestr.cend(), space, -1);
    std::sregex_token_iterator end_of_words;

    m_samples_per_file.emplace_back();
    (m_samples_per_file.back()).first = word->str();
    auto& samples_of_current_file = (m_samples_per_file.back()).second;

    for (++word ; word != end_of_words ; ++word ) {
      samples_of_current_file.emplace_back(word->str());
    }

    const size_t num_samples_of_current_file = samples_of_current_file.size();
    total_num_samples += num_samples_of_current_file;
  }

  return total_num_samples;
}


/**
 * Reads through m_samples_per_file, and populate m_sample_range_per_file
 * by the sequential id of the first sample in each sample file.
 * The last element of m_sample_range_per_file is the total number of samples.
 */
template <typename SN>
inline bool sample_list<SN>::get_sample_range_per_file() {
  // populates m_sample_range_per_file, and requires m_samples_per_file is loaded.
  if (m_samples_per_file.empty()) {
    return false;
  }

  m_sample_range_per_file.clear();
  m_sample_range_per_file.reserve(m_samples_per_file.size()+1u);
  m_sample_range_per_file.push_back(0u);

  size_t total_so_far = 0u;

  for (const auto slist: m_samples_per_file) {
    total_so_far += slist.second.size();
    m_sample_range_per_file.push_back(total_so_far);
  }
  return true;
}


template <typename SN>
inline bool sample_list<SN>::get_sample_range_per_part() {
  // Populates m_sample_range_per_part, requires the total number of samples
  // and number of partitions are known.
  // The former can be obtained once m_sample_range_per_file is populated
  // by calling get_sample_range_per_file()
  const size_t total = static_cast<size_t>(m_sample_range_per_file.back());
  const size_t one_more = total % m_num_partitions;
  const size_t min_per_partition = total/m_num_partitions;

  if (min_per_partition == 0u) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
          + " :: insufficient number of samples for each partition to have at least one.");
    return false;
  }

  m_sample_range_per_part.clear();
  m_sample_range_per_part.resize(m_num_partitions+1u);

  //#pragma omp parallel for
  for (size_t p = 0u; p < m_num_partitions; ++p) {
    const size_t r_start = min_per_partition * p + ((p >= one_more)? one_more : p);
    const size_t r_end = r_start + min_per_partition + ((p < one_more)? 1u : 0u);
    m_sample_range_per_part[p+1] = r_end;
  }

  return true;
}


template <typename SN>
inline bool sample_list<SN>::find_sample_files_of_part(size_t p, size_t& sf_begin, size_t& sf_end) const {
  // requires both m_sample_range_per_file and m_sample_range_per_part is populated
  if (p+1 >= m_sample_range_per_part.size()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: invalid partition id or uninitialized m_sample_range_per_part.");
    return false;
  }
  const sample_id_t sample_start = m_sample_range_per_part[p];
  const sample_id_t sample_end = m_sample_range_per_part[p+1];

  std::vector<sample_id_t>::const_iterator i_begin
    = std::upper_bound(m_sample_range_per_file.cbegin(), m_sample_range_per_file.cend(), sample_start);
  std::vector<sample_id_t>::const_iterator i_end
    = std::lower_bound(m_sample_range_per_file.cbegin(), m_sample_range_per_file.cend(), sample_end);

  sf_begin = std::distance(m_sample_range_per_file.cbegin(), i_begin) - 1u;
  sf_end = std::distance(m_sample_range_per_file.cbegin(), i_end) - 1u;

  if ((sample_start > sample_end) || (sf_begin > sf_end)) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: invalid sample or sample file range.");
    return false;
  }
  return true;
}


template <typename SN>
inline const typename sample_list<SN>::sample_files_t& sample_list<SN>::get_list() const {
  return m_samples_per_file;
}


template <typename SN>
inline bool sample_list<SN>::write(const std::string& out_filename) const {
  // Requires m_samples_per_file is populated.
  const size_t num_files = m_samples_per_file.size();
  std::ofstream ofstr(out_filename);

  if (!ofstr.good() || (m_samples_per_file.size() != num_files)) {
    return false;
  }

  typename sample_files_t::const_iterator it_samplefiles = m_samples_per_file.cbegin();

  for (size_t i = 0u; i < num_files; ++i, ++it_samplefiles) {
    const auto& samples_of_current_file = it_samplefiles->second;
    ofstr << it_samplefiles->first;
    for (const auto& sample : samples_of_current_file) {
      ofstr << ' ' << sample;
    }
    ofstr << std::endl;
  }

  ofstr.close();
  return true;
}


template <typename SN>
inline void sample_list<SN>::clear() {
  m_num_partitions = 1u;
  m_samples_per_file.clear();
  m_sample_range_per_file.clear();
  m_sample_range_per_part.clear();
}

/**
 * TODO: Instead of string, vector<unsigned char> might be a better choice.
 * as it will allow space compression for numeric sample names.
 */
template <typename SN>
inline bool sample_list<SN>::to_string(size_t p, std::string& sstr) const
{
  const size_t num_local_samples = m_sample_range_per_part[p+1] - m_sample_range_per_part[p];

  size_t sf_begin;
  size_t sf_end;

  // Find the range of sample files that covers the range of samples of the partition.
  find_sample_files_of_part(p, sf_begin, sf_end);

  typename sample_files_t::const_iterator it_sfl_begin = m_samples_per_file.cbegin();
  typename sample_files_t::const_iterator it_sfl_end = m_samples_per_file.cbegin();
  std::advance(it_sfl_begin, sf_begin);
  std::advance(it_sfl_end, sf_end);

  size_t s_begin = m_sample_range_per_part[p] - m_sample_range_per_file[sf_begin];
  size_t s_end = m_sample_range_per_part[p+1] - m_sample_range_per_file[sf_end];

  if (s_begin >= it_sfl_begin->second.size() || s_end > it_sfl_end->second.size()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: incorrect sample indices.");
    return false;
  }

  typename sample_files_t::const_iterator it_sfl = it_sfl_begin;
  typename samples_t::const_iterator it_s = (it_sfl->second).cbegin(); // sample name iterator
  std::advance(it_s, s_begin);
  const size_t b_s_begin = std::min((it_sfl_begin->second).size(), num_local_samples + s_begin);

  const size_t estimated_len = (sf_end - sf_begin) * (it_sfl_begin->first).size() +
    ((to_string(*it_s)).size() + 1u) * static_cast<size_t>(num_local_samples * 1.2);
  sstr.reserve(estimated_len);

  sstr += it_sfl->first;
  for (size_t s = s_begin; s < b_s_begin; ++s, ++it_s) {
    sstr += ' ' + to_string(*it_s);
  }
  sstr += '\n';

  if (sf_begin < sf_end) {
    for (size_t sf = sf_begin+1; sf < sf_end; ++sf) {
      sstr += (++it_sfl)->first;
      for (const auto& s : it_sfl->second) {
        sstr += ' ' + to_string(s);
      }
      sstr += '\n';
    }

    sstr += (++it_sfl)->first;
    typename samples_t::const_iterator it_ss = it_sfl->second.cbegin();
    for (size_t s = 0u; s < s_end; ++s, ++it_ss) {
      sstr += ' ' + to_string(*it_ss);
    }
    sstr += '\n';
  }

  std::cerr << "estimated size vs actual size: " << estimated_len << ' ' << sstr.size() << std::endl;
  std::cerr << "num samples: " << num_local_samples << " samples of rank " << p << std::endl;
  return true;
}


struct send_request {
  int m_receiver;
  MPI_Request m_mpi_request;
  std::shared_ptr<std::string> m_data;
  unsigned long m_buf_size;

  send_request() {
    m_data = std::make_shared<std::string>();
  }

  void set_receiver(int recv) {
    m_receiver = recv;
  }

  int get_receiver() const {
    return m_receiver;
  }

  MPI_Request& request() {
    return m_mpi_request;
  }

  std::string* data() const {
    return m_data.get();
  }

  unsigned long& size() {
    m_buf_size = static_cast<unsigned long>(m_data->size());
    return m_buf_size;
  }
};


inline void handle_mpi_error(int ierr) {
  int errclass, resultlen;;
  char err_buffer[MPI_MAX_ERROR_STRING];

  if (ierr != MPI_SUCCESS) {
    MPI_Error_class(ierr, &errclass);
    if (errclass == MPI_ERR_RANK) {
      fprintf(stderr, "Invalid rank used in MPI send call\n");
      MPI_Error_string(ierr, err_buffer, &resultlen);
      fprintf(stderr,err_buffer);
      MPI_Finalize();             /* abort*/
    }
  }
}


inline void distribute_sample_list(const sample_list<std::string>& sn,
                            std::string& my_samples,
                            lbann_comm& _comm) {
  int num_ranks = 1;
  int my_rank = 0;
  int root_rank = 0;
  int size_tag = 0;
  int data_tag = 1;

  // TODO: replace bare MPI calls with corresponding lbann_comm wrappers
  MPI_Comm comm = _comm.get_model_comm().comm;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &num_ranks);
  MPI_Errhandler_set(comm, MPI_ERRORS_RETURN);

  if (my_rank == root_rank) {

    std::deque< send_request > send_requests;

    // Start of serialization and transmission
    MPI_Barrier(comm);

    for(int i = 0; i < num_ranks; ++i) {
      if (i == root_rank) {
        sn.to_string(static_cast<size_t>(root_rank), my_samples);
        continue;
      }

      send_requests.emplace_back();
      auto& req0 = send_requests.back();
      send_requests.emplace_back();
      auto& req1 = send_requests.back();
      req0.set_receiver(i);
      req1.set_receiver(i);
      std::string& sstr = *(req1.data());

      sn.to_string(static_cast<size_t>(i), sstr);
      unsigned long& bufsize = req1.size();

      int ierr;
      ierr = MPI_Isend(reinterpret_cast<void*>(&bufsize), 1,
                       MPI_UNSIGNED_LONG, i, size_tag, comm, &(req0.request()));
      handle_mpi_error(ierr);

      ierr = MPI_Isend(reinterpret_cast<void*>(const_cast<char*>(sstr.data())), static_cast<int>(sstr.size()),
                       MPI_BYTE, i, data_tag, comm, &(req1.request()));
      handle_mpi_error(ierr);

      const int n_prev_reqs = static_cast<int>(send_requests.size() - 2);

      for (int j = 0; j < n_prev_reqs; ++j) {
        MPI_Status status;
        int flag;
        auto& req = send_requests.front();

        MPI_Test(&(req.request()), &flag, &status);

        if (!flag) {
          break;
        }
        send_requests.pop_front();
      }
    }

    for (auto& req: send_requests) {
      MPI_Status status;
      MPI_Wait(&(req.request()), &status);
    }

    send_requests.clear();
  } else {
    // Start of serialization and transmission
    MPI_Barrier(comm);

    MPI_Status status;
    int ierr;
    unsigned long bufsize = 0u;
    ierr = MPI_Recv(reinterpret_cast<void*>(&bufsize), 1,
                    MPI_UNSIGNED_LONG, root_rank, size_tag, comm, &status);
    handle_mpi_error(ierr);

    my_samples.resize(bufsize);

    ierr = MPI_Recv(reinterpret_cast<void*>(&my_samples[0]), static_cast<int>(bufsize),
                    MPI_BYTE, root_rank, data_tag, comm, &status);
    handle_mpi_error(ierr);
  }

  // End of serialization and transmission
  MPI_Barrier(MPI_COMM_WORLD);
}

} // end of namespace lbann

#endif // __SAMPLE_LIST_IMPL_HPP__
