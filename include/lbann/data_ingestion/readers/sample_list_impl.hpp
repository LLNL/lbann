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

#ifndef LBANN_DATA_READERS_SAMPLE_LIST_IMPL_HPP
#define LBANN_DATA_READERS_SAMPLE_LIST_IMPL_HPP

#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <unistd.h>

#include "lbann/comm_impl.hpp"
#include "lbann/data_ingestion/readers/sample_list.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/file_utils.hpp"
#include "lbann/utils/serialize.hpp"

// Add cereal files not included in lbann/utils/serialize.hpp
#include <cereal/types/deque.hpp>
#include <cereal/types/tuple.hpp>

#include <zstr.hpp>

namespace lbann {

template <typename T>
inline std::string to_string(const T val)
{
  return std::to_string(val);
}

template <>
inline std::string to_string(const std::string val)
{
  return val;
}

template <typename sample_name_t>
inline auto to_sample_name_t(const std::string& sn_str)
  -> decltype(sample_name_t())
{
  LBANN_ERROR(std::string{} +
              " :: string conversion is not implement for the sample_name_t");
  return sample_name_t();
}

template <>
inline int to_sample_name_t<int>(const std::string& sn_str)
{
  return std::stoi(sn_str);
}

template <>
inline long to_sample_name_t<long>(const std::string& sn_str)
{
  return std::stol(sn_str);
}

template <>
inline unsigned long to_sample_name_t<unsigned long>(const std::string& sn_str)
{
  return std::stoul(sn_str);
}

template <>
inline long long to_sample_name_t<long long>(const std::string& sn_str)
{
  return std::stoll(sn_str);
}

template <>
inline unsigned long long
to_sample_name_t<unsigned long long>(const std::string& sn_str)
{
  return std::stoull(sn_str);
}

template <>
inline float to_sample_name_t<float>(const std::string& sn_str)
{
  return std::stof(sn_str);
}

template <>
inline double to_sample_name_t<double>(const std::string& sn_str)
{
  return std::stod(sn_str);
}

template <>
inline long double to_sample_name_t<long double>(const std::string& sn_str)
{
  return std::stold(sn_str);
}

template <>
inline std::string to_sample_name_t<std::string>(const std::string& sn_str)
{
  return sn_str;
}

//------------------------
//   sample_list_header
//------------------------

inline sample_list_header::sample_list_header()
  : m_is_multi_sample(false),
    m_is_exclusive(false),
    m_no_label_header(false),
    m_included_sample_count(0u),
    m_excluded_sample_count(0u),
    m_num_files(0u),
    m_file_dir(""),
    m_sample_list_name(""),
    m_label_filename("")
{}

template <class Archive>
void sample_list_header::serialize(Archive& ar)
{
  ar(m_is_multi_sample,
     m_is_exclusive,
     m_no_label_header,
     m_included_sample_count,
     m_excluded_sample_count,
     m_num_files,
     m_file_dir,
     m_sample_list_name,
     m_label_filename);
}

inline void sample_list_header::set_sample_list_type(const std::string& line1)
{
  std::stringstream header1(line1);
  std::string sample_list_type;
  header1 >> sample_list_type;

  std::for_each(sample_list_type.begin(), sample_list_type.end(), [](char& c) {
    c = std::toupper(c);
  });

  m_is_multi_sample = false;
  m_is_exclusive = false;
  m_no_label_header = false;
  m_has_unused_sample_fields = true;

  if (sample_list_type == single_sample) {
  }
  else if (sample_list_type == multi_sample_inclusion ||
           sample_list_type == multi_sample_inclusion_v2) {
    m_is_multi_sample = true;
    m_is_exclusive = false;
    if (sample_list_type == multi_sample_inclusion_v2) {
      m_no_label_header = true;
      m_has_unused_sample_fields = false;
    }
  }
  else if (sample_list_type == multi_sample_exclusion) {
    m_is_multi_sample = true;
    m_is_exclusive = true;
  }
  else if (sample_list_type == conduit_hdf5_inclusion) {
    // For backward compatibility
    m_is_multi_sample = true;
    m_is_exclusive = false;
    m_no_label_header = true; // old format does not use a line for label file
  }
  else if (sample_list_type == conduit_hdf5_exclusion) {
    // For backward compatibility
    m_is_multi_sample = true;
    m_is_exclusive = true;
    m_no_label_header = true;
  }
  else {
    LBANN_ERROR("Unknown sample list type: ", sample_list_type);
  }
}

inline void sample_list_header::set_sample_count(const std::string& line2)
{
  std::stringstream header2(line2);
  if (m_is_multi_sample) {
    header2 >> m_included_sample_count;
    if (m_has_unused_sample_fields) {
      header2 >> m_excluded_sample_count;
    }
    else {
      m_excluded_sample_count = 0ul;
    }
  }
  header2 >> m_num_files;

  if (!m_is_multi_sample) {
    m_included_sample_count = m_num_files;
    m_excluded_sample_count = 0ul;
  }
}

inline void sample_list_header::set_data_file_dir(const std::string& line3)
{
  std::stringstream header3(line3);
  header3 >> m_file_dir;
}

inline void sample_list_header::set_label_filename(const std::string& line4)
{
  std::stringstream header4(line4);
  header4 >> m_label_filename;
}

inline bool sample_list_header::is_multi_sample() const
{
  return m_is_multi_sample;
}

inline bool sample_list_header::is_exclusive() const { return m_is_exclusive; }

inline bool sample_list_header::use_label_header() const
{
  return !m_no_label_header;
}

inline bool sample_list_header::has_unused_sample_fields() const
{
  return m_has_unused_sample_fields;
}

inline size_t sample_list_header::get_sample_count() const
{
  return m_included_sample_count;
}

inline size_t sample_list_header::get_num_files() const { return m_num_files; }

inline const std::string& sample_list_header::get_file_dir() const
{
  return m_file_dir;
}

inline const std::string& sample_list_header::get_sample_list_name() const
{
  return m_sample_list_name;
}

inline void sample_list_header::set_sample_list_name(const std::string& n)
{
  m_sample_list_name = n;
}

inline const std::string& sample_list_header::get_label_filename() const
{
  return m_label_filename;
}

//------------------
//   sample_list
//------------------

template <typename sample_name_t>
sample_list<sample_name_t>::sample_list()
  : m_stride(1ul), m_keep_order(true), m_check_data_file(false)
{}

template <typename sample_name_t>
sample_list<sample_name_t>::~sample_list()
{}

template <typename sample_name_t>
sample_list<sample_name_t>::sample_list(const sample_list& rhs)
{
  copy_members(rhs);
}

template <typename sample_name_t>
inline sample_list<sample_name_t>&
sample_list<sample_name_t>::operator=(const sample_list& rhs)
{
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  copy_members(rhs);

  return (*this);
}

template <typename sample_name_t>
inline sample_list<sample_name_t>&
sample_list<sample_name_t>::copy(const sample_list& rhs)
{
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  copy_members(rhs);

  return (*this);
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::copy_members(const sample_list& rhs)
{
  m_header = rhs.m_header;
  m_stride = rhs.m_stride;
  m_keep_order = rhs.m_keep_order;
  m_check_data_file = rhs.m_check_data_file;
  m_sample_list = rhs.m_sample_list;

  /// Keep track of existing filenames
  m_file_id_stats_map = rhs.m_file_id_stats_map;
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::load(std::istream& istrm,
                                             size_t stride,
                                             size_t offset)
{
  m_stride = stride;
  get_samples_per_file(istrm, stride, offset);
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::load(const std::string& samplelist_file,
                                             const lbann_comm& comm,
                                             bool interleave)
{
  m_header.set_sample_list_name(samplelist_file);
  zstr::ifstream istrm(samplelist_file);
  // std::ifstream istrm(samplelist_file);
  load(istrm, comm, interleave);
  // zstr doesn't provide a close; odd but true
  // istrm.close();
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::load(std::istream& istrm,
                                             const lbann_comm& comm,
                                             bool interleave)
{
  const size_t stride = interleave ? comm.get_procs_per_trainer() : 1ul;
  const size_t offset = interleave ? comm.get_rank_in_trainer() : 0ul;
  load(istrm, stride, offset);
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::load(const sample_list_header& header,
                                             std::istream& istrm,
                                             const lbann_comm& comm,
                                             bool interleave)
{
  m_header = header;
  const size_t stride = interleave ? comm.get_procs_per_trainer() : 1ul;
  const size_t offset = interleave ? comm.get_rank_in_trainer() : 0ul;

  m_stride = stride;
  read_sample_list(istrm, stride, offset);
}

template <typename sample_name_t>
inline void
sample_list<sample_name_t>::load_from_string(const std::string& samplelist,
                                             const lbann_comm& comm,
                                             bool interleave)
{
  m_header.set_sample_list_name("<LOAD_FROM_STRING>");
  std::istringstream istrm(samplelist);
  const size_t stride = interleave ? comm.get_procs_per_trainer() : 1ul;
  const size_t offset = interleave ? comm.get_rank_in_trainer() : 0ul;
  m_stride = stride;
  load(istrm, stride, offset);
}

template <typename sample_name_t>
inline size_t sample_list<sample_name_t>::size() const
{
  return m_sample_list.size();
}

template <typename sample_name_t>
inline size_t sample_list<sample_name_t>::get_num_files() const
{
  return m_file_id_stats_map.size();
}

template <typename sample_name_t>
inline bool sample_list<sample_name_t>::empty() const
{
  return (size() == 0ul);
}

template <typename sample_name_t>
inline std::string
sample_list<sample_name_t>::read_header_line(std::istream& istrm,
                                             const std::string& listname,
                                             const std::string& info)
{
  if (!istrm.good()) {
    LBANN_ERROR("unable to read the header line of sample list ",
                listname,
                " for ",
                info,
                " because !istrm.good()");
  }

  std::string line;
  std::getline(istrm, line);

  if (line.empty()) {
    LBANN_ERROR("unable to read the header line of sample list ",
                listname,
                " for ",
                info,
                " -- the line was empty");
  }
  return line;
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::read_header(std::istream& istrm)
{
  const std::string listname = m_header.get_sample_list_name();

  std::string line1 = read_header_line(istrm, listname, "the exclusiveness\n");
  std::string line2 =
    read_header_line(istrm,
                     listname,
                     "the number of samples and the number of files\n");
  std::string line3 =
    read_header_line(istrm, listname, "the data file directory\n");

  m_header.set_sample_list_type(line1);
  m_header.set_sample_count(line2);
  m_header.set_data_file_dir(line3);

  if (m_header.use_label_header()) {
    std::string line4 =
      read_header_line(istrm, listname, "the path to label/response file\n");
    m_header.set_label_filename(line4);
  }

  if (m_header.get_file_dir().empty() ||
      (m_check_data_file && !check_if_dir_exists(m_header.get_file_dir()))) {
    LBANN_ERROR(std::string{} + "file " + listname +
                " :: data root directory '" + m_header.get_file_dir() +
                "' does not exist.");
  }
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::read_sample_list(std::istream& istrm,
                                                         size_t stride,
                                                         size_t offset)
{
  m_sample_list.reserve(m_header.get_sample_count());

  const std::string whitespaces(" \t\f\v\n\r");
  size_t cnt_files = 0u;
  std::string line;

  while (std::getline(istrm, line)) {
    const size_t end_of_str = line.find_last_not_of(whitespaces);
    if (end_of_str == std::string::npos) { // empty line
      continue;
    }
    if (cnt_files++ >= m_header.get_num_files()) {
      break;
    }
    // Check to see if there is a strided load and skip the lines that are not
    // for this rank
    if ((cnt_files - 1) % stride != offset) {
      continue;
    }

    std::stringstream sstr(line.substr(
      0,
      end_of_str + 1)); // clear trailing spaces for accurate parsing
    std::string filename;

    sstr >> filename;

    const std::string file_path =
      add_delimiter(m_header.get_file_dir()) + filename;

    if (filename.empty() ||
        (m_check_data_file && !check_if_file_exists(file_path))) {
      LBANN_ERROR("data file '", file_path, "' does not exist.");
    }

    const sample_file_id_t index = m_file_id_stats_map.size();
    static const auto sn0 = uninitialized_sample_name<sample_name_t>();
    m_sample_list.emplace_back(std::make_pair(index, sn0));
    m_file_id_stats_map.emplace_back(filename);
  }

  if (m_header.get_num_files() != cnt_files) {
    LBANN_ERROR(std::string("Sample list number of files requested ") +
                std::to_string(m_header.get_num_files()) +
                std::string(" does not equal number of files loaded ") +
                std::to_string(cnt_files));
  }

  if (stride == 1 && m_header.get_sample_count() != m_sample_list.size()) {
    LBANN_ERROR(std::string("Sample list count ") +
                std::to_string(m_header.get_sample_count()) +
                std::string(" does not equal sample list size ") +
                std::to_string(m_sample_list.size()));
  }
}

template <typename sample_name_t>
inline size_t
sample_list<sample_name_t>::get_samples_per_file(std::istream& istrm,
                                                 size_t stride,
                                                 size_t offset)
{
  read_header(istrm);

  m_stride = stride;
  read_sample_list(istrm, stride, offset);

  return size();
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::all_gather_archive(
  const std::string& archive,
  std::vector<std::string>& gathered_archive,
  lbann_comm& comm)
{

  // there's commented out code below to deal with the case where
  // archive.size() > INT_MAX; but for now let's assume we won't
  // encounter that (which is true for the 100M JAG set)
  int constexpr max_int = std::numeric_limits<int>::max();
  size_t n = archive.size();
  if (n > max_int) {
    LBANN_ERROR("(n > max_int");
  }

  // change int to size_t for case where n > max_int (see commented out
  // code block below)
  int size_of_my_archive = archive.size();
  std::vector<int> packed_sizes(comm.get_procs_per_trainer());
  comm.trainer_all_gather(size_of_my_archive, packed_sizes);

  int me = comm.get_rank_in_trainer();
  int np = comm.get_procs_per_trainer();

  for (int p = 0; p < np; p++) {
    gathered_archive[p].resize(packed_sizes[p]);
    if (me == p) {
      gathered_archive[p] = archive;
    }
    int sz = packed_sizes[p];
    if (sz > INT_MAX) {
      LBANN_ERROR("packed_sizes[",
                  p,
                  "] size is: ",
                  sz,
                  "  which is larger than INMAX, hence, broacast doesn't work; "
                  "must be done in rounds");
    }
    char* data = const_cast<char*>(gathered_archive[p].data());
    comm.trainer_broadcast<char>(p, data, sz);
  }

#if 0
  std::vector<int> rounds;
  for (int p=0; p<np; p++) {
    std::string& buf = gathered_archive[p];
    buf.resize(packed_sizes[p]);

    rounds.clear();
    int n = packed_sizes[p]/INT_MAX;
    if (n < 0) {
      LBANN_ERROR("(n < 0; that shouldn't be possible; there's a bug; n: ", n, " packed_sizes[p]: ", packed_sizes[p], " packed_sizes[p]/INT_MAX: ", n);
    }
    for (int k=0; k<n; k++) {
      rounds.push_back(INT_MAX);
    }
    int remainder = packed_sizes[p] - (n*INT_MAX);
    rounds.push_back(remainder);

    if (p != me) {
      gathered_archive[p].resize(packed_sizes[p]);
    }
    size_t offset = 0;
    for (size_t k=0; k<rounds.size(); k++) {
      if (me == p) {
        char *data = const_cast<char*>(archive.data() + offset);
        comm.trainer_broadcast<char>(p, data, rounds[k]);
      } else {
        char *data = const_cast<char*>(gathered_archive[p].data() + offset);
        comm.trainer_broadcast<char>(p, data, rounds[k]);
      }
      offset += rounds[k];
    }
  }
#endif

  return;
}

template <typename sample_name_t>
template <typename T>
inline size_t
sample_list<sample_name_t>::all_gather_field(T data,
                                             std::vector<T>& gathered_data,
                                             lbann_comm& comm)
{
  std::string archive;
  std::ostringstream ss;
  {
    cereal::BinaryOutputArchive oarchive(ss);
    oarchive(data);
  } // archive goes out of scope, ensuring all contents are flushed
  archive = ss.str();

  std::vector<std::string> gathered_archive(comm.get_procs_per_trainer());

  all_gather_archive(archive, gathered_archive, comm);

  std::vector<T> per_rank_data(comm.get_procs_per_trainer());

  size_t gathered_field_size = 0;
  for (size_t i = 0u; i < gathered_archive.size(); ++i) {
    std::string& buf = gathered_archive[i];
    T& tmp = gathered_data[i];

    std::stringstream in_ss(buf);
    cereal::BinaryInputArchive iarchive(in_ss);
    iarchive(tmp);
    gathered_field_size += tmp.size();
  }
  return gathered_field_size;
}

template <typename sample_name_t>
template <class Archive>
void sample_list<sample_name_t>::serialize(Archive& ar)
{
  ar(m_header, m_sample_list, m_file_id_stats_map);
  // The member variables that are only meaningful during initial loading
  // are not included here.
  // e.g., m_stride, m_keep_order, m_check_data_file
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::write_header(std::string& sstr,
                                                     size_t num_files) const
{
  // The first line indicate if the list is single-sample-per-file type,
  // multi-sample-exclusive or multi-sample-inclusive.
  // The second line contains the number of samples (included and excluded
  // when applicable), as well as the number of files.
  // The third line contains the root data file directory.
  // The fourth line contains the path to the label file when applicable

  if (m_header.is_multi_sample()) {
    if (m_header.use_label_header()) {
      sstr += (m_header.is_exclusive() ? multi_sample_exclusion + "\n"
                                       : multi_sample_inclusion + "\n");
    }
    else {
      if (m_header.has_unused_sample_fields()) {
        sstr += (m_header.is_exclusive() ? conduit_hdf5_exclusion + "\n"
                                         : conduit_hdf5_inclusion + "\n");
      }
      else {
        if (m_header.is_exclusive()) {
          LBANN_ERROR("Unknown header format");
        }
        sstr += multi_sample_inclusion_v2 + "\n";
      }
    }
    size_t total, included, excluded;
    get_num_samples(total, included, excluded);

    sstr += std::to_string(included);
    if (m_header.has_unused_sample_fields()) {
      sstr += ' ' + std::to_string(excluded);
    }
    sstr += ' ' + std::to_string(num_files) + '\n';
  }
  else {
    sstr += single_sample + "\n";
    sstr += std::to_string(num_files) + '\n';
  }
  sstr += m_header.get_file_dir() + '\n';
  if (m_header.use_label_header()) {
    sstr += m_header.get_label_filename() + '\n';
  }
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::get_num_samples(size_t& total,
                                                        size_t& included,
                                                        size_t& excluded) const
{
  total = size();
  included = size();
  excluded = 0ul;
}

template <typename sample_name_t>
inline bool sample_list<sample_name_t>::to_string(std::string& sstr) const
{
  size_t total_len = 0ul;
  for (const auto& s : m_sample_list) {
    const std::string& filename = m_file_id_stats_map[s.first];
    total_len += filename.size() + 1u;
  }

  sstr.clear();

  static const size_t max_type_len = std::max(
    std::max(multi_sample_exclusion.size(), multi_sample_inclusion.size()),
    single_sample.size());

  static const size_t max_num_len =
    std::to_string(std::numeric_limits<size_t>::max()).size();

  // reserve the string to hold the entire sample list
  size_t estimated_len =
    max_type_len + max_num_len + 2 + m_header.get_file_dir().size() +
    m_header.get_label_filename().size() + 4 // sizeof('\n') * 4
    + total_len + 1000;
  sstr.reserve(estimated_len);

  // write the list header
  write_header(sstr, get_num_files());

  // write the list body
  for (const auto& s : m_sample_list) {
    // File name
    const std::string& filename = m_file_id_stats_map[s.first];
    sstr += filename + '\n';
  }

  return true;
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::write(const std::string filename) const
{
  std::string dir, basename;
  parse_path(filename, dir, basename);
  if (!dir.empty() && !check_if_dir_exists(dir)) {
    // The creation of a shared directory must be done once in a coordinated
    // fashion among the entities that have access to it. Thus, it must be done
    // in advance
    std::cerr << "The sample list output directory (" + dir + ") does not exist"
              << std::endl;
    return;
  }

  std::fstream ofs(filename, std::fstream::out | std::fstream::binary);

  if (!ofs.good()) {
    return;
  }

  std::string buf;
  to_string(buf);

  ofs.write(buf.data(), buf.size() * sizeof(std::string::value_type));
  ofs.close();
}

template <typename sample_name_t>
inline const typename sample_list<sample_name_t>::samples_t&
sample_list<sample_name_t>::get_list() const
{
  return m_sample_list;
}

template <typename sample_name_t>
inline const sample_list_header& sample_list<sample_name_t>::get_header() const
{
  return m_header;
}

template <typename sample_name_t>
inline const typename sample_list<sample_name_t>::sample_t&
sample_list<sample_name_t>::operator[](size_t idx) const
{
  return m_sample_list[idx];
}

template <typename sample_name_t>
inline const std::string&
sample_list<sample_name_t>::get_samples_filename(sample_file_id_t id) const
{
  return m_file_id_stats_map[id];
}

template <typename sample_name_t>
inline const std::string&
sample_list<sample_name_t>::get_samples_dirname() const
{
  return m_header.get_file_dir();
}

template <typename sample_name_t>
inline const std::string& sample_list<sample_name_t>::get_label_filename() const
{
  return m_header.get_label_filename();
}

template <typename sample_name_t>
inline void
sample_list<sample_name_t>::set_samples_filename(sample_file_id_t id,
                                                 const std::string& filename)
{
  m_file_id_stats_map[id] = filename;
}

#if defined(__cpp_if_constexpr) // c++17
template <typename sample_name_t>
inline void sample_list<sample_name_t>::assign_samples_name()
{
  if constexpr (std::is_integral<sample_name_t>::value &&
                !std::is_same<sample_name_t, bool>::value) {
    sample_name_t i = 0;
    for (auto& s : m_sample_list) {
      s.second = i++;
    }
  }
  else if constexpr (std::is_same<std::string, sample_name_t>::value) {
    for (auto& s : m_sample_list) {
      s.second = get_samples_filename(s.first);
    }
  }
  else {
    LBANN_ERROR(std::string{} +
                " :: base class does not implement this method" +
                " for the current sample name type");
  }
}

template <typename sample_name_t>
inline sample_name_t uninitialized_sample_name()
{
  if constexpr (std::is_integral<sample_name_t>::value) {
    return static_cast<sample_name_t>(0);
  }
  else if constexpr (std::is_same<std::string, sample_name_t>::value) {
    return "";
  }
  else if constexpr (std::is_floating_point<sample_name_t>::value) {
    return 0.0;
  }
  else if constexpr (std::is_default_constructible<sample_name_t>::value &&
                     std::is_copy_constructible<sample_name_t>::value) {
    sample_name_t ret{};
    return ret;
  }
  else {
    LBANN_ERROR(std::string{} +
                " :: base class does not implement this method" +
                " for the current sample name type");
  }
}
#else
template <>
inline void sample_list<size_t>::assign_samples_name()
{
  size_t i = 0ul;
  for (auto& s : m_sample_list) {
    s.second = i++;
  }
}

template <>
inline void sample_list<std::string>::assign_samples_name()
{
  for (auto& s : m_sample_list) {
    s.second = get_samples_filename(s.first);
  }
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::assign_samples_name()
{
  LBANN_ERROR(std::string{} + " :: base class does not implement this method" +
              " for the current sample name type");
}

template <>
inline size_t uninitialized_sample_name<size_t>()
{
  return 0ul;
}

template <>
inline std::string uninitialized_sample_name<std::string>()
{
  return "";
}

template <typename sample_name_t>
inline sample_name_t uninitialized_sample_name()
{
  sample_name_t ret{};
  return ret;
}
#endif // defined(__cpp_if_constexpr)

template <typename sample_name_t>
inline void
sample_list<sample_name_t>::all_gather_packed_lists(lbann_comm& comm)
{
  std::cerr
    << "starting sample_list<sample_name_t> ::all_gather_packed_lists\n";
  int num_ranks = comm.get_procs_per_trainer();
  typename std::vector<samples_t> per_rank_samples(num_ranks);
  typename std::vector<std::vector<std::string>> per_rank_files(num_ranks);

  size_t num_samples = all_gather_field(m_sample_list, per_rank_samples, comm);
  std::cout << "DONE! num_samples: " << num_samples << std::endl;
  size_t num_ids = all_gather_field(m_file_id_stats_map, per_rank_files, comm);
  std::cout << "DONE! num_ids: " << num_ids << std::endl;

  m_sample_list.clear();
  m_file_id_stats_map.clear();

  m_sample_list.reserve(num_samples);
  m_file_id_stats_map.reserve(num_ids);

  for (int r = 0; r < num_ranks; r++) {

    const samples_t& s_list = per_rank_samples[r];
    const auto& files = per_rank_files[r];
    for (const auto& s : s_list) {
      sample_file_id_t index = s.first;
      const std::string& filename = files[index];
      if (index >= m_file_id_stats_map.size() ||
          (m_file_id_stats_map.back() != filename)) {
        index = m_file_id_stats_map.size();
        m_file_id_stats_map.emplace_back(filename);
      }
      else {
        for (size_t i = 0; i < m_file_id_stats_map.size(); i++) {
          if (filename == m_file_id_stats_map[i]) {
            index = i;
            break;
          }
        }
      }
      static const auto sn0 = uninitialized_sample_name<sample_name_t>();
      m_sample_list.emplace_back(std::make_pair(index, sn0));
    }
  }

  if (m_keep_order) {
    reorder();
  }

  assign_samples_name();

  return;
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::reorder()
{
  if (m_stride > 1ul) { // undo interleaving
    const size_t sz = m_sample_list.size();
    const size_t s = sz / m_stride;
    const size_t s_more = (sz + m_stride - 1ul) / m_stride;
    const size_t n_more = sz - s * m_stride;

    samples_t tmp_sample_list;
    tmp_sample_list.reserve(s_more * m_stride);

    for (size_t i = 0ul; i < s_more; ++i) {
      for (size_t j = i, k = 0ul; j < sz; ++k) {
        tmp_sample_list.push_back(m_sample_list[j]);
        // if (tmp_sample_list.size() == sz) break;
        j += ((k < n_more) ? s_more : s);
      }
    }
    tmp_sample_list.resize(sz);
    std::swap(m_sample_list, tmp_sample_list);
    m_stride = 1ul;
  }
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::build_sample_map_from_name_to_index()
{
  m_map_name_to_idx.clear();
  for (size_t i = 0ul; i < m_sample_list.size(); ++i) {
    m_map_name_to_idx.insert(std::make_pair(m_sample_list[i].second, i));
  }
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::clear_sample_map_from_name_to_index()
{
  m_map_name_to_idx.clear();
  m_map_name_to_idx.rehash(0);
  sample_map_t tmp;
  tmp.rehash(0);
  tmp.swap(m_map_name_to_idx);
}

template <typename sample_name_t>
inline typename sample_list<sample_name_t>::sample_idx_t
sample_list<sample_name_t>::get_sample_index(const sample_name_t& sn)
{
  typename sample_map_t::const_iterator it = m_map_name_to_idx.find(sn);
  if (it == m_map_name_to_idx.cend()) {
    return size();
    // LBANN_ERROR(" :: cannot find the sample name ", lbann::to_string(sn));
  }
  return it->second;
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::keep_sample_order(bool keep)
{
  m_keep_order = keep;
}

template <typename sample_name_t>
inline void
sample_list<sample_name_t>::set_sample_list_name(const std::string& n)
{
  m_header.set_sample_list_name(n);
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::set_data_file_check()
{
  m_check_data_file = true;
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>::unset_data_file_check()
{
  m_check_data_file = false;
}

} // end of namespace lbann

#endif // LBANN_DATA_READERS_SAMPLE_LIST_IMPL_HPP
