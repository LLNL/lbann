#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <locale>
#include "lbann/utils/exception.hpp"
#include "lbann/utils/file_utils.hpp"
#include <deque>
#include "hdf5.h"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include <unordered_set>
#include <memory>
#include <type_traits>

#include <cereal/archives/binary.hpp>
#include <sstream>
#include <unistd.h>

namespace lbann {

template<typename T>
inline std::string to_string(const T val) {
  return std::to_string(val);
}

template<>
inline std::string to_string(const std::string val) {
  return val;
}

template <typename sample_name_t>
inline auto to_sample_name_t(const std::string& sn_str) -> decltype (sample_name_t()){
  // TODO: throw exception
  return sample_name_t();
}

template<> inline int to_sample_name_t<int>(const std::string& sn_str) {
  return std::stoi(sn_str);
}

template<> inline long to_sample_name_t<long>(const std::string& sn_str) {
  return std::stol(sn_str);
}

template<> inline unsigned long to_sample_name_t<unsigned long>(const std::string& sn_str) {
  return std::stoul(sn_str);
}

template<> inline long long to_sample_name_t<long long>(const std::string& sn_str) {
  return std::stoll(sn_str);
}

template<> inline unsigned long long to_sample_name_t<unsigned long long>(const std::string& sn_str) {
  return std::stoull(sn_str);
}

template<> inline float to_sample_name_t<float>(const std::string& sn_str) {
  return std::stof(sn_str);
}

template<> inline double to_sample_name_t<double>(const std::string& sn_str) {
  return std::stod(sn_str);
}

template<> inline long double to_sample_name_t<long double>(const std::string& sn_str) {
  return std::stold(sn_str);
}

template<> inline std::string to_sample_name_t<std::string>(const std::string& sn_str) {
  return sn_str;
}


inline sample_list_header::sample_list_header()
  : m_is_exclusive(false), m_included_sample_count(0u),
    m_excluded_sample_count(0u), m_num_files(0u),
    m_file_dir("") {
}

inline bool sample_list_header::is_exclusive() const {
  return m_is_exclusive;
}

inline size_t sample_list_header::get_sample_count() const {
  return m_included_sample_count;
}

inline size_t sample_list_header::get_num_files() const {
  return m_num_files;
}

inline const std::string& sample_list_header::get_sample_list_filename() const {
  return m_sample_list_filename;
}

inline const std::string& sample_list_header::get_file_dir() const {
  return m_file_dir;
}

template <typename file_handle_t, typename sample_name_t>
inline sample_list<file_handle_t, sample_name_t>::sample_list() {
  m_max_open_files = getdtablesize() - LBANN_MAX_OPEN_FILE_MARGIN;
}

template <typename file_handle_t, typename sample_name_t>
inline sample_list<file_handle_t, sample_name_t>::~sample_list() {
  // Close the existing open files
  for(auto& f : m_file_id_stats_map) {
    file_handle_t& h = std::get<1>(f);
    close_and_clear_file_handle(h);
  }
  m_file_id_stats_map.clear();
  m_open_fd_pq.clear();
}

template <typename file_handle_t, typename sample_name_t>
inline void sample_list<file_handle_t, sample_name_t>
::load(const std::string& samplelist_file,
       size_t stride, size_t offset) {
  std::ifstream istr(samplelist_file);
  get_samples_per_file(istr, samplelist_file, stride, offset);
  istr.close();
}

template <typename file_handle_t, typename sample_name_t>
inline sample_list_header sample_list<file_handle_t, sample_name_t>
::load_header(const std::string& samplelist_file) const {
  std::ifstream istr(samplelist_file);
  return read_header(istr, samplelist_file);
}

template <typename file_handle_t, typename sample_name_t>
inline void sample_list<file_handle_t, sample_name_t>
::load_from_string(const std::string& samplelist) {
  std::istringstream istr(samplelist);
  get_samples_per_file(istr, "<LOAD_FROM_STRING>", 1, 0);
}

template <typename file_handle_t, typename sample_name_t>
inline size_t sample_list<file_handle_t, sample_name_t>
::size() const {
  return m_sample_list.size();
}

template <typename file_handle_t, typename sample_name_t>
inline bool sample_list<file_handle_t, sample_name_t>
::empty() const {
  return m_sample_list.empty();
}

template <typename file_handle_t, typename sample_name_t>
inline std::string sample_list<file_handle_t, sample_name_t>
::read_header_line(std::istream& istrm,
                   const std::string& filename,
                   const std::string& info) const {
  if (!istrm.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: unable to read the header line of sample list " + filename + " for " + info);
  }

  std::string line;
  std::getline(istrm, line);

  if (line.empty()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: unable to read the header line of sample list " + filename + " for " + info
                          + " -- the line was empty");
  }
  return line;
}


template <typename file_handle_t, typename sample_name_t>
inline sample_list_header sample_list<file_handle_t, sample_name_t>
::read_header(std::istream& istrm,
              const std::string& filename) const {
  sample_list_header hdr;

  hdr.m_sample_list_filename = filename;

  std::string line1 = read_header_line(istrm, filename, "the exclusiveness");
  std::stringstream header1(line1);

  std::string line2 = read_header_line(istrm, filename, "the number of samples and the number of files");
  std::stringstream header2(line2);

  std::string line3 = read_header_line(istrm, filename, "the data file directory");
  std::stringstream header3(line3);

  std::string sample_list_type;
  header1 >> sample_list_type;
  std::for_each(sample_list_type.begin(), sample_list_type.end(), [](char& c){ c = std::toupper(c); });

  const std::string type_exclusive = sample_exclusion_list;
  size_t found = sample_list_type.find(type_exclusive);

  if (found != std::string::npos) {
    hdr.m_is_exclusive = true;
  } else {
    hdr.m_is_exclusive = false;
  }

  header2 >> hdr.m_included_sample_count;
  header2 >> hdr.m_excluded_sample_count;
  header2 >> hdr.m_num_files;

  header3 >> hdr.m_file_dir;

  if (hdr.get_file_dir().empty() || !check_if_dir_exists(hdr.get_file_dir())) {
    LBANN_ERROR(std::string{} + "file " + filename
                 + " :: data root directory '" + hdr.get_file_dir() + "' does not exist.");
  }

  return hdr;
}

template <typename file_handle_t, typename sample_name_t>
inline void sample_list<file_handle_t, sample_name_t>
::read_exclusive_list(std::istream& istrm,
                      size_t stride, size_t offset) {
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
    // Check to see if there is a strided load and skip the lines that are not for this rank
    if ((cnt_files-1)%stride != offset) {
      continue;
    }

    std::stringstream sstr(line.substr(0, end_of_str + 1)); // clear trailing spaces for accurate parsing
    std::string filename;
    size_t included_samples;
    size_t excluded_samples;
    std::unordered_set<std::string> excluded_sample_indices;

    sstr >> filename >> included_samples >> excluded_samples;

    const std::string file_path = add_delimiter(m_header.get_file_dir()) + filename;

    if (filename.empty() || !check_if_file_exists(file_path)) {
      LBANN_ERROR(std::string{} + " :: data file '" + file_path + "' does not exist.");
    }

    excluded_sample_indices.reserve(excluded_samples);

    while(!sstr.eof()) {
      std::string index;
      sstr >> index;
      excluded_sample_indices.insert(index);
    }

    if(excluded_sample_indices.size() != excluded_samples) {
      LBANN_ERROR(std::string("Index file does not contain the correct number of excluded samples: expected ")
                  + std::to_string(excluded_samples)
                  + std::string(" exclusions but found ")
                  + std::to_string(excluded_sample_indices.size()));
    }

    std::vector<std::string> sample_names;
    file_handle_t file_hnd = get_bundled_sample_names(file_path, sample_names, included_samples, excluded_samples);
    if (!is_file_handle_valid(file_hnd)) {
      continue; // skipping the file
    }

    if(m_file_map.count(filename) > 0) {
      if(sample_names.size() != m_file_map[filename]) {
        LBANN_ERROR(std::string("The same file ")
                    + filename
                    + " was opened multiple times and reported different sizes: "
                    + std::to_string(sample_names.size())
                    + " and "
                    + std::to_string(m_file_map[filename]));
      }
    }else {
      m_file_map[filename] = sample_names.size();
    }

    sample_file_id_t index = m_file_id_stats_map.size();
    m_file_id_stats_map.emplace_back(std::make_tuple(filename, 0, std::deque<std::pair<int,int>>{}));
    set_files_handle(filename, file_hnd);

    size_t valid_sample_count = 0u;
    for(auto s : sample_names) {
      std::unordered_set<std::string>::const_iterator found = excluded_sample_indices.find(s);
      if (found != excluded_sample_indices.cend()) {
        continue;
      }
      m_sample_list.emplace_back(index, to_sample_name_t<sample_name_t>(s));
      valid_sample_count++;
    }

    if(valid_sample_count != included_samples) {
      LBANN_ERROR(std::string("Bundle file does not contain the correct number of included samples: expected ")
                  + std::to_string(included_samples)
                  + std::string(" samples, but found ")
                  + std::to_string(valid_sample_count));
    }
  }

  if (m_header.get_num_files() != cnt_files) {
    LBANN_ERROR(std::string("Sample list ")
                + m_header.get_sample_list_filename()
                + std::string(": number of files requested ")
                + std::to_string(m_header.get_num_files())
                + std::string(" does not equal number of files loaded ")
                + std::to_string(cnt_files));
  }

  m_header.m_is_exclusive = false;
}


template <typename file_handle_t, typename sample_name_t>
inline void sample_list<file_handle_t, sample_name_t>
::read_inclusive_list(std::istream& istrm,
                      size_t stride, size_t offset) {
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
    // Check to see if there is a strided load and skip the lines that are not for this rank
    if ((cnt_files-1)%stride != offset) {
      continue;
    }

    std::stringstream sstr(line.substr(0, end_of_str + 1)); // clear trailing spaces for accurate parsing
    std::string filename;
    size_t included_samples;
    size_t excluded_samples;

    sstr >> filename >> included_samples >> excluded_samples;

    const std::string file_path = add_delimiter(m_header.get_file_dir()) + filename;

    if (filename.empty() || !check_if_file_exists(file_path)) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                            + " :: data file '" + filename + "' does not exist.");
    }

    std::vector<std::string> sample_names;
    file_handle_t file_hnd = get_bundled_sample_names(file_path, sample_names, included_samples, excluded_samples);
    if (!is_file_handle_valid(file_hnd)) {
      continue; // skipping the file
    }

    if(m_file_map.count(filename) > 0) {
      if(sample_names.size() != m_file_map[filename]) {
        LBANN_ERROR(std::string("The same file ")
                    + filename
                    + " was opened multiple times and reported different sizes: "
                    + std::to_string(sample_names.size())
                    + " and "
                    + std::to_string(m_file_map[filename]));
      }
    }else {
      m_file_map[filename] = sample_names.size();
    }

    std::unordered_set<std::string> set_of_samples(sample_names.begin(), sample_names.end());

    sample_file_id_t index = m_file_id_stats_map.size();
    m_file_id_stats_map.emplace_back(std::make_tuple(filename, 0, std::deque<std::pair<int,int>>{}));
    set_files_handle(filename, file_hnd);

    size_t valid_sample_count = 0u;
    while(!sstr.eof()) {
      std::string sample_name_str;
      sstr >> sample_name_str;
      std::unordered_set<std::string>::const_iterator found = set_of_samples.find(sample_name_str);
      if (found == set_of_samples.cend()) {
        LBANN_ERROR(std::string("Illegal request for a data ID that does not exist: ") + sample_name_str);
      }
      m_sample_list.emplace_back(index, to_sample_name_t<sample_name_t>(sample_name_str));
      valid_sample_count++;
    }
    if(valid_sample_count != included_samples) {
      LBANN_ERROR(std::string("Bundle file does not contain the correct number of included samples: expected ")
                  + std::to_string(included_samples)
                  + std::string(" samples, but found ")
                  + std::to_string(valid_sample_count));
    }
  }

  if (m_header.get_num_files() != cnt_files) {
    LBANN_ERROR(std::string("Sample list number of files requested ")
                + std::to_string(m_header.get_num_files())
                + std::string(" does not equal number of files loaded ")
                + std::to_string(cnt_files));
  }
}


template <typename file_handle_t, typename sample_name_t>
inline size_t sample_list<file_handle_t, sample_name_t>
::get_samples_per_file(std::istream& istrm,
                       const std::string& filename,
                       size_t stride, size_t offset) {
  m_header = read_header(istrm, filename);
  m_sample_list.reserve(m_header.get_sample_count());

  if (m_header.is_exclusive()) {
    read_exclusive_list(istrm, stride, offset);
  } else {
    read_inclusive_list(istrm, stride, offset);
  }

  if(stride == 1 && m_header.get_sample_count() != m_sample_list.size()) {
    LBANN_ERROR(std::string("Sample list count ")
                + std::to_string(m_header.get_sample_count())
                + std::string(" does not equal sample list size ")
                + std::to_string(m_sample_list.size()));
  }

  return m_sample_list.size();
}


template <typename file_handle_t, typename sample_name_t>
inline void sample_list<file_handle_t, sample_name_t>
::all_gather_archive(const std::string &archive,
                     std::vector<std::string>& gathered_archive,
                     lbann_comm& comm) {
  int size_of_list_archive = archive.size();
  std::vector<int> packed_sizes(comm.get_procs_per_trainer());

  comm.trainer_all_gather(size_of_list_archive, packed_sizes);

  int total_packed_size = 0;
  std::vector<int> displ;
  displ.assign(comm.get_procs_per_trainer()+1, 0);

  for (size_t i = 0u; i < packed_sizes.size(); ++i) {
    const auto sz = packed_sizes[i];
    displ[i+1] = displ[i] + sz;
  }
  total_packed_size = displ.back();

  if (total_packed_size <= 0) {
    return;
  }

  std::string all_samples;
  all_samples.resize(static_cast<size_t>(total_packed_size));

  std::vector<El::byte> local_data(archive.begin(), archive.end());
  std::vector<El::byte> packed_data(all_samples.begin(), all_samples.end());
  comm.trainer_all_gather(local_data,
                          packed_data,
                          packed_sizes,
                          displ);

  for (size_t i = 0u; i < packed_sizes.size(); ++i) {
    std::string& buf = gathered_archive[i];
    const auto sz = packed_sizes[i];
    displ[i+1] = displ[i] + sz;
    std::vector<El::byte>::const_iterator first = packed_data.begin() + displ[i];
    std::vector<El::byte>::const_iterator last = packed_data.begin() + displ[i] + sz;
    buf.resize(sz);
    buf.assign(first, last);
  }
  return;
}

template <typename file_handle_t, typename sample_name_t>
template <typename T>
inline size_t sample_list<file_handle_t, sample_name_t>
::all_gather_field(T data,
                   std::vector<T>& gathered_data,
                   lbann_comm& comm) {
  std::string archive;
  std::stringstream ss;
  cereal::BinaryOutputArchive oarchive(ss);
  oarchive(data);
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

template <typename file_handle_t, typename sample_name_t>
inline void sample_list<file_handle_t, sample_name_t>
::clear() {
  m_sample_list.clear();
}

template <typename file_handle_t, typename sample_name_t>
template <class Archive>
void sample_list<file_handle_t, sample_name_t>
::serialize( Archive & ar ) {
  ar(m_header, m_sample_list, m_file_id_stats_map);
}

template <typename file_handle_t, typename sample_name_t>
inline void sample_list<file_handle_t, sample_name_t>
::write_header(std::string& sstr, size_t num_files) const {
  // The first line indicate if the list is exclusive or inclusive
  // The next line contains the number of samples and the number of files, which are the same in this caes
  // The next line contains the root data file directory

  sstr += (m_header.is_exclusive()? sample_exclusion_list + "\n" : sample_inclusion_list + "\n");
  /// Include the number of invalid samples, which for an inclusive index list is always 0
  sstr += std::to_string(m_sample_list.size()) + " 0 " + std::to_string(num_files) + '\n';
  sstr += m_header.get_file_dir() + '\n';
}


template <typename file_handle_t, typename sample_name_t>
inline bool sample_list<file_handle_t, sample_name_t>
::to_string(std::string& sstr) const {
  std::map<std::string, std::template vector<sample_name_t>> tmp_file_map;
  for (const auto& s : m_sample_list) {
    std::string filename = std::get<0>(m_file_id_stats_map[s.first]);
    tmp_file_map[filename].emplace_back(s.second);
  }

  typename samples_t::const_iterator it_begin = m_sample_list.cbegin();
  typename samples_t::const_iterator it_end = m_sample_list.cbegin();

  sstr.clear();

  // reserve the string to hold the entire sample lit
  size_t estimated_len = 30 + 42 + m_header.get_file_dir().size() + 1;
  if (it_begin < it_end) {
    estimated_len += tmp_file_map.size();
    sstr.reserve(estimated_len);
  }

  // write the list header
  write_header(sstr, tmp_file_map.size());

  // write the list body
  for (const auto& f : tmp_file_map) {
    // File name
    sstr += f.first;
    // Number of included samples
    sstr += std::string(" ") + std::to_string(f.second.size());
    // Number of excluded samples
    sstr += std::string(" ") + std::to_string(m_file_map.at(f.first) - f.second.size());
    // Inclusion sample list
    for (const auto& s : f.second) {
      sstr += ' ' + lbann::to_string(s);
    }
    sstr += '\n';
  }

  return true;
}

template <typename file_handle_t, typename sample_name_t>
inline void sample_list<file_handle_t, sample_name_t>
::write(const std::string filename) const {
  std::string dir, basename;
  parse_path(filename, dir, basename);
  if (!dir.empty() && !check_if_dir_exists(dir)) {
    // The creation of a shared directory must be done once in a coordinated fashion
    // among the entities that have access to it. Thus, it must be done in advance
    std::cerr << "The sample list output directory (" + dir + ") does not exist" << std::endl;
    return;
  }

  std::fstream ofs(filename, std::fstream::out | std::fstream::binary);

  if (!ofs.good()) {
    return;
  }

  std::string buf;
  to_string(buf);

  ofs.write(buf.data(), buf.size()*sizeof(std::string::value_type));
  ofs.close();
}

template <typename file_handle_t, typename sample_name_t>
inline const typename sample_list<file_handle_t, sample_name_t>::samples_t&
sample_list<file_handle_t, sample_name_t>::get_list() const {
  return m_sample_list;
}

template <typename file_handle_t, typename sample_name_t>
inline const sample_list_header&
sample_list<file_handle_t, sample_name_t>::get_header() const {
  return m_header;
}

template <typename file_handle_t, typename sample_name_t>
inline const typename sample_list<file_handle_t, sample_name_t>::sample_t&
sample_list<file_handle_t, sample_name_t>::operator[](size_t idx) const {
  return m_sample_list[idx];
}

template <typename file_handle_t, typename sample_name_t>
inline const std::string& sample_list<file_handle_t, sample_name_t>
::get_samples_filename(sample_file_id_t id) const {
  return std::get<0>(m_file_id_stats_map[id]);
}

template <typename file_handle_t, typename sample_name_t>
inline   const std::string& sample_list<file_handle_t, sample_name_t>
::get_samples_dirname() const {
  return m_header.get_file_dir();
}

template <typename file_handle_t, typename sample_name_t>
inline file_handle_t sample_list<file_handle_t, sample_name_t>
::get_samples_file_handle(sample_file_id_t id) const {
  file_handle_t h = std::get<1>(m_file_id_stats_map[id]);
  return h;
}

template <typename file_handle_t, typename sample_name_t>
inline void sample_list<file_handle_t, sample_name_t>
::set_samples_filename(sample_file_id_t id, const std::string& filename) {
  std::get<0>(m_file_id_stats_map[id]) = filename;
}

template <typename file_handle_t, typename sample_name_t>
inline void sample_list<file_handle_t, sample_name_t>
::set_files_handle(const std::string& filename, file_handle_t h) {
  sample_file_id_t id = sample_file_id_t(0);
  for (auto&& e : m_file_id_stats_map) {
    if(std::get<0>(e) == filename) {
      std::get<1>(e) = h;
      break;
    }
    id++;
  }
  manage_open_file_handles(id, true);
}

} // end of namespace lbann
