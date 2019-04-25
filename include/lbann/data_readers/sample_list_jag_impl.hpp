#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <locale>
#include "sample_list_jag.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/file_utils.hpp"
#include <deque>
#include "hdf5.h"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include <unordered_set>
#include <memory>

#include <cereal/archives/binary.hpp>
#include <sstream>
#include <unistd.h>

namespace lbann {

inline sample_list_header::sample_list_header()
  : m_is_exclusive(false), m_included_sample_count(0u), m_excluded_sample_count(0u), m_num_files(0u), m_file_dir("") {
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

inline sample_list_jag::sample_list_jag() {
  m_max_open_files = getdtablesize() - LBANN_MAX_OPEN_FILE_MARGIN;
}

inline sample_list_jag::~sample_list_jag() {
  // Close the existing open files
  for(auto f : m_file_id_stats_map) {
    if(std::get<1>(f) > 0) {
      conduit::relay::io::hdf5_close_file(std::get<1>(f));
    }
    std::get<1>(f) = 0;
    std::get<2>(f).clear();
  }
  m_file_id_stats_map.clear();
  m_open_fd_pq.clear();
}

inline sample_list_jag::sample_list_jag(const sample_list_jag& rhs) {
  copy_members(rhs);
}

inline sample_list_jag& sample_list_jag::operator=(const sample_list_jag& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  copy_members(rhs);

  return (*this);
}

inline sample_list_jag& sample_list_jag::copy(const sample_list_jag& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  copy_members(rhs);

  return (*this);
}

inline void sample_list_jag::copy_members(const sample_list_jag& rhs) {
  m_header = rhs.m_header;
  m_sample_list = rhs.m_sample_list;
  m_file_id_stats_map = rhs.m_file_id_stats_map;
  m_file_map = rhs.m_file_map;
  m_max_open_files = rhs.m_max_open_files;

  /// Keep track of existing filenames but do not copy any file
  /// descriptor information
  for(auto&& e : m_file_id_stats_map) {
    if(std::get<1>(e) > 0) {
      std::get<1>(e) = 0;
    }
    std::get<2>(e).clear();
  }

  /// Do not copy the open file descriptor priority queue
  /// File handle ownership is not transfered in the copy
  m_open_fd_pq.clear();
}

inline void sample_list_jag::load(const std::string& samplelist_file, size_t stride, size_t offset) {
  std::ifstream istr(samplelist_file);
  get_samples_per_file(istr, samplelist_file, stride, offset);
  istr.close();
}

inline sample_list_header sample_list_jag::load_header(const std::string& samplelist_file) const {
  std::ifstream istr(samplelist_file);
  return read_header(istr, samplelist_file);
}

inline void sample_list_jag::load_from_string(const std::string& samplelist) {
  std::istringstream istr(samplelist);
  get_samples_per_file(istr, "<LOAD_FROM_STRING>", 1, 0);
}

inline size_t sample_list_jag::size() const {
  return m_sample_list.size();
}

inline bool sample_list_jag::empty() const {
  return m_sample_list.empty();
}

inline std::string sample_list_jag::read_header_line(std::istream& istrm, const std::string& filename, const std::string& info) const {
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


inline sample_list_header sample_list_jag::read_header(std::istream& istrm, const std::string& filename) const {
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

  const std::string type_exclusive = conduit_hdf5_exclusion_list;
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

inline hid_t sample_list_jag::get_conduit_bundle_samples(std::string conduit_file_path, std::vector<std::string>& sample_names, size_t included_samples, size_t excluded_samples) {
  hid_t hdf5_file_hnd = 0;
  bool retry = false;
  int retry_cnt = 0;
  do {
    try {
      hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( conduit_file_path );
    }catch (conduit::Error const& e) {
      LBANN_WARNING(" :: trying to open the file " + conduit_file_path + " and got " + e.what());
      retry = true;
      retry_cnt++;
    }
  }while(retry && retry_cnt < LBANN_MAX_OPEN_FILE_RETRY);

  if (hdf5_file_hnd <= static_cast<hid_t>(0)) {
    std::cout << "Opening the file didn't work" << std::endl;
    return hdf5_file_hnd;
  }

  conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", sample_names);

  if(sample_names.size() != (included_samples + excluded_samples)) {
    LBANN_ERROR(std::string("File does not contain the correct number of samples: found ")
                + std::to_string(sample_names.size())
                + std::string(" -- this does not equal the expected number of samples that are marked for inclusion: ")
                + std::to_string(included_samples)
                + std::string(" and exclusion: ")
                + std::to_string(excluded_samples));
  }

  return hdf5_file_hnd;
}

inline void sample_list_jag::read_exclusive_list(std::istream& istrm, size_t stride, size_t offset) {
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

    const std::string conduit_file_path = add_delimiter(m_header.get_file_dir()) + filename;

    if (filename.empty() || !check_if_file_exists(conduit_file_path)) {
      LBANN_ERROR(std::string{} + " :: data file '" + conduit_file_path + "' does not exist.");
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
    hid_t hdf5_file_hnd = get_conduit_bundle_samples(conduit_file_path, sample_names, included_samples, excluded_samples);
    if(hdf5_file_hnd <= static_cast<hid_t>(0)) {
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
    set_files_hdf5_handle(filename, hdf5_file_hnd);

    size_t valid_sample_count = 0u;
    for(auto s : sample_names) {
      std::unordered_set<std::string>::const_iterator found = excluded_sample_indices.find(s);
      if (found != excluded_sample_indices.cend()) {
        continue;
      }
      m_sample_list.emplace_back(index, s);
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


inline void sample_list_jag::read_inclusive_list(std::istream& istrm, size_t stride, size_t offset) {
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

    const std::string conduit_file_path = add_delimiter(m_header.get_file_dir()) + filename;

    if (filename.empty() || !check_if_file_exists(conduit_file_path)) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                            + " :: data file '" + filename + "' does not exist.");
    }

    std::vector<std::string> sample_names;
    hid_t hdf5_file_hnd = get_conduit_bundle_samples(conduit_file_path, sample_names, included_samples, excluded_samples);
    if(hdf5_file_hnd <= static_cast<hid_t>(0)) {
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
    set_files_hdf5_handle(filename, hdf5_file_hnd);

    size_t valid_sample_count = 0u;
    while(!sstr.eof()) {
      std::string sample_name;;
      sstr >> sample_name;
      std::unordered_set<std::string>::const_iterator found = set_of_samples.find(sample_name);
      if (found == set_of_samples.cend()) {
        LBANN_ERROR(std::string("Illegal request for a data ID that does not exist: ") + sample_name);
      }
      m_sample_list.emplace_back(index, sample_name);
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


inline size_t sample_list_jag::get_samples_per_file(std::istream& istrm, const std::string& filename, size_t stride, size_t offset) {
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


inline void sample_list_jag::all_gather_archive(const std::string &archive, std::vector<std::string>& gathered_archive, lbann_comm& comm) {
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

template<typename T>
inline size_t sample_list_jag::all_gather_field(T data, std::vector<T>& gathered_data, lbann_comm& comm) {
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

inline void sample_list_jag::all_gather_packed_lists(lbann_comm& comm) {
  int num_ranks = comm.get_procs_per_trainer();
  std::vector<samples_t> per_rank_samples(num_ranks);
  std::vector<file_id_stats_v_t> per_rank_file_id_stats_map(num_ranks);
  std::vector<std::unordered_map<std::string, size_t>> per_rank_file_map(num_ranks);

  // Close the existing open files
  for(auto&& e : m_file_id_stats_map) {
    if(std::get<1>(e) > 0) {
      conduit::relay::io::hdf5_close_file(std::get<1>(e));
      std::get<1>(e) = 0;
    }
    std::get<2>(e).clear();
  }
  m_open_fd_pq.clear();

  size_t num_samples = all_gather_field(m_sample_list, per_rank_samples, comm);
  size_t num_ids = all_gather_field(m_file_id_stats_map, per_rank_file_id_stats_map, comm);
  size_t num_files = all_gather_field(m_file_map, per_rank_file_map, comm);

  m_sample_list.clear();
  m_file_id_stats_map.clear();

  m_sample_list.reserve(num_samples);
  m_file_id_stats_map.reserve(num_ids);
  m_file_map.reserve(num_files);

  for(int r = 0; r < num_ranks; r++) {
    const samples_t& sample_list = per_rank_samples[r];
    const file_id_stats_v_t& file_id_stats_map = per_rank_file_id_stats_map[r];
    const std::unordered_map<std::string, size_t>& file_map = per_rank_file_map[r];
    for (const auto& s : sample_list) {
      sample_file_id_t index = s.first;
      const std::string& filename = std::get<0>(file_id_stats_map[index]);
      if(index >= m_file_id_stats_map.size()
         || (std::get<0>(m_file_id_stats_map.back()) != filename)) {
        index = m_file_id_stats_map.size();
        m_file_id_stats_map.emplace_back(std::make_tuple(filename, 0, std::deque<std::pair<int,int>>{}));
        // Update the file map structure
        if(m_file_map.count(filename) == 0) {
          m_file_map[filename] = file_map.at(filename);
        }
      }else {
        for(size_t i = 0; i < m_file_id_stats_map.size(); i++) {
          if(filename == std::get<0>(m_file_id_stats_map[i])) {
            index = i;
            break;
          }
        }
      }
      m_sample_list.emplace_back(std::make_pair(index, s.second));
    }
  }

  return;
}

inline void sample_list_jag::compute_epochs_file_usage(const std::vector<int>& shuffled_indices, int mini_batch_size, const lbann_comm& comm) {
  for (auto&& e : m_file_id_stats_map) {
    if(std::get<1>(e) > 0) {
      conduit::relay::io::hdf5_close_file(std::get<1>(e));
    }
    std::get<1>(e) = 0;
    std::get<2>(e).clear();
  }
  // Once all of the file handles are closed, clear the priority queue
  m_open_fd_pq.clear();

  for (size_t i = 0; i < shuffled_indices.size(); i++) {
    int idx = shuffled_indices[i];
    const auto& s = m_sample_list[idx];
    sample_file_id_t index = s.first;

    if((i % mini_batch_size) % comm.get_procs_per_trainer() == static_cast<size_t>(comm.get_rank_in_trainer())) {
      /// Enqueue the iteration step when the sample will get used
      int step = i / mini_batch_size;
      int substep = (i % mini_batch_size) / comm.get_procs_per_trainer();
      std::get<2>(m_file_id_stats_map[index]).emplace_back(std::make_pair(step, substep));
    }
  }
}

inline void sample_list_jag::clear() {
  m_sample_list.clear();
}

template <class Archive> void sample_list_jag::serialize( Archive & ar ) {
  ar(m_header, m_sample_list, m_file_id_stats_map);
}

inline void sample_list_jag::write_header(std::string& sstr, size_t num_files) const {
  // The first line indicate if the list is exclusive or inclusive
  // The next line contains the number of samples and the number of files, which are the same in this caes
  // The next line contains the root data file directory

  sstr += (m_header.is_exclusive()? conduit_hdf5_exclusion_list + "\n" : conduit_hdf5_inclusion_list + "\n");
  /// Include the number of invalid samples, which for an inclusive index list is always 0
  sstr += std::to_string(m_sample_list.size()) + " 0 " + std::to_string(num_files) + '\n';
  sstr += m_header.get_file_dir() + '\n';
}


inline bool sample_list_jag::to_string(std::string& sstr) const {
  std::map<std::string, std::vector<sample_name_t>> tmp_file_map;
  for (const auto& s : m_sample_list) {
    std::string filename = std::get<0>(m_file_id_stats_map[s.first]);
    tmp_file_map[filename].emplace_back(s.second);
  }

  samples_t::const_iterator it_begin = m_sample_list.cbegin();
  samples_t::const_iterator it_end = m_sample_list.cbegin();

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
      sstr += ' ' + s;
    }
    sstr += '\n';
  }

  return true;
}

inline void sample_list_jag::write(const std::string filename) const {
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

inline const sample_list_jag::samples_t& sample_list_jag::get_list() const {
  return m_sample_list;
}

inline const sample_list_header& sample_list_jag::get_header() const {
  return m_header;
}

inline const sample_list_jag::sample_t& sample_list_jag::operator[](size_t idx) const {
  return m_sample_list[idx];
}

} // end of namespace lbann
