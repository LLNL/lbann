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
  LBANN_ERROR(std::string{} + " :: string conversion is not implement for the sample_name_t");
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

//------------------------
//   sample_list_header
//------------------------

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

//------------------
//   sample_list
//------------------

template <typename sample_name_t>
inline sample_list<sample_name_t>::sample_list() {
}

template <typename sample_name_t>
inline sample_list<sample_name_t>::~sample_list() {
}

template <typename sample_name_t>
inline sample_list<sample_name_t>
::sample_list(const sample_list& rhs) {
  copy_members(rhs);
}

template <typename sample_name_t>
inline sample_list<sample_name_t>& sample_list<sample_name_t>
::operator=(const sample_list& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  copy_members(rhs);

  return (*this);
}

template <typename sample_name_t>
inline sample_list<sample_name_t>& sample_list<sample_name_t>
::copy(const sample_list& rhs) {
  // check for self-assignment
  if (this == &rhs) {
    return (*this);
  }

  copy_members(rhs);

  return (*this);
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>
::copy_members(const sample_list& rhs) {
  m_header = rhs.m_header;
  m_sample_list = rhs.m_sample_list;

  /// Keep track of existing filenames
  m_file_id_stats_map = rhs.m_file_id_stats_map;
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>
::load(const std::string& samplelist_file,
       size_t stride, size_t offset) {
  std::ifstream istr(samplelist_file);
  get_samples_per_file(istr, samplelist_file, stride, offset);
  istr.close();
}

template <typename sample_name_t>
inline sample_list_header sample_list<sample_name_t>
::load_header(const std::string& samplelist_file) const {
  std::ifstream istr(samplelist_file);
  return read_header(istr, samplelist_file);
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>
::load_from_string(const std::string& samplelist) {
  std::istringstream istr(samplelist);
  get_samples_per_file(istr, "<LOAD_FROM_STRING>", 1, 0);
}

template <typename sample_name_t>
inline size_t sample_list<sample_name_t>
::size() const {
  return m_sample_list.size();
}

template <typename sample_name_t>
inline size_t sample_list<sample_name_t>
::get_num_files() const {
  return m_file_id_stats_map.size();
}

template <typename sample_name_t>
inline bool sample_list<sample_name_t>
::empty() const {
  return (size() == 0ul);
}

template <typename sample_name_t>
inline std::string sample_list<sample_name_t>
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


template <typename sample_name_t>
inline sample_list_header sample_list<sample_name_t>
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


template <typename sample_name_t>
inline void sample_list<sample_name_t>
::read_sample_list(std::istream& istrm,
                      size_t stride, size_t offset) {
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
    // Check to see if there is a strided load and skip the lines that are not for this rank
    if ((cnt_files-1)%stride != offset) {
      continue;
    }

    std::stringstream sstr(line.substr(0, end_of_str + 1)); // clear trailing spaces for accurate parsing
    std::string filename;

    sstr >> filename;

    const std::string file_path = add_delimiter(m_header.get_file_dir()) + filename;

    if (filename.empty() || !check_if_file_exists(file_path)) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                            + " :: data file '" + filename + "' does not exist.");
    }

    const sample_file_id_t index = m_file_id_stats_map.size();
    static const auto sn0 = uninitialized_sample_name<sample_name_t>();
    m_sample_list.emplace_back(std::make_pair(index, sn0));
    m_file_id_stats_map.emplace_back(filename);
  }

  if (m_header.get_num_files() != cnt_files) {
    LBANN_ERROR(std::string("Sample list number of files requested ")
                + std::to_string(m_header.get_num_files())
                + std::string(" does not equal number of files loaded ")
                + std::to_string(cnt_files));
  }

  if(stride == 1 && m_header.get_sample_count() != m_sample_list.size()) {
    LBANN_ERROR(std::string("Sample list count ")
                + std::to_string(m_header.get_sample_count())
                + std::string(" does not equal sample list size ")
                + std::to_string(m_sample_list.size()));
  }
}


template <typename sample_name_t>
inline size_t sample_list<sample_name_t>
::get_samples_per_file(std::istream& istrm,
                       const std::string& filename,
                       size_t stride, size_t offset) {
  m_header = read_header(istrm, filename);

  read_sample_list(istrm, stride, offset);

  return size();
}


template <typename sample_name_t>
inline void sample_list<sample_name_t>
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
  std::vector<El::byte> packed_data(all_samples.size() * sizeof(decltype(all_samples)::value_type));
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

template <typename sample_name_t>
template <typename T>
inline size_t sample_list<sample_name_t>
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

template <typename sample_name_t>
template <class Archive>
void sample_list<sample_name_t>
::serialize( Archive & ar ) {
  ar(m_header, m_sample_list, m_file_id_stats_map);
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>
::write_header(std::string& sstr, size_t num_files) const {
  // The first line indicate if the list is exclusive or inclusive
  // The next line contains the number of samples (included and excluded),
  // as well as the number of files, which are the same in this caes
  // The next line contains the root data file directory

  sstr += (m_header.is_exclusive()? sample_exclusion_list + "\n" : sample_inclusion_list + "\n");
  size_t total, included, excluded;
  get_num_samples(total, included, excluded);
  /// TODO: clarify the comment below
  /// Include the number of invalid samples, which for an inclusive index list is always 0
  sstr += std::to_string(included) + ' '  + std::to_string(excluded) + ' '  + std::to_string(num_files) + '\n';
  sstr += m_header.get_file_dir() + '\n';
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>
::get_num_samples(size_t& total, size_t& included, size_t& excluded) const {
  total = size();
  included = size();
  excluded = 0ul;
}

template <typename sample_name_t>
inline bool sample_list<sample_name_t>
::to_string(std::string& sstr) const {
  size_t total_len = 0ul;
  for (const auto& s : m_sample_list) {
    const std::string& filename = m_file_id_stats_map[s.first];
    total_len += filename.size() + 1u;
  }

  sstr.clear();

  // reserve the string to hold the entire sample lit
  size_t estimated_len = 30 + 42 + m_header.get_file_dir().size() + 1 + total_len + 1000;
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
inline void sample_list<sample_name_t>
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

template <typename sample_name_t>
inline const typename sample_list<sample_name_t>::samples_t&
sample_list<sample_name_t>::get_list() const {
  return m_sample_list;
}

template <typename sample_name_t>
inline const sample_list_header&
sample_list<sample_name_t>::get_header() const {
  return m_header;
}

template <typename sample_name_t>
inline const typename sample_list<sample_name_t>::sample_t&
sample_list<sample_name_t>::operator[](size_t idx) const {
  return m_sample_list[idx];
}

template <typename sample_name_t>
inline const std::string& sample_list<sample_name_t>
::get_samples_filename(sample_file_id_t id) const {
  return m_file_id_stats_map[id];
}

template <typename sample_name_t>
inline   const std::string& sample_list<sample_name_t>
::get_samples_dirname() const {
  return m_header.get_file_dir();
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>
::set_samples_filename(sample_file_id_t id, const std::string& filename) {
  m_file_id_stats_map[id] = filename;
}

#if defined(__cpp_if_constexpr) // c++17
template <typename sample_name_t>
inline void sample_list<sample_name_t>
::assign_samples_name() {
  if constexpr (std::is_integral<sample_name_t>::value
            && !std::is_same<sample_name_t, bool>::value) {
    sample_name_t i = 0;
    for (auto& s: m_sample_list) {
      s.second = i++;
    }
  } else if constexpr (std::is_same<std::string, sample_name_t>::value) {
    for (auto& s: m_sample_list) {
      s.second = s.first;
    }
  } else {
    LBANN_ERROR(std::string{} + " :: base class does not implement this method"
                              + " for the current sample name type");
  }
}

template <typename sample_name_t>
inline sample_name_t uninitialized_sample_name() {
  if constexpr (std::is_integral<sample_name_t>::value) {
    return static_cast<sample_name_t>(0);
  } else if constexpr (std::is_same<std::string, sample_name_t>::value) {
    return "";
  } else if constexpr (std::is_floating_point<sample_name_t>::value) {
    return 0.0;
  } else if constexpr (std::is_default_constructible<sample_name_t>::value
                      && std::is_copy_constructible<sample_name_t>::value) {
    sample_name_t ret{};
    return ret;
  } else {
    LBANN_ERROR(std::string{} + " :: base class does not implement this method"
                              + " for the current sample name type");
  }
}
#else
template<> inline void sample_list<size_t>
::assign_samples_name() {
  size_t i = 0ul;
  for (auto& s: m_sample_list) {
    s.second = i++;
  }
}

template<> inline void sample_list<std::string>
::assign_samples_name() {
  for (auto& s: m_sample_list) {
    s.second = s.first;
  }
}

template <typename sample_name_t>
inline void sample_list<sample_name_t>
::assign_samples_name() {
  LBANN_ERROR(std::string{} + " :: base class does not implement this method"
                            + " for the current sample name type");
}

template<> inline size_t uninitialized_sample_name<size_t>() {
  return 0ul;
}

template<> inline std::string uninitialized_sample_name<std::string>() {
  return "";
}

template <typename sample_name_t>
inline sample_name_t uninitialized_sample_name() {
  sample_name_t ret{};
  return ret;
}
#endif // defined(__cpp_if_constexpr)

template <typename sample_name_t>
inline void sample_list<sample_name_t>
::all_gather_packed_lists(lbann_comm& comm) {
  int num_ranks = comm.get_procs_per_trainer();
  typename std::vector<samples_t> per_rank_samples(num_ranks);
  typename std::vector<std::vector<std::string>> per_rank_files(num_ranks);

  size_t num_samples = all_gather_field(m_sample_list, per_rank_samples, comm);
  size_t num_ids = all_gather_field(m_file_id_stats_map, per_rank_files, comm);

  m_sample_list.clear();
  m_file_id_stats_map.clear();

  m_sample_list.reserve(num_samples);
  m_file_id_stats_map.reserve(num_ids);

  for(int r = 0; r < num_ranks; r++) {
    const samples_t& s_list = per_rank_samples[r];
    const auto& files = per_rank_files[r];
    for (const auto& s : s_list) {
      sample_file_id_t index = s.first;
      const std::string& filename = files[index];
      if(index >= m_file_id_stats_map.size()
         || (m_file_id_stats_map.back() != filename)) {
        index = m_file_id_stats_map.size();
        m_file_id_stats_map.emplace_back(filename);
      }else {
        for(size_t i = 0; i < m_file_id_stats_map.size(); i++) {
          if(filename == m_file_id_stats_map[i]) {
            index = i;
            break;
          }
        }
      }
      static const auto sn0 = uninitialized_sample_name<sample_name_t>();
      m_sample_list.emplace_back(std::make_pair(index, sn0));
    }
  }

  assign_samples_name();

  return;
}

} // end of namespace lbann
