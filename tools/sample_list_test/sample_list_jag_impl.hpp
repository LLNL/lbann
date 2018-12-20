#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include "sample_list_jag.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/file_utils.hpp"

namespace lbann {

inline void sample_list_jag::set_num_partitions(size_t n) {
  if (n == 0) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: number of partitions must be a positive number ("
                          + std::to_string(n) + ")");
  }
  clear();
  m_num_partitions = n;
}


inline void sample_list_jag::load(const std::string& samplelist_file) {
  std::ifstream istr(samplelist_file);
  get_samples_per_file(istr);
  istr.close();
  get_sample_range_per_part();

  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_sample_list.size());
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
}


inline void sample_list_jag::load_from_string(const std::string& samplelist) {
  std::istringstream istr(samplelist);
  get_samples_per_file(istr);
  get_sample_range_per_part();
}


inline size_t sample_list_jag::get_samples_per_file(std::istream& ifstr)
{
  if (!ifstr.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: unable to read from the input stream of sample list");
  }

  std::string line;
  std::getline(ifstr, line);

  if(!ifstr.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + "Unable to read HDF5 sample index format");
  }

  // Read the first header line and allocate the list space
  std::stringstream header1(line);
  size_t sample_count, num_files;
  header1 >> sample_count;
  header1 >> num_files;

  m_sample_list.reserve(sample_count);

  if(!ifstr.good() && (num_files > 0u)) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + "Unable to read HDF5 sample index format");
  }

  std::getline(ifstr, line);
  std::stringstream header2(line);
  m_file_dir.clear();
  header2 >> m_file_dir;

  if (m_file_dir.empty() || !check_if_dir_exists(m_file_dir)) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: data root directory '" + m_file_dir + "' does not exist.");
  }

  const std::string whitespaces(" \t\f\v\n\r");

  size_t cnt_files = 0u;

  while (std::getline(ifstr, line)) {
    const size_t end_of_str = line.find_last_not_of(whitespaces);
    if (end_of_str == std::string::npos) { // empty line
      continue;
    }
    std::stringstream sstr(line.substr(0, end_of_str + 1)); // clear trailing spaces for accurate parsing
    std::string filename;
    size_t valid_samples;
    size_t invalid_samples;
    std::unordered_set<size_t> invalid_sample_indices;

    sstr >> filename >> valid_samples >> invalid_samples;

    const std::string conduit_file_path = add_delimiter(m_file_dir) + filename;

    if (filename.empty() || !check_if_file_exists(conduit_file_path)) {
      throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                            + " :: data file '" + filename + "' does not exist.");
    }

    if (++cnt_files >= num_files) {
      break;
    }

    invalid_sample_indices.reserve(valid_samples + invalid_samples);

    while(!sstr.eof()) {
      size_t index;
      sstr >> index;
      invalid_sample_indices.insert(index);
    }

    //std::cout << "I am going to load the file " << filename << " which has " << valid_samples << " valid samples and " << invalid_samples << std::endl;

    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( conduit_file_path );

    if (hdf5_file_hnd <= static_cast<hid_t>(0)) {
      std::cout << "Opening the file didn't work" << std::endl;
      continue; // skipping the file
    }

    std::vector<std::string> sample_names;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", sample_names);

    size_t i = 0u;
    for(auto s : sample_names) {
      std::unordered_set<size_t>::const_iterator found = invalid_sample_indices.find(i++);
      if (found != invalid_sample_indices.cend()) {
        continue;
      }

      m_sample_list.emplace_back(conduit_file_path, s);
    }
  }

  return m_sample_list.size();
}


inline void sample_list_jag::get_sample_range_per_part() {
  // Populates m_sample_range_per_part, requires the total number of samples
  // and number of partitions are known.
  const size_t total = static_cast<size_t>(m_sample_list.size());
  const size_t one_more = total % m_num_partitions;
  const size_t min_per_partition = total/m_num_partitions;

  if (min_per_partition == 0u) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
          + " :: insufficient number of samples for each partition to have at least one.");
  }

  m_sample_range_per_part.clear();
  m_sample_range_per_part.resize(m_num_partitions+1u);

  #pragma omp parallel for
  for (size_t p = 0u; p < m_num_partitions; ++p) {
    const size_t r_start = min_per_partition * p + ((p >= one_more)? one_more : p);
    const size_t r_end = r_start + min_per_partition + ((p < one_more)? 1u : 0u);
    m_sample_range_per_part[p+1] = r_end;
  }
}


inline void sample_list_jag::clear() {
  m_num_partitions = 1u;
  m_sample_list.clear();
  m_sample_range_per_part.clear();
}


inline bool sample_list_jag::to_string(size_t p, std::string& sstr) const {
  if (p >= m_num_partitions) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
          + " :: partition id is out of range.");
  }

  const size_t i_begin = m_sample_range_per_part[p];
  const size_t i_end = m_sample_range_per_part[p+1];

  if (i_begin > i_end) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
          + " :: incorrect partition range.");
  }

  samples_t::const_iterator it_begin = m_sample_list.cbegin();
  samples_t::const_iterator it_end = m_sample_list.cbegin();
  std::advance(it_begin, i_begin);
  std::advance(it_end, i_end);

  sstr.clear();

  size_t estimated_len = 42 + m_file_dir.size() + 1;
  if (i_begin < i_end) {
    estimated_len += static_cast<size_t>(1.5 * (i_end - i_begin) * (it_begin->first.size() + it_begin->second.size() + 6));
    sstr.reserve(estimated_len);
  }

  // write header
  // The first line contains the number of samples and the number of files, which are the same in this caes
  // The second line contains the root data file directory

  sstr += std::to_string(m_sample_list.size()) + ' ' + std::to_string(m_sample_list.size()) + '\n';
  sstr += m_file_dir + '\n';

  // write list
  for (samples_t::const_iterator it = it_begin; it != it_end; ++it) {
    sstr += it->first + " 1 0 " + it->second + '\n';
  }

  std::cerr << "estimated size vs actual size: " << estimated_len << ' ' << sstr.size() << std::endl;
  std::cerr << "num samples: " << i_end - i_begin << " samples of rank " << p << std::endl;
  return true;
}


} // end of namespace lbann
