#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <list>
#include <algorithm>
#include "sample_list.hpp"

namespace lbann {

template <typename SN>
std::string sample_list<SN>::to_string(const std::string& s) {
  return s;
}

template <typename SN>
template <typename T>
std::string sample_list<SN>::to_string(const T v) {
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
  ok = get_samples_per_file(samplelist_file);
  ok = ok && get_sample_range_per_file();
  ok = ok && get_sample_range_per_part();
  return ok;
}

template <typename SN>
inline size_t sample_list<SN>::get_samples_per_file(const std::string& samplelist_file)
{
  std::ifstream ifstr(samplelist_file);
  if (!ifstr.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: failed to open " + samplelist_file + " for reading");
  }

  std::string line;

  size_t total_num_samples = 0u;
  m_filenames.clear();

  while (getline(ifstr, line)) {
    std::stringstream sstr(line);
    std::string filename;

    sstr >> filename;
    m_filenames.emplace_back(filename);

    m_samples_per_file.emplace_back();
    auto& samples_of_current_file = m_samples_per_file.back();

    sample_name_t sample_name;

    while (sstr >> sample_name) {
      samples_of_current_file.emplace_back(sample_name);
    }

    const size_t num_samples_of_current_file = samples_of_current_file.size();
    total_num_samples += num_samples_of_current_file;
  }

  ifstr.close();

  return total_num_samples;
}


/**
 * Reads through m_samples_per_file, and populate m_sample_range_per_file
 * by the sequential id of the first sample in each sample file.
 * The last element of m_sample_range_per_file is the total number of samples.
 */
template <typename SN>
inline bool sample_list<SN>::get_sample_range_per_file()
{
  if (m_samples_per_file.empty()) {
    return false;
  }

  m_sample_range_per_file.clear();
  m_sample_range_per_file.reserve(m_samples_per_file.size()+1u);
  m_sample_range_per_file.push_back(0u);

  size_t total_so_far = 0u;

  for (const auto slist: m_samples_per_file) {
    total_so_far += slist.size();
    m_sample_range_per_file.push_back(total_so_far);
  }
  return true;
}


template <typename SN>
inline bool sample_list<SN>::get_sample_range_per_part()
{
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

  #pragma omp parallel for
  for (size_t p = 0u; p < m_num_partitions; ++p) {
    const size_t r_start = min_per_partition * p + ((p >= one_more)? one_more : p);
    const size_t r_end = r_start + min_per_partition + ((p < one_more)? 1u : 0u);
    m_sample_range_per_part[p+1] = r_end;
  }

  return true;
}


template <typename SN>
inline bool sample_list<SN>::find_sample_files_of_part(size_t p, size_t& sf_begin, size_t& sf_end) const
{
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
inline bool sample_list<SN>::write(const std::string& out_filename) const
{
  const size_t num_files = m_filenames.size();
  std::ofstream ofstr(out_filename);

  if (!ofstr.good() || (m_samples_per_file.size() != num_files)) {
    return false;
  }

  file_name_list_t::const_iterator it_filename = m_filenames.cbegin();
  typename sample_files_t::const_iterator it_samples = m_samples_per_file.cbegin();

  for (size_t i = 0u; i < num_files; ++i, ++it_filename, ++it_samples) {
    const auto& samples_of_current_file = *it_samples;
    ofstr << *it_filename;
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
  m_filenames.clear();
  m_samples_per_file.clear();
  m_sample_range_per_file.clear();
  m_sample_range_per_part.clear();
}


template <typename SN>
inline bool sample_list<SN>::to_string(size_t p, std::string& sstr)
{
  const size_t num_local_samples = m_sample_range_per_part[p+1] - m_sample_range_per_part[p];

  size_t sf_begin;
  size_t sf_end;

  // Find the range of sample files that covers the range of samples of the partition.
  find_sample_files_of_part(p, sf_begin, sf_end);

  file_name_list_t::const_iterator it_fl_begin = m_filenames.cbegin();
  file_name_list_t::const_iterator it_fl_end = m_filenames.cbegin();
  std::advance(it_fl_begin, sf_begin);
  std::advance(it_fl_end, sf_end);
  file_name_list_t::const_iterator it_fl = it_fl_begin; // filenmae iterator

  typename sample_files_t::const_iterator it_sfl_begin = m_samples_per_file.cbegin();
  typename sample_files_t::const_iterator it_sfl_end = m_samples_per_file.cbegin();
  std::advance(it_sfl_begin, sf_begin);
  std::advance(it_sfl_end, sf_end);

  size_t s_begin = m_sample_range_per_part[p] - m_sample_range_per_file[sf_begin];
  size_t s_end = m_sample_range_per_part[p+1] - m_sample_range_per_file[sf_end];

  if (s_begin >= it_sfl_begin->size() || s_end > it_sfl_end->size()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: incorrect sample indices.");
    return false;
  }

  typename sample_files_t::const_iterator it_sfl = it_sfl_begin;
  typename samples_t::const_iterator it_s = it_sfl->cbegin(); // sample name iterator
  std::advance(it_s, s_begin);
  const size_t b_s_begin = std::min(it_sfl_begin->size(), num_local_samples + s_begin);

  const size_t estimated_len = (sf_end - sf_begin) * it_fl->size() +
    ((to_string(*it_s)).size() + 1u) * static_cast<size_t>(num_local_samples * 1.2);
  sstr.reserve(estimated_len);

  sstr += *(it_fl++);
  for (size_t s = s_begin; s < b_s_begin; ++s, ++it_s) {
    sstr += ' ' + to_string(*it_s);
  }
  sstr += '\n';

  if (sf_begin < sf_end) {
    for (size_t sf = sf_begin+1; sf < sf_end; ++sf) {
      sstr += *(it_fl++);
      for (const auto& s : *(++it_sfl)) {
        sstr += ' ' + to_string(s);
      }
      sstr += '\n';
    }

    typename samples_t::const_iterator it_s = (++it_sfl)->cbegin();
    sstr += *(it_fl++);
    for (size_t s = 0u; s < s_end; ++s, ++it_s) {
      sstr += ' ' + to_string(*it_s);
    }
    sstr += '\n';
  }

  std::cerr << "estimated size vs actual size: " << estimated_len << ' ' << sstr.size() << std::endl;
  std::cerr << "num samples: " << num_local_samples << " samples of rank " << p << std::endl;
  return true;
}


} // end of namespace lbann
