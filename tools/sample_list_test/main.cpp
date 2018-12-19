#include "hdf5.h"
#include "conduit/conduit.hpp"
#include "conduit/conduit_relay.hpp"
#include "conduit/conduit_relay_io_hdf5.hpp"
#include "sample_list.hpp"
#include <unordered_set>

namespace lbann {
template<typename SN>
inline size_t sample_list<SN>::get_samples_per_hdf5_file(std::istream& ifstr)
{
  if (!ifstr.good()) {
    throw lbann_exception(std::string{} + __FILE__ + " " + std::to_string(__LINE__)
                          + " :: unable to read from the data input stream");
  }

  std::string line;

  size_t total_num_samples = 0u;
  m_samples_per_file.clear();

  if(!ifstr.good()) { std::cerr << "Unable to read HDF5 sample index format\n"; return 0u;}
  std::getline(ifstr, line);
  std::stringstream header1(line);
  int sample_count, num_files;
  header1 >> sample_count;
  header1 >> num_files;

  // If we know the number of data files, we can reserve the space here for vector
  // but not for list.
  m_samples_per_file.reserve(num_files);


  if(!ifstr.good()) { std::cerr << "stream not good\n"; return 0u; }
  std::getline(ifstr, line);
  std::stringstream header2(line);
  std::string file_dir;
  header2 >> file_dir;
  const std::string whitespaces (" \t\f\v\n\r");

  std::cout << "Reading " << sample_count << " samples from " << num_files << " files and dir " << file_dir << std::endl;
  while (std::getline(ifstr, line)) {
    const size_t end_of_str = line.find_last_not_of(whitespaces);
    if (end_of_str == std::string::npos) {
      continue;
    }
    std::stringstream sstr(line.substr(0, end_of_str + 1));
    std::string filename;
    int valid_samples;
    int invalid_samples;
    std::unordered_set<size_t> invalid_sample_indices;
    invalid_sample_indices.reserve(valid_samples + invalid_samples);

    sstr >> filename >> valid_samples >> invalid_samples;
    while(!sstr.eof()) {
      size_t index;
      sstr >> index;
      invalid_sample_indices.insert(index);
    }

    std::cout << "I am going to load the file " << filename << " which has " << valid_samples << " valid samples and " << invalid_samples << std::endl;
    m_samples_per_file.emplace_back();
    (m_samples_per_file.back()).first = file_dir + "/" + filename;
    auto& samples_of_current_file = (m_samples_per_file.back()).second;
    samples_of_current_file.reserve(valid_samples + invalid_samples);

    std::string conduit_file_path = file_dir + "/" + filename;
    hid_t hdf5_file_hnd = conduit::relay::io::hdf5_open_file_for_read( conduit_file_path );
    if (hdf5_file_hnd <= static_cast<hid_t>(0)) {
      std::cout << "Opening the file didn't work" << std::endl;
      continue; // skipping the file
    }
    std::vector<std::string> sample_names;
    conduit::relay::io::hdf5_group_list_child_names(hdf5_file_hnd, "/", /*samples_of_current_file*/sample_names);
    // std::cout << " I have found that there are " << samples_of_current_file.size() << " samples" << std::endl;
    //    for(auto s : samples_of_current_file) {
    size_t i = 0u;
    for(auto s : sample_names) {
      std::unordered_set<size_t>::const_iterator found = invalid_sample_indices.find(i++);
      if (found == invalid_sample_indices.cend()) {
        continue;
      }
      std::cout << " I have found a sample " << s << std::endl;
      m_sample_list.emplace_back(sample_t(conduit_file_path, s));
      //      m_sample_list.emplace_back(std::make_tuple(hdf5_file_hnd/*conduit_file_path*/, s));
    }
    for(auto s : m_sample_list) {
    //  std::cout << "I have found a sample " << s.first << " and " << s.second << std::endl;
      //      std::cout << "I have found a sample " << std::get<0>(s) << " and " << std::get<1>(s) << std::endl;
    }

    const size_t num_samples_of_current_file = samples_of_current_file.size();
    total_num_samples += num_samples_of_current_file;
  }

  return total_num_samples;
}

}

using namespace lbann;

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cout << "usage : > " << argv[0] << " sample_list_file num_ranks" << std::endl;
    return 0;
  }

  // The file name of the sample file list
  std::string sample_list_file(argv[1]);

  // The number of ranks to divide samples with
  int num_ranks = atoi(argv[2]);

  sample_list<> sn;
  //sample_list<int> sn;
  sn.set_num_partitions(num_ranks);
/*
  sn.load(sample_list_file);
*/
  std::ifstream istr(sample_list_file);
  sn.get_samples_per_hdf5_file(istr);
  // Create a copy of the sample file list read for verification
  sn.write("copy_of_" + sample_list_file);


  for(int i = 0; i < num_ranks; ++i) {
    std::string sstr;
    sn.to_string(static_cast<size_t>(i), sstr);
    std::cout << sstr;
  }
  return 0;
}


