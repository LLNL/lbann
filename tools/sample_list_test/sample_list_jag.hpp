#ifndef __SAMPLE_LIST_JAG_HPP__
#define __SAMPLE_LIST_JAG_HPP__

#include <iostream>
#include <string>
#include <vector>

namespace lbann {

class sample_list_jag {
 public:
  /// The type of the native identifier of a sample rather than an arbitrarily assigned index
  using sample_name_t = std::string;
  /// The type for arbitrarily assigned index
  using sample_id_t = size_t;
  /// To describe a sample as a pair of the file to which it belongs and its name
  using sample_t = std::pair<std::string, sample_name_t>;
  /// Type for the list of samples
  using samples_t = std::vector< sample_t >;

  sample_list_jag() : m_num_partitions(1u) {}

  /// Set the number of partitions and clear internal states
  void set_num_partitions(size_t n);

  /// Load a sample list from a file
  void load(const std::string& samplelist_file);

  /// Extract a sample list from a serialized sample list in a string
  void load_from_string(const std::string& samplelist);

  /// Write the current sample list into a file
  //bool write(const std::string& out_filename) const;

  /// Clear internal states
  void clear();

  /// Serialize sample list for a partition
  bool to_string(size_t p, std::string& sstr) const;

  /// Reads a samlpe list and populates the internal list
  size_t get_samples_per_file(std::istream& ifstr);

  void get_sample_range_per_part();

 protected:

  /// The number of partitions to divide samples into
  size_t m_num_partitions;

  /// The root directory of data file paths
  std::string m_file_dir;

  /// Contains list of all sample
  samples_t m_sample_list;

  /// Contains starting sample id of each partition
  std::vector<sample_id_t> m_sample_range_per_part;

  /// indices to m_samples_per_file used for shuffling
  std::vector<unsigned> m_shuffled_indices;

};


} // end of namespace

#include "sample_list_jag_impl.hpp"

#endif // __SAMPLE_LIST_JAG_HPP__
