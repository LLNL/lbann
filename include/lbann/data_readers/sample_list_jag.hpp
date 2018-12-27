#ifndef __SAMPLE_LIST_JAG_HPP__
#define __SAMPLE_LIST_JAG_HPP__

#include <iostream>
#include <string>
#include <vector>
#include "lbann/comm.hpp"

namespace lbann {

struct sample_list_header {
  bool m_is_exclusive;
  size_t m_sample_count;
  size_t m_num_files;
  std::string m_file_dir;

  sample_list_header();
};

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

  /// Serialize sample list for all partitions
  bool to_string(std::string& sstr) const;

  /// Write the sample list of partition p
  void write(size_t p, const std::string filename) const;

  /// Write the sample list of each partitions
  void write(const std::string filename) const;

 protected:

  /// Reads a header line from the sample list given as a stream, and use the info string for error message
  std::string read_header_line(std::istream& ifs, const std::string& info) const;

  /// Reads the header of a sample list
  sample_list_header read_header(std::istream& istrm) const;

  /// read the body of exclusive sample list
  void read_exclusive_list(std::istream& istrm);

  /// read the body of inclusive sample list
  void read_inclusive_list(std::istream& istrm);

  /// Reads a samlpe list and populates the internal list
  size_t get_samples_per_file(std::istream& istrm);

  /// Compute the sample index range that each partition covers
  void get_sample_range_per_part();

  /// Compute the samlpe index range that partition p covers
  void get_sample_range_per_part(const size_t p, size_t& sid_start, size_t& sid_end) const;

  /// Add the header info to the gievn string
  void write_header(std::string& sstr) const;

 protected:

  /// The number of partitions to divide samples into
  size_t m_num_partitions;

  /// header info of sample list
  sample_list_header m_header;

  /// Contains list of all sample
  samples_t m_sample_list;

  /// Contains starting sample id of each partition
  std::vector<sample_id_t> m_sample_range_per_part;

  /// indices to m_samples_per_file used for shuffling
  std::vector<unsigned> m_shuffled_indices;

};

void handle_mpi_error(int ierr);

void distribute_sample_list(const sample_list<std::string>& sn,
                            std::string& my_samples,
                            lbann_comm& comm);

} // end of namespace

#include "sample_list_jag_impl.hpp"

#endif // __SAMPLE_LIST_JAG_HPP__
