#ifndef __SAMPLE_LIST_JAG_HPP__
#define __SAMPLE_LIST_JAG_HPP__

#include <iostream>
#include <string>
#include <vector>
#include <functional>

#ifndef _JAG_OFFLINE_TOOL_MODE_
#include "lbann/comm.hpp"
#else
#include <mpi.h>
#endif

namespace lbann {

struct sample_list_header {
  bool m_is_exclusive;
  size_t m_sample_count;
  size_t m_num_files;
  std::string m_file_dir;

  sample_list_header();

  bool is_exclusive() const;
  size_t get_sample_count() const;
  size_t get_num_files() const;
  const std::string& get_file_dir() const;
};

/**
 * Maps a global index of a sample list to a local index.
 * When managing the sample list in a distributed fashion, with which every
 * one has the same copy (the whole global list), m_partition_offset must be
 * zero. In this case, the local index is the same as the global index.
 * When managing the sample list in a centralized fashion, with which each
 * has a portion of the list that corresponds to the only samples it needs,
 * a global index is subtracted by m_partition_offset for local indexing.
 */
struct sample_list_indexer {
  sample_list_indexer();
  size_t operator()(size_t idx) const;

  void set_partition_offset(size_t o);
  size_t get_partition_offset() const;
  bool check_index(size_t i) const;

  size_t m_partition_offset;
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

  sample_list_jag();

  /// Set the number of partitions and clear internal states
  void set_num_partitions(size_t n);

  /// Set the index mapping function
  void set_indexer(const sample_list_indexer& indexer);

  /// Get the index mapping function
  const sample_list_indexer& get_indexer() const;

  /// Load a sample list file
  void load(const std::string& samplelist_file);

  /// Load the header of a sample list file
  sample_list_header load_header(const std::string& samplelist_file) const;

  /// Extract a sample list from a serialized sample list in a string
  void load_from_string(const std::string& samplelist);

  /// Tells how many samples in the list 
  size_t size() const;

  /// Tells if the internal list is empty
  bool empty() const;

  /// Clear internal states
  void clear();

  /// Check if a sample index is in the valid range
  bool check_index(size_t idx) const;

  /// Serialize sample list for a partition
  bool to_string(size_t p, std::string& sstr) const;

  /// Serialize sample list for all partitions
  bool to_string(std::string& sstr) const;

  /// Write the sample list of partition p
  void write(size_t p, const std::string filename) const;

  /// Write the sample list of each partitions
  void write(const std::string filename) const;

  /// Allow read-only access to the internal list data
  const samples_t& get_list() const;

  /// Copy the internal list data for partition p
  bool get_list(size_t p, samples_t& l_p) const;

  /// Allow read-only access to the internal list data for partition p via iterators
  std::pair<samples_t::const_iterator, samples_t::const_iterator> get_list(size_t p) const;

  /// Allow the read-only access to the list header
  const sample_list_header& get_header() const;

  /// Allow read-only access to the metadata of the idx-th sample in the list
  const sample_t& operator[](size_t idx) const;

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

  /// Maps a global index to a local index
  sample_list_indexer m_indexer;
};

void handle_mpi_error(int ierr);

#ifndef _JAG_OFFLINE_TOOL_MODE_
void distribute_sample_list(const sample_list_jag& sn,
                            std::string& my_samples,
                            lbann_comm& comm);
#else
void distribute_sample_list(const sample_list_jag& sn,
                            std::string& my_samples,
                            MPI_Comm& comm);
#endif

} // end of namespace

#include "sample_list_jag_impl.hpp"

#endif // __SAMPLE_LIST_JAG_HPP__
