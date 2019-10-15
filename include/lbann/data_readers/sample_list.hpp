#ifndef __SAMPLE_LIST_HPP__
#define __SAMPLE_LIST_HPP__

#include <iostream>
#include <string>
#include <vector>
#include <functional>

#include "lbann/comm.hpp"

#include "lbann/utils/file_utils.hpp"
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/utility.hpp>

namespace lbann {

static const std::string sample_exclusion_list = "CONDUIT_HDF5_EXCLUSION";
static const std::string sample_inclusion_list = "CONDUIT_HDF5_INCLUSION";

struct sample_list_header {
  bool m_is_exclusive;
  /// Number of included samples
  size_t m_included_sample_count;
  /// Number of excluded samples
  size_t m_excluded_sample_count;
  size_t m_num_files;
  std::string m_file_dir;
  std::string m_sample_list_filename;

  sample_list_header();

  bool is_exclusive() const;
  size_t get_sample_count() const;
  size_t get_num_files() const;
  const std::string& get_sample_list_filename() const;
  const std::string& get_file_dir() const;
  template <class Archive> void serialize( Archive & ar ) {
    ar(m_is_exclusive, m_included_sample_count, m_excluded_sample_count, m_num_files, m_file_dir, m_sample_list_filename);
  }
};

template <typename sample_name_t>
class sample_list {
 public:
  /// The type for the index assigned to each sample file
  using sample_file_id_t = std::size_t;
  /** To describe a sample as the id of the file to which it belongs.
    * Each file contains only one sample. */
  using sample_t = std::template pair<sample_file_id_t, sample_name_t>;
  /// Type for the list of samples
  using samples_t = std::template vector< sample_t >;
  /// Mapping of the file index to the filename
  using file_id_stats_v_t = std::vector< std::string >;

  sample_list();
  virtual ~sample_list();
  sample_list(const sample_list& rhs);
  sample_list& operator=(const sample_list& rhs);
  sample_list& copy(const sample_list& rhs);

  void copy_members(const sample_list& rhs);

  /// Load a sample list file
  void load(const std::string& samplelist_file, size_t stride=1, size_t offset=0);

  /// Load the header of a sample list file
  sample_list_header load_header(const std::string& samplelist_file) const;

  /// Restore a sample list from a serialized string
  void load_from_string(const std::string& samplelist);

  /// Tells how many samples in the list
  virtual size_t size() const;

  /// Tells how many sample files are there
  virtual size_t get_num_files() const;

  /// Tells if the internal list is empty
  bool empty() const;

  /// Serialize to and from an archive using the cereal library
  template <class Archive> void serialize( Archive & ar );

  /// Serialize sample list
  virtual bool to_string(std::string& sstr) const;

  /// Write the sample list
  void write(const std::string filename) const;

  /// Allow read-only access to the internal list data
  const samples_t& get_list() const;

  /// Allow the read-only access to the list header
  const sample_list_header& get_header() const;

  /// Allow read-only access to the metadata of the idx-th sample in the list
  const sample_t& operator[](size_t idx) const;

  virtual const std::string& get_samples_filename(sample_file_id_t id) const;

  const std::string& get_samples_dirname() const;

  void all_gather_archive(const std::string &archive, std::vector<std::string>& gathered_archive, lbann_comm& comm);
  template<typename T> size_t all_gather_field(T data, std::vector<T>& gathered_data, lbann_comm& comm);
  virtual void all_gather_packed_lists(lbann_comm& comm);

 protected:

  /// Reads a header line from the sample list given as a stream, and use the info string for error message
  std::string read_header_line(std::istream& ifs, const std::string& filename, const std::string& info) const;

  /// Reads the header of a sample list
  sample_list_header read_header(std::istream& istrm, const std::string& filename) const;

  /// read the body of a sample list, which is the list of sample files, where each file contains a single sample.
  virtual void read_sample_list(std::istream& istrm, size_t stride=1, size_t offset=0);

  /// Assign names to samples when there is only one sample per file without a name.
  virtual void assign_samples_name();

  /// Reads a sample list and populates the internal list
  size_t get_samples_per_file(std::istream& istrm, const std::string& filename, size_t stride=1, size_t offset=0);

  /// Add the header info to the given string
  void write_header(std::string& sstr, size_t num_files) const;

  /// Get the number of total/included/excluded samples
  virtual void get_num_samples(size_t& total, size_t& included, size_t& excluded) const;

  virtual void set_samples_filename(sample_file_id_t id, const std::string& filename);

 protected:
  /// header info of sample list
  sample_list_header m_header;

 private:
  /// List of all samples with a file identifier and sample name for each sample
  samples_t m_sample_list;

  /// Maps sample's file id to file names, file descriptors, and use counts
  file_id_stats_v_t m_file_id_stats_map;

};

void handle_mpi_error(int ierr);

template<typename T>
inline T uninitialized_sample_name();

} // end of namespace

#include "sample_list_impl.hpp"

#endif // __SAMPLE_LIST_HPP__
