#ifndef __SAMPLE_LIST_HPP__
#define __SAMPLE_LIST_HPP__

#include <iostream>
#include <string>
#include <vector>
#include <functional>

#ifndef _JAG_OFFLINE_TOOL_MODE_
#include "lbann/comm.hpp"
#else
#include <mpi.h>
#endif

#include "lbann/utils/file_utils.hpp"
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/utility.hpp>

/// Number of system and other files that may be open during execution
#define LBANN_MAX_OPEN_FILE_MARGIN 128
#define LBANN_MAX_OPEN_FILE_RETRY 3

namespace lbann {

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

static const std::string sample_exclusion_list = "CONDUIT_HDF5_EXCLUSION";
static const std::string sample_inclusion_list = "CONDUIT_HDF5_INCLUSION";

template <typename file_handle_t, typename sample_name_t>
class sample_list {
 public:
  /// The type of the native identifier of a sample rather than an arbitrarily assigned index
  //using sample_name_t = std::string;
  /// The type for arbitrarily assigned index
  using sample_file_id_t = std::size_t;
  /// To describe a sample as a pair of the file to which it belongs and its name
  //  using sample_t = std::pair<std::string, sample_name_t>;
  using sample_t = std::pair<sample_file_id_t, sample_name_t>;
  /// Statistics for each file used by the sample list: includes the file name, file descriptor, and
  /// and a queue of each step and substep when data will be loaded from the file
  using file_id_stats_t = std::tuple<std::string, file_handle_t, std::deque<std::pair<int,int>>>;

  /// Type for the list of samples
  using samples_t = std::template vector< sample_t >;
  /// Mapping of the file index to the statistics for each file
  using file_id_stats_v_t = std::vector< file_id_stats_t >; // rename to sample_to_file_v or something
  /// Type for the map of file descriptors to usage step and substep
  using fd_use_map_t = std::template pair<sample_file_id_t, std::pair<int,int>>;

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

  /// Extract a sample list from a serialized sample list in a string
  void load_from_string(const std::string& samplelist);

  /// Tells how many samples in the list
  size_t size() const;

  /// Tells if the internal list is empty
  bool empty() const;

  /// Clear internal states
  void clear();

  template <class Archive> void save( Archive & ar ) const;
  template <class Archive> void load( Archive & ar );

  /// Check if a sample index is in the valid range
  bool check_index(size_t idx) const;

  /// Serialize sample list
  bool to_string(std::string& sstr) const;

  /// Write the sample list
  void write(const std::string filename) const;

  /// Allow read-only access to the internal list data
  const samples_t& get_list() const;

  /// Allow the read-only access to the list header
  const sample_list_header& get_header() const;

  /// Allow read-only access to the metadata of the idx-th sample in the list
  const sample_t& operator[](size_t idx) const;

  const std::string& get_samples_filename(sample_file_id_t id) const;

  const std::string& get_samples_dirname() const;

  file_handle_t get_samples_file_handle(sample_file_id_t id) const;

  void set_samples_filename(sample_file_id_t id, const std::string& filename);

  void set_files_handle(const std::string& filename, file_handle_t h);

  void manage_open_file_handles(sample_file_id_t id, bool pre_open_fd = false);

  file_handle_t open_samples_file_handle(const size_t i, bool pre_open_fd = false);

  void close_if_done_samples_file_handle(const size_t i);

  void all_gather_archive(const std::string &archive, std::vector<std::string>& gathered_archive, lbann_comm& comm);
  template<typename T> size_t all_gather_field(T data, std::vector<T>& gathered_data, lbann_comm& comm);
  void all_gather_packed_lists(lbann_comm& comm);

  void compute_epochs_file_usage(const std::vector<int>& shufled_indices, int mini_batch_size, const lbann_comm& comm);

  virtual bool is_file_handle_valid(const file_handle_t& h) const;

 protected:

  /// Reads a header line from the sample list given as a stream, and use the info string for error message
  std::string read_header_line(std::istream& ifs, const std::string& filename, const std::string& info) const;

  /// Reads the header of a sample list
  sample_list_header read_header(std::istream& istrm, const std::string& filename) const;

  /// Get the list of samples from a specific type of bundle file
  virtual void obtain_sample_names(file_handle_t& h, std::vector<std::string>& sample_names) const;

  /// Get the list of samples that exist in a bundle file
  file_handle_t get_bundled_sample_names(std::string file_path, std::vector<std::string>& sample_names, size_t included_samples, size_t excluded_samples);

  /// read the body of exclusive sample list
  void read_exclusive_list(std::istream& istrm, size_t stride=1, size_t offset=0);

  /// read the body of inclusive sample list
  void read_inclusive_list(std::istream& istrm, size_t stride=1, size_t offset=0);

  /// Reads a sample list and populates the internal list
  size_t get_samples_per_file(std::istream& istrm, const std::string& filename, size_t stride=1, size_t offset=0);

  /// Add the header info to the given string
  void write_header(std::string& sstr, size_t num_files) const;

  static bool pq_cmp(fd_use_map_t left, fd_use_map_t right) {
    return ((left.second).first < (right.second).first) ||
           (((left.second).first == (right.second).first) &&
            ((left.second).second < (right.second).second)); }

  virtual file_handle_t open_file_handle_for_read(const std::string& file_path);
  virtual void close_file_handle(file_handle_t& h);
  virtual void clear_file_handle(file_handle_t& h);

  /// Maps sample's file id to file names, file descriptors, and use counts
  file_id_stats_v_t m_file_id_stats_map;

 private:
  /// header info of sample list
  sample_list_header m_header;

  /// List of all samples with a file identifier and sample name for each sample
  samples_t m_sample_list;

  /// Track the number of samples per file
  std::unordered_map<std::string, size_t> m_file_map;

  /// Track the number of open file descriptors and when they will be used next
  std::deque<fd_use_map_t> m_open_fd_pq;

  size_t m_max_open_files;
};

void handle_mpi_error(int ierr);

template<typename T>
inline T uninitialized_file_handle();

#ifndef _JAG_OFFLINE_TOOL_MODE_
template <typename file_handle_t, typename sample_name_t>
void distribute_sample_list(const sample_list<file_handle_t, sample_name_t>& sn,
                            std::string& my_samples,
                            lbann_comm& comm);
#else
template <typename file_handle_t, typename sample_name_t>
void distribute_sample_list(const sample_list<file_handle_t, sample_name_t>& sn,
                            std::string& my_samples,
                            MPI_Comm& comm);
#endif

} // end of namespace

#include "sample_list_impl.hpp"

#endif // __SAMPLE_LIST_HPP__
