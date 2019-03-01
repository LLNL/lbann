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

#include "lbann/utils/file_utils.hpp"
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/utility.hpp>
#include "conduit/conduit_relay_io_hdf5.hpp"

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

static const std::string conduit_hdf5_exclusion_list = "CONDUIT_HDF5_EXCLUSION";
static const std::string conduit_hdf5_inclusion_list = "CONDUIT_HDF5_INCLUSION";

class sample_list_jag {
 public:
  /// The type of the native identifier of a sample rather than an arbitrarily assigned index
  using sample_name_t = std::string;
  /// The type for arbitrarily assigned index
  using sample_file_id_t = std::size_t;
  /// To describe a sample as a pair of the file to which it belongs and its name
  //  using sample_t = std::pair<std::string, sample_name_t>;
  using sample_t = std::pair<sample_file_id_t, sample_name_t>;
  /// Statistics for each file used by the sample list: includes the file name, file descriptor, and
  /// and a queue of each step and substep when data will be loaded from the file
  using file_id_stats_t = std::tuple<std::string, hid_t, std::deque<std::pair<int,int>>>;

  /// Type for the list of samples
  using samples_t = std::vector< sample_t >;
  /// Mapping of the file index to the statistics for each file
  using file_id_stats_v_t = std::vector< file_id_stats_t >; // rename to sample_to_file_v or something
  /// Type for the map of file descriptors to usage step and substep
  using fd_use_map_t = std::pair<sample_file_id_t, std::pair<int,int>>;

  sample_list_jag();
  ~sample_list_jag();

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

  template <class Archive> void serialize( Archive & ar );

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

  const std::string& get_samples_filename(sample_file_id_t id) const {
    return std::get<0>(m_file_id_stats_map[id]);
  }

  const std::string& get_samples_dirname() const {
    return m_header.get_file_dir();
  }

  hid_t get_samples_hdf5_handle(sample_file_id_t id) const {
    hid_t h = std::get<1>(m_file_id_stats_map[id]);
    return h;
  }

  void set_samples_filename(sample_file_id_t id, const std::string& filename) {
    std::get<0>(m_file_id_stats_map[id]) = filename;
  }

  void set_files_hdf5_handle(const std::string& filename, hid_t h) {
    sample_file_id_t id = 0;
    for (auto&& e : m_file_id_stats_map) {
      if(std::get<0>(e) == filename) {
        std::get<1>(e) = h;
        break;
      }
      id++;
    }
    manage_open_hdf5_handles(id, true);
  }

  void manage_open_hdf5_handles(sample_file_id_t id, bool pre_open_fd = false) {
    /// When we enter this function the priority queue is either empty or a heap
    if(!m_open_fd_pq.empty()) {
      if(m_open_fd_pq.size() > m_max_open_files) {
        auto& f = m_open_fd_pq.front();
        auto& victim = m_file_id_stats_map[f.first];
        hid_t victim_fd = std::get<1>(victim);
        std::pop_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
        m_open_fd_pq.pop_back();
        if(victim_fd > 0) {
          conduit::relay::io::hdf5_close_file(victim_fd);
          std::get<1>(victim) = 0;
        }
      }
    }

    /// Before we can enqueue the any new access times for this descriptor, remove any
    /// earlier descriptor
    std::sort_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
    if(m_open_fd_pq.front().first == id) {
      m_open_fd_pq.pop_front();
    }
    std::make_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);

    auto& e = m_file_id_stats_map[id];
    auto& file_access_queue = std::get<2>(e);
    if(!file_access_queue.empty()) {
      if(!pre_open_fd) {
        file_access_queue.pop_front();
      }
    }
    if(!file_access_queue.empty()) {
      m_open_fd_pq.emplace_back(std::make_pair(id,file_access_queue.front()));
    }else {
      /// If there are no future access of the file place a terminator entry to track
      /// the open file, but is always sorted to the top of the heap
      m_open_fd_pq.emplace_back(std::make_pair(id,std::make_pair(INT_MAX,id)));
    }
    std::push_heap(m_open_fd_pq.begin(), m_open_fd_pq.end(), pq_cmp);
    return;
  }

  hid_t open_samples_hdf5_handle(const size_t i, bool pre_open_fd = false) {
    const sample_t& s = m_sample_list[i];
    sample_file_id_t id = s.first;
    hid_t h = get_samples_hdf5_handle(id);
    if (h <= static_cast<hid_t>(0)) {
      const std::string& file_name = get_samples_filename(id);
      const std::string conduit_file_path = add_delimiter(get_samples_dirname()) + file_name;
      if (file_name.empty() || !check_if_file_exists(conduit_file_path)) {
        LBANN_ERROR(std::string{} + " :: data file '" + conduit_file_path + "' does not exist.");
      }
      bool retry = false;
      int retry_cnt = 0;
      do {
        try {
          h = conduit::relay::io::hdf5_open_file_for_read( conduit_file_path );
        }catch (conduit::Error const& e) {
          LBANN_WARNING(" :: trying to open the file " + conduit_file_path + " and got " + e.what());
          retry = true;
          retry_cnt++;
        }
      }while(retry && retry_cnt < 3);

      if (h <= static_cast<hid_t>(0)) {
        LBANN_ERROR(std::string{} + " :: data file '" + conduit_file_path + "' could not be opened.");
      }
      auto& e = m_file_id_stats_map[id];
      std::get<1>(e) = h;
    }
    manage_open_hdf5_handles(id, pre_open_fd);
    return h;
  }

  void close_if_done_samples_hdf5_handle(const size_t i) {
    const sample_t& s = m_sample_list[i];
    sample_file_id_t id = s.first;
    hid_t h = get_samples_hdf5_handle(id);
    if (h > static_cast<hid_t>(0)) {
      auto& e = m_file_id_stats_map[id];
      auto& file_access_queue = std::get<2>(e);
      if(file_access_queue.empty()) {
        conduit::relay::io::hdf5_close_file(std::get<1>(e));
        std::get<1>(e) = 0;
      }
    }
  }

  void all_gather_archive(const std::string &archive, std::vector<std::string>& gathered_archive, lbann_comm& comm);
  template<typename T> size_t all_gather_field(T data, std::vector<T>& gathered_data, lbann_comm& comm);
  void all_gather_packed_lists(lbann_comm& comm);

  void compute_epochs_file_usage(const std::vector<int>& shufled_indices, int mini_batch_size, const lbann_comm& comm);

 protected:

  /// Reads a header line from the sample list given as a stream, and use the info string for error message
  std::string read_header_line(std::istream& ifs, const std::string& filename, const std::string& info) const;

  /// Reads the header of a sample list
  sample_list_header read_header(std::istream& istrm, const std::string& filename) const;

  /// Get the list of samples that exist in a conduit bundle
  hid_t get_conduit_bundle_samples(std::string conduit_file_path, std::vector<std::string>& sample_names, size_t included_samples, size_t excluded_samples);

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

 private:
  /// header info of sample list
  sample_list_header m_header;

  /// List of all samples with a file identifier and sample name for each sample
  samples_t m_sample_list;

  /// Maps sample's file id to file names, file descriptors, and use counts
  file_id_stats_v_t m_file_id_stats_map;

  /// Track the number of samples per file
  std::unordered_map<std::string, size_t> m_file_map;

  /// Track the number of open file descriptors and when they will be used next
  std::deque<fd_use_map_t> m_open_fd_pq;

  size_t m_max_open_files;
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
