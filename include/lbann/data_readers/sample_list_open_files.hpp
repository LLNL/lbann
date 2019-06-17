#ifndef __SAMPLE_LIST_OPEN_FILES_HPP__
#define __SAMPLE_LIST_OPEN_FILES_HPP__

#include "sample_list.hpp"

/// Number of system and other files that may be open during execution
#define LBANN_MAX_OPEN_FILE_MARGIN 128
#define LBANN_MAX_OPEN_FILE_RETRY 3

namespace lbann {

template <typename sample_name_t, typename file_handle_t>
class sample_list_open_files : public sample_list<sample_name_t> {
 public:
  /// The type for the index assigned to each sample file
  using sample_file_id_t = std::size_t;
  /** To describe a sample as a pair of the file to which it belongs and its name
      Each file may contain multiple samples. */
  using sample_t = std::pair<sample_file_id_t, sample_name_t>;
  /// Information for each file used by the sample list: includes the file name, file descriptor, and
  /// and a queue of each step and substep when data will be loaded from the file
  using file_id_stats_t = std::tuple<std::string, file_handle_t, std::deque<std::pair<int,int>>>;

  /// Type for the list of samples
  using samples_t = std::template vector< sample_t >;
  /// Mapping of the file index to the statistics for each file
  using file_id_stats_v_t = std::vector< file_id_stats_t >; // rename to sample_to_file_v or something
  /// Type for the map of file descriptors to usage step and substep
  using fd_use_map_t = std::template pair<sample_file_id_t, std::pair<int,int>>;

  sample_list_open_files();
  virtual ~sample_list_open_files();
  /** Copy constructor repllicates all the member variables as they are except
    * the file information vector, for which only the file name is copied. */
  sample_list_open_files(const sample_list_open_files& rhs);
  /** assignemnt operation repllicates all the member variables as they are except
    * the file information vector, for which only the file name is copied. */
  sample_list_open_files& operator=(const sample_list_open_files& rhs);
  sample_list_open_files& copy(const sample_list_open_files& rhs);

  void copy_members(const sample_list_open_files& rhs);

  /// Tells how many samples in the list
  size_t size() const override;

  /// Tells how many sample files are there
  size_t get_num_files() const override;

  using sample_list<sample_name_t>::load;
  /// Emit a serialized archive using the cereal library
  template <class Archive> void save( Archive & ar ) const;
  /// Restore the member variables from a given archrive serialized by the cereal library
  template <class Archive> void load( Archive & ar );

  /// Serialize this sample list into an std::string object
  bool to_string(std::string& sstr) const override;

  /// Allow read-only access to the internal list data
  const samples_t& get_list() const;

  /// Allow read-only access to the metadata of the idx-th sample in the list
  const sample_t& operator[](size_t idx) const;

  const std::string& get_samples_filename(sample_file_id_t id) const override;

  file_handle_t get_samples_file_handle(sample_file_id_t id) const;

  void set_files_handle(const std::string& filename, file_handle_t h);

  void delete_file_handle_pq_entry(sample_file_id_t id);

  void manage_open_file_handles(sample_file_id_t id, bool pre_open_fd = false);

  file_handle_t open_samples_file_handle(const size_t i, bool pre_open_fd = false);

  virtual void close_if_done_samples_file_handle(const size_t i);

  void compute_epochs_file_usage(const std::vector<int>& shufled_indices, int mini_batch_size, const lbann_comm& comm);

  virtual bool is_file_handle_valid(const file_handle_t& h) const = 0;

  void all_gather_packed_lists(lbann_comm& comm) override;

 protected:

  void set_samples_filename(sample_file_id_t id, const std::string& filename) override;

  /// Get the list of samples from a specific type of bundle file
  virtual void obtain_sample_names(file_handle_t& h, std::vector<std::string>& sample_names) const = 0;

  /// Get the list of samples that exist in a bundle file
  file_handle_t get_bundled_sample_names(std::string file_path, std::vector<std::string>& sample_names, size_t included_samples, size_t excluded_samples);

  /// read the body of exclusive sample list
  void read_exclusive_list(std::istream& istrm, size_t stride=1, size_t offset=0);

  /// read the body of inclusive sample list
  void read_inclusive_list(std::istream& istrm, size_t stride=1, size_t offset=0);

  /// read the body of a sample list
  void read_sample_list(std::istream& istrm, size_t stride=1, size_t offset=0) override;

  void assign_samples_name() override {}

  /// Get the number of total/included/excluded samples
  void get_num_samples(size_t& total, size_t& included, size_t& excluded) const override;

  static bool pq_cmp(fd_use_map_t left, fd_use_map_t right) {
    return ((left.second).first < (right.second).first) ||
           (((left.second).first == (right.second).first) &&
            ((left.second).second < (right.second).second)); }

  virtual file_handle_t open_file_handle_for_read(const std::string& file_path) = 0;
  virtual void close_file_handle(file_handle_t& h) = 0;
  virtual void clear_file_handle(file_handle_t& h) = 0;

 private:
  using sample_list<sample_name_t>::serialize;
  template <class Archive> void serialize( Archive & ar ) = delete;

 protected:
  using sample_list<sample_name_t>::m_header;

  /// Maps sample's file id to file names, file descriptors, and use counts
  file_id_stats_v_t m_file_id_stats_map;

 private:
  /// List of all samples with a file identifier and sample name for each sample
  samples_t m_sample_list;

  /// Track the number of samples per file
  std::unordered_map<std::string, size_t> m_file_map;

  /// Track the number of open file descriptors and when they will be used next
  std::deque<fd_use_map_t> m_open_fd_pq;

  size_t m_max_open_files;
};

template<typename T>
inline T uninitialized_file_handle();

} // end of namespace

#include "sample_list_open_files_impl.hpp"

#endif // __SAMPLE_LIST_OPEN_FILES_HPP__
