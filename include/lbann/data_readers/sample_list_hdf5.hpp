#ifndef __SAMPLE_LIST_HDF5_HPP__
#define __SAMPLE_LIST_HDF5_HPP__

#include "sample_list.hpp"

namespace lbann {

template <typename file_handle_t, typename sample_name_t>
class sample_list_hdf5 : public sample_list<file_handle_t, sample_name_t> {
 public:
  using typename sample_list<file_handle_t, sample_name_t>::sample_file_id_t;
  using typename sample_list<file_handle_t, sample_name_t>::sample_t;
  using typename sample_list<file_handle_t, sample_name_t>::samples_t;
  using typename sample_list<file_handle_t, sample_name_t>::file_id_stats_t;
  using typename sample_list<file_handle_t, sample_name_t>::file_id_stats_v_t;
  using typename sample_list<file_handle_t, sample_name_t>::fd_use_map_t;

  sample_list_hdf5();
  ~sample_list_hdf5() override;

  void manage_open_file_handles(sample_file_id_t id, bool pre_open_fd = false) override;

  file_handle_t open_samples_file_handle(const size_t i, bool pre_open_fd = false) override;

  void close_if_done_samples_file_handle(const size_t i) override;

  void all_gather_packed_lists(lbann_comm& comm) override;

  void compute_epochs_file_usage(const std::vector<int>& shufled_indices, int mini_batch_size, const lbann_comm& comm) override;

 protected:
  /// Get the list of samples that exist in a bundle file
  file_handle_t get_bundled_sample_names(std::string file_path, std::vector<std::string>& sample_names, size_t included_samples, size_t excluded_samples) override;

  bool is_file_handle_valid(const file_handle_t& h) const override;
  void clear_file_handle(file_handle_t& h) override;
  void close_and_clear_file_handle(file_handle_t& h) override;

 protected: // accessors for private member variables of the base class
  sample_list_header& my_header() { return sample_list<file_handle_t, sample_name_t>::my_header(); }
  const sample_list_header& my_header() const { return sample_list<file_handle_t, sample_name_t>::my_header(); }
  samples_t& my_sample_list() { return sample_list<file_handle_t, sample_name_t>::my_sample_list(); }
  const samples_t& my_sample_list() const { return sample_list<file_handle_t, sample_name_t>::my_sample_list(); }
  file_id_stats_v_t& my_file_id_stats_map() { return sample_list<file_handle_t, sample_name_t>::my_file_id_stats_map(); }
  const file_id_stats_v_t& my_file_id_stats_map() const { return sample_list<file_handle_t, sample_name_t>::my_file_id_stats_map(); }
  std::unordered_map<std::string, size_t>& my_file_map() { return sample_list<file_handle_t, sample_name_t>::my_file_map(); }
  const std::unordered_map<std::string, size_t>& my_file_map() const { return sample_list<file_handle_t, sample_name_t>::my_file_map(); }
  std::deque<fd_use_map_t>& my_open_fd_pq() { return sample_list<file_handle_t, sample_name_t>::my_open_fd_pq(); }
  const std::deque<fd_use_map_t>& my_open_fd_pq() const { return sample_list<file_handle_t, sample_name_t>::my_open_fd_pq(); }
  size_t& my_max_open_files() { return sample_list<file_handle_t, sample_name_t>::my_max_open_files(); }
  const size_t& my_max_open_files() const { return sample_list<file_handle_t, sample_name_t>::my_max_open_files(); }
};


template <typename file_handle_t, typename sample_name_t>
inline sample_list_hdf5<file_handle_t, sample_name_t>::sample_list_hdf5()
: sample_list<file_handle_t, sample_name_t>() {} 

template <typename file_handle_t, typename sample_name_t>
inline sample_list_hdf5<file_handle_t, sample_name_t>::~sample_list_hdf5() {
  // Close the existing open files
  for(auto& f : my_file_id_stats_map()) {
    file_handle_t& h = std::get<1>(f);
    close_and_clear_file_handle(h);
  }
  my_file_id_stats_map().clear();
  my_open_fd_pq().clear();
}

template <typename file_handle_t, typename sample_name_t>
inline file_handle_t sample_list_hdf5<file_handle_t, sample_name_t>
::get_bundled_sample_names(std::string file_path,
                           std::vector<std::string>& sample_names,
                           size_t included_samples,
                           size_t excluded_samples) {
  file_handle_t file_hnd;
  clear_file_handle(file_hnd);
  bool retry = false;
  int retry_cnt = 0;
  do {
    try {
      file_hnd = conduit::relay::io::hdf5_open_file_for_read( file_path );
    }catch (conduit::Error const& e) {
      LBANN_WARNING(" :: trying to open the file " + file_path + " and got " + e.what());
      retry = true;
      retry_cnt++;
    }
  }while(retry && retry_cnt < LBANN_MAX_OPEN_FILE_RETRY);

  if (!is_file_handle_valid(file_hnd)) {
    std::cout << "Opening the file didn't work" << std::endl;
    return file_hnd;
  }

  conduit::relay::io::hdf5_group_list_child_names(file_hnd, "/", sample_names);

  if(sample_names.size() != (included_samples + excluded_samples)) {
    LBANN_ERROR(std::string("File does not contain the correct number of samples: found ")
                + std::to_string(sample_names.size())
                + std::string(" -- this does not equal the expected number of samples that are marked for inclusion: ")
                + std::to_string(included_samples)
                + std::string(" and exclusion: ")
                + std::to_string(excluded_samples));
  }

  return file_hnd;
}

template <typename file_handle_t, typename sample_name_t>
inline void sample_list_hdf5<file_handle_t, sample_name_t>
::all_gather_packed_lists(lbann_comm& comm) {
  int num_ranks = comm.get_procs_per_trainer();
  typename std::vector<samples_t> per_rank_samples(num_ranks);
  typename std::vector<file_id_stats_v_t> per_rank_file_id_stats_map(num_ranks);
  std::vector<std::unordered_map<std::string, size_t>> per_rank_file_map(num_ranks);

  // Close the existing open files
  for(auto&& e : my_file_id_stats_map()) {
    if(std::get<1>(e) > 0) {
      conduit::relay::io::hdf5_close_file(std::get<1>(e));
      std::get<1>(e) = 0;
    }
    std::get<2>(e).clear();
  }
  my_open_fd_pq().clear();

  size_t num_samples = sample_list<file_handle_t, sample_name_t>::all_gather_field(my_sample_list(), per_rank_samples, comm);
  size_t num_ids = sample_list<file_handle_t, sample_name_t>::all_gather_field(my_file_id_stats_map(), per_rank_file_id_stats_map, comm);
  size_t num_files = sample_list<file_handle_t, sample_name_t>::all_gather_field(my_file_map(), per_rank_file_map, comm);

  my_sample_list().clear();
  my_file_id_stats_map().clear();

  my_sample_list().reserve(num_samples);
  my_file_id_stats_map().reserve(num_ids);
  my_file_map().reserve(num_files);

  for(int r = 0; r < num_ranks; r++) {
    const samples_t& s_list = per_rank_samples[r];
    const file_id_stats_v_t& file_id_stats_map = per_rank_file_id_stats_map[r];
    const std::unordered_map<std::string, size_t>& file_map = per_rank_file_map[r];
    for (const auto& s : s_list) {
      sample_file_id_t index = s.first;
      const std::string& filename = std::get<0>(file_id_stats_map[index]);
      if(index >= my_file_id_stats_map().size()
         || (std::get<0>(my_file_id_stats_map().back()) != filename)) {
        index = my_file_id_stats_map().size();
        my_file_id_stats_map().emplace_back(std::make_tuple(filename, 0, std::deque<std::pair<int,int>>{}));
        // Update the file map structure
        if(my_file_map().count(filename) == 0) {
          my_file_map()[filename] = file_map.at(filename);
        }
      }else {
        for(size_t i = 0; i < my_file_id_stats_map().size(); i++) {
          if(filename == std::get<0>(my_file_id_stats_map()[i])) {
            index = i;
            break;
          }
        }
      }
      my_sample_list().emplace_back(std::make_pair(index, s.second));
    }
  }

  return;
}

template <typename file_handle_t, typename sample_name_t>
inline void sample_list_hdf5<file_handle_t, sample_name_t>
::compute_epochs_file_usage(const std::vector<int>& shuffled_indices,
                            int mini_batch_size,
                            const lbann_comm& comm) {
  for (auto&& e : my_file_id_stats_map()) {
    if(std::get<1>(e) > 0) {
      conduit::relay::io::hdf5_close_file(std::get<1>(e));
      std::get<1>(e) = 0;
    }
    std::get<2>(e).clear();
  }

  for (size_t i = 0; i < shuffled_indices.size(); i++) {
    int idx = shuffled_indices[i];
    const auto& s = my_sample_list()[idx];
    sample_file_id_t index = s.first;

    if((i % mini_batch_size) % comm.get_procs_per_trainer() == static_cast<size_t>(comm.get_rank_in_trainer())) {
      /// Enqueue the iteration step when the sample will get used
      int step = i / mini_batch_size;
      int substep = (i % mini_batch_size) / comm.get_procs_per_trainer();
      std::get<2>(my_file_id_stats_map()[index]).emplace_back(std::make_pair(step, substep));
    }
  }
}

template <typename file_handle_t, typename sample_name_t>
inline bool sample_list_hdf5<file_handle_t, sample_name_t>
::is_file_handle_valid(const file_handle_t& h) const {
  return (h > static_cast<file_handle_t>(0));
}

template <typename file_handle_t, typename sample_name_t>
inline void sample_list_hdf5<file_handle_t, sample_name_t>
::clear_file_handle(file_handle_t& h) {
  h = file_handle_t(0);
}

template <typename file_handle_t, typename sample_name_t>
inline void sample_list_hdf5<file_handle_t, sample_name_t>
::close_and_clear_file_handle(file_handle_t& h) {
  if(is_file_handle_valid(h)) {
    conduit::relay::io::hdf5_close_file(h);
  }
  clear_file_handle(h);
}

template <typename file_handle_t, typename sample_name_t>
inline void sample_list_hdf5<file_handle_t, sample_name_t>
::manage_open_file_handles(sample_file_id_t id, bool pre_open_fd) {
  /// When we enter this function the priority queue is either empty or a heap
  if(!my_open_fd_pq().empty()) {
    if(my_open_fd_pq().size() > my_max_open_files()) {
      auto& f = my_open_fd_pq().front();
      auto& victim = my_file_id_stats_map()[f.first];
      file_handle_t victim_fd = std::get<1>(victim);
      std::pop_heap(my_open_fd_pq().begin(), my_open_fd_pq().end(),
                    sample_list<file_handle_t, sample_name_t>::pq_cmp);
      my_open_fd_pq().pop_back();
      if(victim_fd > 0) {
        conduit::relay::io::hdf5_close_file(victim_fd);
        std::get<1>(victim) = 0;
      }
    }
  }

  /// Before we can enqueue the any new access times for this descriptor, remove any
  /// earlier descriptor
  std::sort_heap(my_open_fd_pq().begin(), my_open_fd_pq().end(),
                 sample_list<file_handle_t, sample_name_t>::pq_cmp);
  if(my_open_fd_pq().front().first == id) {
    my_open_fd_pq().pop_front();
  }
  std::make_heap(my_open_fd_pq().begin(), my_open_fd_pq().end(),
                 sample_list<file_handle_t, sample_name_t>::pq_cmp);

  auto& e = my_file_id_stats_map()[id];
  auto& file_access_queue = std::get<2>(e);
  if(!file_access_queue.empty()) {
    if(!pre_open_fd) {
      file_access_queue.pop_front();
    }
  }
  if(!file_access_queue.empty()) {
    my_open_fd_pq().emplace_back(std::make_pair(id,file_access_queue.front()));
  }else {
    /// If there are no future access of the file place a terminator entry to track
    /// the open file, but is always sorted to the top of the heap
    my_open_fd_pq().emplace_back(std::make_pair(id,std::make_pair(INT_MAX,id)));
  }
  std::push_heap(my_open_fd_pq().begin(), my_open_fd_pq().end(),
                 sample_list<file_handle_t, sample_name_t>::pq_cmp);
  return;
}

template <typename file_handle_t, typename sample_name_t>
inline file_handle_t sample_list_hdf5<file_handle_t, sample_name_t>
::open_samples_file_handle(const size_t i, bool pre_open_fd) {
  const sample_t& s = my_sample_list()[i];
  sample_file_id_t id = s.first;
  file_handle_t h = sample_list<file_handle_t, sample_name_t>::get_samples_file_handle(id);
  if (!is_file_handle_valid(h)) {
    const std::string& file_name = sample_list<file_handle_t, sample_name_t>::get_samples_filename(id);
    const std::string& file_dir = sample_list<file_handle_t, sample_name_t>::get_samples_dirname();
    const std::string file_path = add_delimiter(file_dir) + file_name;
    if (file_name.empty() || !check_if_file_exists(file_path)) {
      LBANN_ERROR(std::string{} + " :: data file '" + file_path + "' does not exist.");
    }
    bool retry = false;
    int retry_cnt = 0;
    do {
      try {
        h = conduit::relay::io::hdf5_open_file_for_read( file_path );
      }catch (conduit::Error const& e) {
        LBANN_WARNING(" :: trying to open the file " + file_path + " and got " + e.what());
        retry = true;
        retry_cnt++;
      }
    }while(retry && retry_cnt < 3);

    if (!is_file_handle_valid(h)) {
      LBANN_ERROR(std::string{} + " :: data file '" + file_path + "' could not be opened.");
    }
    auto& e = my_file_id_stats_map()[id];
    std::get<1>(e) = h;
  }
  manage_open_file_handles(id, pre_open_fd);
  return h;
}

template <typename file_handle_t, typename sample_name_t>
inline void sample_list_hdf5<file_handle_t, sample_name_t>
::close_if_done_samples_file_handle(const size_t i) {
  const sample_t& s = my_sample_list()[i];
  sample_file_id_t id = s.first;
  file_handle_t h = sample_list<file_handle_t, sample_name_t>::get_samples_file_handle(id);
  if (!is_file_handle_valid(h)) {
    auto& e = my_file_id_stats_map()[id];
    auto& file_access_queue = std::get<2>(e);
    if(file_access_queue.empty()) {
      conduit::relay::io::hdf5_close_file(std::get<1>(e));
      std::get<1>(e) = 0;
    }
  }
}

} // end of namespace lbann

#endif // __SAMPLE_LIST_HDF5_HPP__
