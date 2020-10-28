////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef __DATA_STORE_CONDUIT_HPP__
#define __DATA_STORE_CONDUIT_HPP__

#include "lbann_config.hpp"

#include "lbann/base.hpp"
#include "lbann/comm.hpp"
#include "lbann/utils/exception.hpp"
#include "conduit/conduit_node.hpp"
#include <unordered_map>
#include <unordered_set>
#include <mutex>


namespace lbann {

// support for encoding data_id in conduit::Node, used by
// conduit_data_store and associated code
#define LBANN_SAMPLE_ID_PAD 9
#define LBANN_DATA_ID_STR(data_id) pad(std::to_string(data_id), LBANN_SAMPLE_ID_PAD, '0')

class generic_data_reader;

/** Create a hash function for hashing a std::pair type */
struct size_t_pair_hash
{
  template <class T1, class T2>
  std::size_t operator() (const std::pair<T1, T2> &pair) const
  {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

class data_store_conduit {

 public:

  // need to quickly change from unordered_map to map for debugging
  using map_ii_t = std::unordered_map<int,int>;
  using map_is_t = std::unordered_map<int,size_t>;

  // Hash map for tracking the node and hyperslab partition ID
  using map_pssi_t = std::unordered_map<std::pair<size_t,size_t>,int,size_t_pair_hash>;

  // not currently used; will be in the future
  using map_ss_t = std::unordered_map<size_t,size_t>;

  //! ctor
  data_store_conduit(generic_data_reader *reader);

  //! copy ctor
  data_store_conduit(const data_store_conduit&);

  //! copy / split ctor
  data_store_conduit(const data_store_conduit&, const std::vector<int>&);

  //! operator=
  data_store_conduit& operator=(const data_store_conduit&);

  data_store_conduit * copy() const { return new data_store_conduit(*this); }

  //! dtor
  ~data_store_conduit();

  void set_data_reader_ptr(generic_data_reader *reader);

  //! convenience handle
  void set_shuffled_indices(const std::vector<int> *indices);

  /** @brief Returns the number of samples summed over all ranks */
  size_t get_num_global_indices() const;

  void setup(int mini_batch_size);

  // TODO FIXME
  void check_mem_capacity(lbann_comm *comm, const std::string sample_list_file, size_t stride, size_t offset);

  /** @brief Returns the conduit Node associated with the data_id */
  const conduit::Node & get_conduit_node(int data_id) const;

  /** @brief Set a conduit node in the data store
   *
   * if 'already_have = true' then the passed 'node' was obtained by a call to
   * get_empty_node(); note, we do this to prevent copying the node
   */
  void set_conduit_node(int data_id, const conduit::Node &node, bool already_have = false);

  void set_preloaded_conduit_node(int data_id, const conduit::Node &node);

  void spill_preloaded_conduit_node(int data_id, const conduit::Node &node);

  const conduit::Node & get_random_node() const;

  const conduit::Node & get_random_node(const std::string &field) const;

  /// returns an empty node
  conduit::Node & get_empty_node(int data_id);

  //=================================================================
  // methods for setting and querying the data store's mode
  //=================================================================
  /** @brief Returns true if preloading is turned on
   *
   * See notes in: is_explicitly_loading()
   */
  bool is_preloading() const { return m_preloading; }

  /** @brief Returns true if explicitly loading is turned on
   *
   * 'explicitly loading' means that the data that will be owned
   * by each rank is passed into the data store during the first epoch.
   * This is in contrast to preloading, in which the data is passed into
   * the data store prior to the first epoch. Explicit and preloading
   * are exclusive: at most only one may be true, however, both will
   * be set to false when all loading is complete.
   */
  bool is_explicitly_loading() const { return m_explicitly_loading; }

  /** @brief Returns true if all loading has been completed
   *
   * See notes in: set_loading_is_complete()
   */
  bool is_fully_loaded() const;

  /** @brief Returns "true" is running in local cache mode
   *
   * In local cache mode, each node contains a complete copy
   * of the data set. This is stored in a shared memory segment,
   * but part of the set may be spilled to disk if memory is
   * insufficient. Local cache mode is activated via the cmd line
   * flag: --data_store_cache
   */
  bool is_local_cache() const { return m_is_local_cache; }

  /** @brief Turn preloading on or off */
  void set_is_preloading(bool flag);

  /** @brief Turn on explicit loading */
  void set_is_explicitly_loading(bool flag);

  /** @brief Marks the data_store as fully loaded
   *
   * Fully loaded means that each rank has all the data that it
   * is intended to own. When not running in local cache mode, this
   * occurs (1) at the conclusion of preloading, prior to the beginning of
   * the first epoch, or (2) at the conclusion of the first epoch, if
   * explicitly loading. When running in local cache mode, this occurs
   * (1) at the conclusion of preload_local_cache(), which is called prior
   * to the first epoch, or (2) at the conclusion of exchange_local_caches(),
   * at th conclusion of the first epoch, if explicitly loading.
   */
  void set_loading_is_complete();

  /** @brief turns local cache mode on of off */
  void set_is_local_cache(bool flag = true) { m_is_local_cache = flag; }

  /** @brief Check that explicit loading, preloading, and fully loaded flags are consistent */
  void check_query_flags() const;

  //=================================================================
  // END methods for setting and querying the data store's mode
  //=================================================================

//XX   void { m_owner_maps_were_exchanged = false; }
  /// fills in m_owner, which maps index -> owning processor
  void exchange_owner_maps();

  /// fills in m_owner, which maps index -> owning processor
  void build_preloaded_owner_map(const std::vector<int>& per_rank_list_sizes);

  /// fills in m_owner, which maps index -> owning processor
  void set_preloaded_owner_map(const std::unordered_map<int,int> &owner) {
    for(auto&& i : owner) {
      m_owner[std::make_pair(i.first, m_offset_in_partition)] = i.second;
    }
  }

  /** @brief Special hanling for ras_lipid_conduit_data_reader; may go away in the future */
  void clear_owner_map();

  void set_owner_map(const std::unordered_map<int, int> &m) {
    for(auto&& i : m) {
      m_owner[std::make_pair(i.first, m_offset_in_partition)] = i.second;
    }
  }

  /** @brief Special handling for ras_lipid_conduit_data_reader; may go away in the future */
  void add_owner(int data_id, int owner) { m_owner[std::make_pair(data_id, m_offset_in_partition)] = owner; }

  /** @brief Special handling for ras_lipid_conduit_data_reader; may go away in the future */
  void set_finished_building_map() { m_owner_maps_were_exchanged = true; }

  /// Recompact the nodes because they are not copied properly when instantiating
  /// using the copy constructor
  void compact_nodes();

  /// returns the processor that owns the data associated
  /// with the index
  int get_index_owner(int idx);


  /** @brief Read the data set into memory
   *
   * Each rank reads a portion of the data set, then
   * bcasts to all other ranks.
   */
  void preload_local_cache();

  void exchange_mini_batch_data(size_t current_pos, size_t mb_size);

  void set_node_sizes_vary() { m_node_sizes_vary = true; }

  bool has_conduit_node(int data_id) const;

  /// only used for debugging; pass --debug on cmd line to get
  /// each data store to print to a different file. This is made
  /// public so data readers can also print to the file
  std::ofstream *m_debug = nullptr;
  std::ofstream *m_profile = nullptr;

  /// for use during development and debugging
  int get_data_size() { return m_data.size(); }

  /// made public for debugging during development
  void copy_members(const data_store_conduit& rhs);

  /** @brief Closes then reopens the debug logging file
   *
   * Debug logging is enabled on all ranks via the cmd line flag: --data_store_debug
   */
  void flush_debug_file();

  /** @brief Closes then reopens the profile logging file
   *
   * Profile logging is enabled on P_0 via the cmd line flag: --data_store_profile
   */
  void flush_profile_file() const;

  /** @brief Writes object's state to file */
  void write_checkpoint(std::string dir_name);

  /** @brief Loads object's state from file */
  void load_checkpoint(std::string dir_name, generic_data_reader *reader = nullptr);

  /** @brief Add text to the profiling file, if it's opened */
  void set_profile_msg(std::string);

  /** @brief Runs an internal test to ensure the locally cached conduit data is correct
   *
   * For use during development and testing. This test is activated via
   * the cmd line flag: --data_store_test_cache. Output may be written to
   * cout, and the profile and debug files (if they are opened)
   * @param n is the maximum number of samples to test; set to -1 to test all
   * @return true, if all samples read from file match those constructed from
   *               the local shared memory segment (aka, cache)
   */
  bool test_local_cache_imagenet(int n);

  void test_imagenet_node(int sample_id, bool dereference = true);

  size_t get_mem_usage();

private :

  bool m_bcast_sample_size = true;

  // if not null, 'm_other' points from a train to a validation
  // data store; this permits communication which is needed in
  // special cases (e.g, see: data_reader_npz_ras_lipid.cpp)
  data_store_conduit *m_other = nullptr;

  bool m_owner_maps_were_exchanged = false;

  bool m_run_checkpoint_test = false;

  /** @brief The number of samples that this processor owns */
  size_t m_my_num_indices = 0;

  /** @brief if true, then we are spilling (offloading) samples to disk */
  bool m_spill = false;

  /** @brief if true, then all samples have been spilled */
  bool m_is_spilled = false;

  /** During spilling, the conduit file pathnames are written to this file */
  std::ofstream m_metadata;

  /** @brief Base directory for spilling (offloading) conduit nodes */
  std::string m_spill_dir_base;

  /** @brief Used to form the directory path for spilling conduit nodes */
  int m_cur_spill_dir_integer = -1;

  /** @brief @brief Current directory for spilling (writing to file) conduit nodes
   *
   * m_cur_spill_dir = m_spill_dir_base/<m_cur_spill_dir_integer>
   */
  std::string m_cur_spill_dir;

  /** @brief The directory to use for testing checkpointing
   *
   * Testing is activated by passing the cmd flag: --data_store_test_checkpoint=<dir>
   */
  std::string m_test_dir;

  /** @brief Contains the number of conduit nodes that have been written to m_cur_dir
   *
   * When m_num_files_in_cur_spill_dir == m_max_files_per_directory,
   * m_cur_spill_dir_integer is incremented and a new m_cur_dir is created
   */
  int m_num_files_in_cur_spill_dir;

  /** @brief maps data_id to m_m_cur_spill_dir_integer. */
  map_ii_t m_spilled_nodes;

  /// used in set_conduit_node(...)
  std::mutex m_mutex;
  std::mutex m_mutex_2;

  /// for use in local cache mode
  char *m_mem_seg = 0;
  size_t m_mem_seg_length = 0;
  std::string m_seg_name;

  const std::string m_debug_filename_base = "debug";
  std::string m_debug_filename;

  const std::string m_profile_filename_base = "data_store_profile";
  std::string m_profile_filename;

  bool m_was_loaded_from_file = false;
  const std::string m_cereal_fn = "data_store_cereal";

  /// used in spill_to_file
  /// (actually, conduit::Node.save() writes both a
  ///  json file and a binary file, so double this number
  const int m_max_files_per_directory = 500;

  //===========================================================
  // timers for profiling exchange_data
  //===========================================================

  // applicable to imagenet; NA for JAG
  double m_exchange_sample_sizes_time = 0;

  // time from beginning of exchange_data_by_sample to wait_all
  double m_start_snd_rcv_time = 0;

  // time for wait_all
  double m_wait_all_time = 0;

  // time to unpack nodes received from other ranks
  double m_rebuild_time = 0;

  // total time for exchange_mini_batch_data
  double m_exchange_time = 0;

  // sanity check:
  //   m_start_snd_rcv_time + m_wait_all_time + m_rebuild_time
  // should be only slightly less than m_exchange_time;
  // Note that, for imagenet, the first call to exchange_data_by_sample
  // involves additional communication for exchanging sample sizes

  //===========================================================
  // END: timers for profiling exchange_data
  //===========================================================

  bool m_is_setup = false;

  /// set to true if data_store is preloaded
  bool m_loading_is_complete = false;

  /** @brief True, if we are in preload mode */
  bool m_preloading = false;

  /** @brief True, if we are in explicit loading mode
   *
   * There is some redundancy here: m_preloading and m_explicitly_loading
   * can not both be true, but both may be false. When m_loading_is_complete
   * is true, both m_preloading and m_preloading should be false.
   */
  bool m_explicitly_loading = false;

  /// The size of the mini-batch that was used to calculate ownership
  /// of samples when building the owner map.  This size has to be
  /// used consistently when computing the indices that will be sent
  /// and received.
  int m_owner_map_mb_size = 0;

  /// size of a compacted conduit::Node that contains a single sample
  int m_compacted_sample_size = 0;

  bool m_is_local_cache = false;

  bool m_node_sizes_vary = false;

  /// used in exchange_data_by_sample, when sample sizes are non-uniform
  bool m_have_sample_sizes = false;

  generic_data_reader *m_reader;

  lbann_comm *m_comm = nullptr;

  /// convenience handles
  bool m_world_master;
  bool m_trainer_master;
  int  m_rank_in_trainer;
  int  m_rank_in_world = -1; // -1 for debugging
  int  m_partition_in_trainer;
  int  m_offset_in_partition;

  /// number of procs in the trainer; convenience handle
  int  m_np_in_trainer;
  int  m_num_partitions_in_trainer;

  /** @brief Maps an index to the processor that owns the associated data
   * First value of index is the sample ID and second value is the partiton ID
   *
   * Must be mutable since rhs.m_owner may be modified in copy_members,
   * in which rhs is const.
   */
  mutable map_pssi_t m_owner;

  /// convenience handle
  const std::vector<int> *m_shuffled_indices;

  /** @brief Contains the conduit nodes that are "owned" by this rank
   *
   * Map data_id -> conduit::Node.
   * Must be mutable since rhs.m_owner may be modified in copy_members,
   * in which rhs is const.
   */
  mutable std::unordered_map<int, conduit::Node> m_data;

  /** @brief Contains a cache of the conduit nodes that are
   * "owned" by this rank
   *
   * This differs from m_data in that this holds temporarily,
   * during the first epoch, if we're running in local cache mode
   * and explicitly loading
   */
  std::unordered_map<int, conduit::Node> m_data_cache;

  /// Contains the list of data IDs that will be received
  std::vector<int> m_recv_data_ids;
  map_ii_t m_recv_sample_sizes;

  /// This vector contains Nodes that this processor needs for
  /// the current minibatch; this is filled in by exchange_data()
  std::unordered_map<int, conduit::Node> m_minibatch_data;

  /// work space; used in exchange_data
  std::vector<conduit::Node> m_send_buffer;
  std::vector<conduit::Node> m_send_buffer_2;
  std::vector<El::mpi::Request<El::byte>> m_send_requests;
  std::vector<El::mpi::Request<El::byte>> m_recv_requests;
  std::vector<conduit::Node> m_recv_buffer;
  std::vector<size_t> m_outgoing_msg_sizes;
  std::vector<size_t> m_incoming_msg_sizes;

  /** @brief Maps a data_id to its image size
   *
   * Used when conduit Nodes have non-uniform size, e.g, imagenet;
   * see: set_node_sizes_vary()
   */
  map_is_t m_sample_sizes;

  /** @brief Maps a data_id to the image location in a shared memory segment */
  map_is_t m_image_offsets;

  /// maps processor id -> set of indices (whose associated samples)
  /// this proc needs to send. (formerly called "proc_to_indices);
  /// this is filled in by build_indices_i_will_send()
  std::vector<std::unordered_set<int>> m_indices_to_send;

  /// maps processor id -> set of indices (whose associated samples)
  /// this proc needs to recv from others. (formerly called "needed")
  std::vector<std::unordered_set<int>> m_indices_to_recv;

  //=========================================================================
  // methods follow
  //=========================================================================

  void exchange_data_by_sample(size_t current_pos, size_t mb_size);

  void setup_data_store_buffers();

  /// called by exchange_data
  void build_node_for_sending(const conduit::Node &node_in, conduit::Node &node_out);

  /// for use when conduit Nodes have non-uniform size, e.g, imagenet
  void exchange_sample_sizes();

  /// fills in m_indices_to_send and returns the number of samples
  /// that will be sent
  int build_indices_i_will_send(int current_pos, int mb_size);

  /// fills in m_indices_to_recv and returns the number of samples
  /// that will be received
  int build_indices_i_will_recv(int current_pos, int mb_size);

  void error_check_compacted_node(const conduit::Node &nd, int data_id);

  /** @brief All ranks exchange their cached data */
  void exchange_local_caches();

  /// Currently only used for imagenet. On return, 'sizes' maps a sample_id to image size, and indices[p] contains the sample_ids that P_p owns
  /// for use in local cache mode
  void get_image_sizes(map_is_t &sizes, std::vector<std::vector<int>> &indices);

  /// for use in local cache mode
  void allocate_shared_segment(map_is_t &sizes, std::vector<std::vector<int>> &indices);

  /// for use in local cache mode
  void read_files(std::vector<char> &work, map_is_t &sizes, std::vector<int> &indices);

  /// fills in m_image_offsets for use in local cache mode
  void compute_image_offsets(map_is_t &image_sizes, std::vector<std::vector<int>> &indices);

  /// for use in local cache mode
  void exchange_images(std::vector<char> &work, map_is_t &image_sizes, std::vector<std::vector<int>> &indices);

  void build_conduit_nodes(map_is_t &sizes);

  /// for use in local cache mode
  void fillin_shared_images(char* images, size_t size, size_t offset);

  /** @brief For testing during development
   *
   * At the beginning of the 2nd epoch, calls write_checkpoint(),
   * clears some variables, calls load_checkpoint then continues.
   * To activate this test use cmd flag: --data_store_test_checkpoint=
   */
  void test_checkpoint(const std::string&);

  /** @brief Called by test_checkpoint */
  void print_variables();

  /** @brief Called by test_checkpoint
   *
   * For testing and development. Prints the first 'n' entries from
   * the owner map * (which maps sample_id -> owning rank) to std::cout
   */
  void print_partial_owner_map(int n);

  std::string get_conduit_dir() const;
  std::string get_cereal_fn() const;
  std::string get_metadata_fn() const;

  /** @brief Creates the directory if it does not already exist */
  void make_dir_if_it_doesnt_exist(const std::string &dir);

  /** @brief Writes conduit node to file */
  void spill_conduit_node(const conduit::Node &node, int data_id);

  /** @brief Loads conduit nodes from file into m_data */
  void load_spilled_conduit_nodes();

  /** @brief Creates directory structure, opens metadata file for output, etc
   *
   * This method is called for both --data_store_spill and
   * --data_store_test_checkpoint
   */
  void setup_spill(std::string dir);

  /** @brief Saves this object's state to file
   *
   * Here, "state" is all data, except for conduit nodes, that is
   * needed to reload from checkpoint
   */
  void save_state();

  /** @brief Optionally open debug and profiling files
   *
   * A debug file is opened for every <rank, data reader role> pair;
   * files are opened if the cmd flag --data_store_debug is passed.
   * A profiling file is opened only be <world_master, data reader role>
   * pairs; files are opened if the cmd flag --data_store_profile is passed.
   */
  void open_informational_files();

  /** @brief Creates a directory for spilling conduit nodes */
  void open_next_conduit_spill_directory();

  /** @brief Write timing data for data exchange to the profile file, if it's opened */
  void profile_timing();

  void setup_checkpoint_test();

  std::string get_lassen_spill_dir();

  void verify_sample_size();

  //=========================================================================
  // functions and templates for optional profiling and debug files follow
  //=========================================================================

  void PROFILE() const {
    if (!m_profile) {
      return;
    }
    (*m_profile) << std::endl;
    flush_profile_file();
  }

  template <typename T, typename... Types>
  void PROFILE(T var1, Types... var2) const {
    if (!m_world_master) {
      return;
    }
    if (!m_profile) {
      return;
    }
    (*m_profile) << var1 << " ";
    PROFILE(var2...) ;
    flush_profile_file();
  }

  void DEBUG_DS() {
    if (!m_debug) {
      return;
    }
    (*m_debug) << std::endl;
    flush_debug_file();
  }

  template <typename T, typename... Types>
  void DEBUG_DS(T var1, Types... var2) {
    if (!m_debug) {
      return;
    }
    (*m_debug) << var1 << " ";
    DEBUG_DS(var2...) ;
    flush_debug_file();
  }
};

}  // namespace lbann


#endif  // __DATA_STORE_JAG_HPP__
