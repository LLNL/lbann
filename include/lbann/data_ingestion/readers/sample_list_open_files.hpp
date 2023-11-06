////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_DATA_READERS_SAMPLE_LIST_OPEN_FILES_HPP
#define LBANN_DATA_READERS_SAMPLE_LIST_OPEN_FILES_HPP

#include "sample_list.hpp"

#include <deque>

/// Number of system and other files that may be open during execution
#define LBANN_MAX_OPEN_FILE_MARGIN 128
#define LBANN_MAX_OPEN_FILE_RETRY 3

namespace lbann {

template <typename sample_name_t, typename file_handle_t>
class sample_list_open_files : public sample_list<sample_name_t>
{
public:
  /// The type for the index assigned to each sample file
  using sample_file_id_t = std::size_t;
  /** To describe a sample as a pair of the file to which it belongs and its
     name Each file may contain multiple samples. */
  using sample_t = typename sample_list<sample_name_t>::sample_t;
  /// Information for each file used by the sample list: includes the file name,
  /// file descriptor, and and a queue of each step and substep when data will
  /// be loaded from the file
  using file_id_stats_t =
    std::tuple<std::string, file_handle_t, std::deque<std::pair<int, int>>>;
  /// Accessor macros for the file_id_stats_t tuple
  enum fid_stats
  {
    FID_STATS_NAME = 0,
    FID_STATS_HANDLE = 1,
    FID_STATS_DEQUE = 2
  };

  /// Type for the list of samples
  using samples_t = typename sample_list<sample_name_t>::samples_t;
  /// Mapping of the file index to the statistics for each file
  using file_id_stats_v_t =
    std::vector<file_id_stats_t>; // rename to sample_to_file_v or something
  /// Type for the map of file descriptors to usage step and substep
  using fd_use_map_t =
    std::template pair<sample_file_id_t, std::pair<int, int>>;

  sample_list_open_files();
  virtual ~sample_list_open_files();
  /** Copy constructor repllicates all the member variables as they are except
   * the file information vector, for which only the file name is copied. */
  sample_list_open_files(const sample_list_open_files& rhs);
  /** assignemnt operation repllicates all the member variables as they are
   * except the file information vector, for which only the file name is copied.
   */
  sample_list_open_files& operator=(const sample_list_open_files& rhs);
  sample_list_open_files& copy(const sample_list_open_files& rhs);

  void copy_members(const sample_list_open_files& rhs);

  /// Tells how many samples in the list
  size_t size() const override;

  /// Tells how many sample files are there
  size_t get_num_files() const override;

  using sample_list<sample_name_t>::load;
  /// Emit a serialized archive using the cereal library
  template <class Archive>
  void save(Archive& ar) const;
  /// Restore the member variables from a given archrive serialized by the
  /// cereal library
  template <class Archive>
  void load(Archive& ar);

  /// Serialize this sample list into an std::string object
  bool to_string(std::string& sstr) const override;

  const std::string& get_samples_filename(sample_file_id_t id) const override;

  file_handle_t get_samples_file_handle(sample_file_id_t id) const;

  void set_files_handle(const std::string& filename, file_handle_t h);

  void delete_file_handle_pq_entry(sample_file_id_t id);

  void manage_open_file_handles(sample_file_id_t id);

  file_handle_t open_samples_file_handle(const size_t i);

  virtual void close_samples_file_handle(const size_t i,
                                         bool check_if_in_use = false);

  void compute_epochs_file_usage(const std::vector<uint64_t>& shufled_indices,
                                 uint64_t mini_batch_size,
                                 const lbann_comm& comm);

  virtual bool is_file_handle_valid(const file_handle_t& h) const = 0;

  void all_gather_packed_lists(lbann_comm& comm) override;

protected:
  void set_samples_filename(sample_file_id_t id,
                            const std::string& filename) override;

  void reorder() override;

  /// Get the list of samples from a specific type of bundle file
  virtual void
  obtain_sample_names(file_handle_t& h,
                      std::vector<std::string>& sample_names) const = 0;

  file_handle_t open_file_handle(std::string file_path);

  /// Get the list of samples that exist in a bundle file
  file_handle_t get_bundled_sample_names(std::string file_path,
                                         std::vector<std::string>& sample_names,
                                         size_t included_samples,
                                         size_t excluded_samples);

  /// Check that the list of samples given actually exist in a bundle file
  void
  validate_implicit_bundles_sample_names(std::string file_path,
                                         std::string filename,
                                         std::vector<std::string>& sample_names,
                                         size_t included_samples,
                                         size_t excluded_samples);

  size_t read_line_integral_type(std::istringstream& sstr,
                                 sample_file_id_t index);

  size_t read_line(std::istringstream& sstr, sample_file_id_t index);

  /// read the body of exclusive sample list
  void read_exclusive_list(std::istream& istrm,
                           size_t stride = 1,
                           size_t offset = 0);

  /// read the body of inclusive sample list
  void read_inclusive_list(std::istream& istrm,
                           size_t stride = 1,
                           size_t offset = 0);

  /// read the body of a sample list
  void read_sample_list(std::istream& istrm,
                        size_t stride = 1,
                        size_t offset = 0) override;

  void assign_samples_name() override {}

  /// Get the number of total/included/excluded samples
  void get_num_samples(size_t& total,
                       size_t& included,
                       size_t& excluded) const override;

  static bool pq_cmp(fd_use_map_t left, fd_use_map_t right)
  {
    return ((left.second).first < (right.second).first) ||
           (((left.second).first == (right.second).first) &&
            ((left.second).second < (right.second).second));
  }

  virtual file_handle_t
  open_file_handle_for_read(const std::string& file_path) = 0;
  virtual void close_file_handle(file_handle_t& h) = 0;
  virtual void clear_file_handle(file_handle_t& h) = 0;

private:
  using sample_list<sample_name_t>::serialize;
  template <class Archive>
  void serialize(Archive& ar) = delete;

protected:
  using sample_list<sample_name_t>::m_header;

  /// Maps sample's file id to file names, file descriptors, and use counts
  file_id_stats_v_t m_file_id_stats_map;

private:
  /// Track the number of samples per file
  std::unordered_map<std::string, size_t> m_file_map;

  /// Track the number of open file descriptors and when they will be used next
  std::deque<fd_use_map_t> m_open_fd_pq;

  size_t m_max_open_files;
};

template <typename T>
inline T uninitialized_file_handle();

} // namespace lbann

#endif // LBANN_DATA_READERS_SAMPLE_LIST_OPEN_FILES_HPP
