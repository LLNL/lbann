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

#ifndef LBANN_DATA_READERS_SAMPLE_LIST_HPP
#define LBANN_DATA_READERS_SAMPLE_LIST_HPP

#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "lbann/utils/file_utils.hpp"

namespace lbann {

// Forward Declarations
class lbann_comm;

static const std::string multi_sample_exclusion = "MULTI-SAMPLE_EXCLUSION";
static const std::string multi_sample_inclusion = "MULTI-SAMPLE_INCLUSION";
static const std::string single_sample = "SINGLE-SAMPLE";
static const std::string multi_sample_inclusion_v2 =
  "MULTI-SAMPLE_INCLUSION_V2";
static const std::string conduit_hdf5_exclusion = "CONDUIT_HDF5_EXCLUSION";
static const std::string conduit_hdf5_inclusion = "CONDUIT_HDF5_INCLUSION";

struct sample_list_header
{
  /// Whether each data file includes multiple samples
  bool m_is_multi_sample;
  /// Whether to list the IDs of samples to exclude or to include
  bool m_is_exclusive;
  /// Whether to read the header line for a label file
  bool m_no_label_header;
  /// Whether the sample list has fields to represent unused samples
  bool m_has_unused_sample_fields;
  /// Number of included samples
  size_t m_included_sample_count;
  /// Number of excluded samples
  size_t m_excluded_sample_count;
  size_t m_num_files;
  /// Data file directory
  std::string m_file_dir;
  std::string m_sample_list_name;
  std::string m_label_filename;

  sample_list_header();

  void set_sample_list_type(const std::string& line1);
  void set_sample_count(const std::string& line2);
  void set_data_file_dir(const std::string& line3);
  void set_label_filename(const std::string& line4);

  bool is_multi_sample() const;
  bool is_exclusive() const;
  bool use_label_header() const;
  bool has_unused_sample_fields() const;
  size_t get_sample_count() const;
  size_t get_num_files() const;
  const std::string& get_file_dir() const;
  const std::string& get_sample_list_name() const;
  /// Save the filename or stream name of this sample list for debugging
  void set_sample_list_name(const std::string& n);
  const std::string& get_label_filename() const;
  template <class Archive>
  void serialize(Archive& ar);
};

template <typename sample_name_t>
class sample_list
{
public:
  using name_t = sample_name_t;
  /// The type for the index assigned to each sample file
  using sample_file_id_t = std::size_t;
  /** To describe a sample as the id of the file to which it belongs.
   * Each file contains only one sample. */
  using sample_t = std::template pair<sample_file_id_t, sample_name_t>;
  /// Type for the list of samples
  using samples_t = std::template vector<sample_t>;
  /// Type for the index into the sample list
  using sample_idx_t = typename samples_t::size_type;
  /// Type for the map from sample name to the sample list index
  using sample_map_t = std::unordered_map<sample_name_t, sample_idx_t>;
  /// Mapping of the file index to the filename
  using file_id_stats_v_t = std::vector<std::string>;

  sample_list();
  virtual ~sample_list();
  sample_list(const sample_list& rhs);
  sample_list& operator=(const sample_list& rhs);
  sample_list& copy(const sample_list& rhs);

  void copy_members(const sample_list& rhs);

  /// Load a sample list file using the given stride and offset on the sample
  /// sequence
  void load(std::istream& istrm, size_t stride = 1, size_t offset = 0);

  /** Load a sample list file using the stride as the number of processes per
   *  trainer and the offset as the current rank within the trainer if
   *  interleaving option is on.
   */
  void load(const std::string& samplelist_file,
            const lbann_comm& comm,
            bool interleave);
  void load(std::istream& istrm, const lbann_comm& comm, bool interleave);
  /// Load sample list using the given header instead of reading it from the
  /// input stream
  void load(const sample_list_header& header,
            std::istream& istrm,
            const lbann_comm& comm,
            bool interleave);

  /// Restore a sample list from a serialized string
  void load_from_string(const std::string& samplelist,
                        const lbann_comm& comm,
                        bool interleave);

  /// Tells how many samples in the list
  virtual size_t size() const;

  /// Tells how many sample files are there
  virtual size_t get_num_files() const;

  /// Tells if the internal list is empty
  bool empty() const;

  /// Serialize to and from an archive using the cereal library
  template <class Archive>
  void serialize(Archive& ar);

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
  const std::string& get_label_filename() const;

  void all_gather_archive(const std::string& archive,
                          std::vector<std::string>& gathered_archive,
                          lbann_comm& comm);
  void all_gather_archive_new(const std::string& archive,
                              std::vector<std::string>& gathered_archive,
                              lbann_comm& comm);

  template <typename T>
  size_t
  all_gather_field(T data, std::vector<T>& gathered_data, lbann_comm& comm);
  virtual void all_gather_packed_lists(lbann_comm& comm);

  /// Set to maintain the original sample order as listed in the file
  void keep_sample_order(bool keep);

  /// Manually set the sample list name, which can be used for stream-based
  /// sources
  void set_sample_list_name(const std::string& n);

  /// Set to check the existence of data file in the list
  void set_data_file_check();
  /// Set not to check the existence of data file in the list
  void unset_data_file_check();

  /// Build map from sample names to indices for sample list
  void build_sample_map_from_name_to_index();

  /// Clear the map from sample names to indices
  void clear_sample_map_from_name_to_index();

  /// Return the index of the sample with the specified name
  sample_idx_t get_sample_index(const sample_name_t& sn);

protected:
  /// Reads a header line from the sample list given as a stream, and use the
  /// info string for error message
  std::string read_header_line(std::istream& ifs,
                               const std::string& listname,
                               const std::string& info);

  /// Reads the header of a sample list
  void read_header(std::istream& istrm);

  /// read the body of a sample list, which is the list of sample files, where
  /// each file contains a single sample.
  virtual void
  read_sample_list(std::istream& istrm, size_t stride = 1, size_t offset = 0);

  /// Assign names to samples when there is only one sample per file without a
  /// name.
  virtual void assign_samples_name();

  /// Reads a sample list and populates the internal list
  size_t get_samples_per_file(std::istream& istrm,
                              size_t stride = 1,
                              size_t offset = 0);

  /// Add the header info to the given string
  void write_header(std::string& sstr, size_t num_files) const;

  /// Get the number of total/included/excluded samples
  virtual void
  get_num_samples(size_t& total, size_t& included, size_t& excluded) const;

  virtual void set_samples_filename(sample_file_id_t id,
                                    const std::string& filename);

  /// Reorder the sample list to its initial order
  virtual void reorder();

protected:
  /// header info of sample list
  sample_list_header m_header;

  /// The stride used in loading sample list file
  size_t m_stride;

  /// maintain the original sample order as listed in the file
  bool m_keep_order;

  /// Whether to check the existence of data file
  bool m_check_data_file;

  /// List of all samples with a file identifier and sample name for each sample
  samples_t m_sample_list;

  /// Map from sample name to the corresponding index into the sample list
  sample_map_t m_map_name_to_idx;

private:
  /// Maps sample's file id to file names, file descriptors, and use counts
  file_id_stats_v_t m_file_id_stats_map;
};

void handle_mpi_error(int ierr);

template <typename T>
inline T uninitialized_sample_name();

} // namespace lbann

#endif // LBANN_DATA_READERS_SAMPLE_LIST_HPP
