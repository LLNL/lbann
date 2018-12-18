////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#ifndef __SAMPLE_LIST_HPP__
#define __SAMPLE_LIST_HPP__

#include <iostream>
#include <string>
#include <vector>
#include <list>
#include "lbann/utils/exception.hpp"
#include "lbann/comm.hpp"

namespace lbann {

template <typename SN = std::string>
class sample_list {
 // TODO: currently using std::list as we do not know the number of lines the sample list file
 // Either copy the list to a final vector, or have a header line that telss the number of lines
 // in the list file. Using a vector will allow shuffling.
 // In addition, all the login related to select a subset of data should be implemented here.
 public:
  /// The type of the native identifier of a sample rather than an arbitrarily assigned index
  using sample_name_t = SN;
  /// The type for arbitrarily assigned index
  using sample_id_t = size_t;
  /// Type for list of sample names in a sample file
  using samples_t = std::template vector<sample_name_t>;
  //  using samples_t = std::template list<sample_name_t>;
  /// Type for list where an element is the list of samples in a sample file
  using sample_files_t = std::template vector< std::pair<std::string, samples_t> >;
  //  using sample_files_t = std::template list< std::pair<std::string, samples_t> >;

  sample_list() : m_num_partitions(1u) {}

  /// Set the number of partitions and clear internal states
  bool set_num_partitions(size_t n);

  /// Load a sample list from a file
  bool load(const std::string& samplelist_file);

  /// Extract a sample list from a serialized sample list in a string
  bool load_from_string(const std::string& samplelist);

  /// Allow read only access to the internal list
  const sample_files_t& get_list() const;

  /// Write the current sample list into a file
  bool write(const std::string& out_filename) const;

  /// Clear internal states
  void clear();

  /// Serialize sample list for a partition
  bool to_string(size_t p, std::string& sstr) const;

 protected:

  /// Populate m_samples_per_file by reading from input stream
  size_t get_samples_per_file(std::istream& istr);

  size_t get_samples_per_hdf5_file(std::istream& istr);

  /// Extract m_samples_per_file by parsing a serialized string
  size_t get_samples_per_file(const std::string& samplelist);

  /// Populate the list of starting sample id for each sample file
  bool get_sample_range_per_file();

  /// Populate the list of starting sample id for each partition
  bool get_sample_range_per_part();

  /// Find the range of sample files that covers the range of samples of a partition
  bool find_sample_files_of_part(size_t p, size_t& sf_begin, size_t& sf_end) const;

  static std::string to_string(const std::string& s);

  template <typename T>
  static std::string to_string(const T v);

 protected:

  /// The number of partitions to divide samples into
  size_t m_num_partitions;

  /** In a sample file list, each line begins with a sample file name
   * that is followed by the names of the samples in the file.
   */
  sample_files_t m_samples_per_file;

  /// Contains list of all sample
  samples_t m_sample_list;

  /// Contains starting sample id of each file
  std::vector<sample_id_t> m_sample_range_per_file;

  /// Contains starting sample id of each partition
  std::vector<sample_id_t> m_sample_range_per_part;
};

void handle_mpi_error(int ierr);

void distribute_sample_list(const sample_list<std::string>& sn,
                            std::string& my_samples,
                            lbann_comm& comm);

} // end of namespace

#include "sample_list_impl.hpp"

#endif // __SAMPLE_LIST_HPP__
