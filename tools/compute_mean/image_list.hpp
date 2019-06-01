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

#ifndef _TOOLS_COMPUTE_MEAN_IMAGE_LIST_HPP_
#define _TOOLS_COMPUTE_MEAN_IMAGE_LIST_HPP_

#include "mpi_states.hpp"
#include <vector>
#include <string>

namespace tools_compute_mean {

class image_list {
 protected:
  /// The num of ranks that only have assigned data
  unsigned int m_effective_num_ranks;
  /// The root data directory
  std::string m_root_data_path;
  /**
   * The file that contains the list of image data, in which each line consists
   * of the path of a image file relative to the root data directory and
   * its label (which we do not really use, and thus ignore).
   */
  std::string m_data_list;
  /** The root output directory, where the mean image will be stored.
   *  If cropped images are stored, they will be written to the same relative path
   *  as they were in the input directory but relative to the root output directory
   *  instead of the root data directory.
   */
  std::string m_out_dir;
  /// The list of images to process, divided among ranks.
  std::vector<std::string> m_image_list;

 protected:
  static bool read_path(const std::string& line, std::vector<std::string>& container);
  static bool read_image_file_name(const std::string& line, std::vector<std::string>& container);
  static void load_file(const std::string& file_name, const mpi_states& ms, std::string& buf);

  void load_list(const mpi_states& ms);
  void create_dirs(const mpi_states& ms, const bool write_cropped) const;
  void create_subdirs(const mpi_states& ms) const;
  void split_list(const mpi_states& ms);

 public:
  image_list(const std::string data_path_file, const bool write_cropped, const mpi_states& ms);

  /// Return the num of ranks that only have assigned data.
  unsigned int get_effective_num_ranks() const {
    return m_effective_num_ranks;
  }
  /// Return the list of images to process by the current rank.
  const std::vector<std::string>& get_image_list() const {
    return m_image_list;
  }
  /// Return the root data directory.
  std::string get_root_data_path() const {
    return m_root_data_path;
  }
  /// Return the root output directory.
  std::string get_out_dir() const {
    return m_out_dir;
  }
  std::string get_image_name_with_new_ext(const size_t i, const std::string new_ext) const;
  void description() const;

};

} // end of namespace tools_compute_mean
#endif // _TOOLS_COMPUTE_MEAN_IMAGE_LIST_HPP_
