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

#include <mpi.h>
#include <unordered_set>
#include "image_list.hpp"
#include "text_read.hpp"
#include "file_utils.hpp"


namespace tools_compute_mean {

/**
 * Read the data path file, load the input image list, and determine the subset
 * of images for the current rank to process.
 * In addition, the root rank will create the output directories as needed.
 */
image_list::image_list(const std::string data_path_file, const bool write_cropped,
                       const mpi_states& ms ) {
  std::string buf;
  load_file(data_path_file, ms, buf);

  std::vector<std::string> lines;
  bool ok = read_text(buf, lines, read_path);
  if (!ok) {
    ms.abort("Could not read the data path file " + data_path_file);
  }

  if (lines.size() != 3) {
    ms.abort(data_path_file + " must contain three lines (paths).");
  }

  m_effective_num_ranks = ms.get_num_ranks();
  m_root_data_path = lbann::add_delimiter(lines[0]);
  m_data_list = lines[1];
  m_out_dir = lbann::add_delimiter(lines[2]);

  load_list(ms);
  create_dirs(ms, write_cropped);
  split_list(ms);
}


/// Parse a line in a data path file
bool image_list::read_path(const std::string& line, std::vector<std::string>& container) {
  std::stringstream sstr(line.c_str());

  std::string path;

  sstr >> path;
  if (sstr.bad()) {
    std::cerr << '[' << line << ']' << std::endl;
    return false;
  }
  container.push_back(path);

  return true;
}


/**
 * Parse a line of the data list file., and add the result into the container
 * Each line contains the path to the image data file relative to the
 * m_root_data_path and its label.
 */
bool image_list::read_image_file_name(const std::string& line, std::vector<std::string>& container) {
  std::stringstream sstr(line.c_str());

  std::string fileName;
  int label;

  sstr >> fileName;
  if (!sstr.good()) {
    std::cerr << '[' << line << ']' << std::endl;
    return false;
  }
  sstr >> label;
  if (sstr.bad()) {
    std::cerr << '[' << line << ']' << std::endl;
    return false;
  }
  container.push_back(fileName); // we do not need label here

  return true;
}


/**
 *  The root MPI rank loads a file into a buffer, then broadcast it to other ranks
 */
void image_list::load_file(const std::string& file_name, const mpi_states& ms, std::string& buf) {

  int mc = MPI_SUCCESS;
  unsigned long buf_size = 0ul;
  bool ok = true;

  // root rank load a file into a buffer, which wil be broadcast later
  if (ms.is_root()) {
    ok = lbann::load_file(file_name, buf);
    if (!ok) {
      ms.abort("Could not read the file : " + file_name);
    }
    buf_size = static_cast<unsigned long>(buf.size());
  }

  if (!ms.is_serial_run()) {
    // first broadcast the size of the buffer
    mc = MPI_Bcast(_VoidP(&buf_size), 1, MPI_UNSIGNED_LONG, ms.m_root, ms.get_comm());
    ms.check_mpi(mc);

    if (!ms.is_root()) { // resize the buffer accordingly to receive data
      buf.resize(buf_size, 0);
    }

    if (buf_size == 0ul) {
      ms.abort("Empty text buffer!");
    }

    // broadcast the text buffer
    mc = MPI_Bcast(_VoidP(&buf[0]), static_cast<int>(buf.size()), MPI_CHAR, ms.m_root, ms.get_comm());
    ms.check_mpi(mc);
  }
}


/// Split the image file list among MPI ranks and have each rank put its portion into a vector
void image_list::load_list(const mpi_states& ms) {
  std::string textbuf;
  load_file(m_data_list, ms, textbuf);

  bool ok = read_text(textbuf, m_image_list, read_image_file_name);
  if (!ok) {
    ms.abort("Could not read the list in buffer.");
  }
}


/// Create the root output directory under which the mean image will be stored.
void image_list::create_dirs(const mpi_states& ms, const bool write_cropped) const {
  if (ms.is_root()) {
    bool ok = lbann::create_dir(m_out_dir);
    if (!ok) {
      ms.abort("Could not create the out dir: " + m_out_dir);
    }

    if (write_cropped) { // only create subdirectories when needed to write cropped images
      create_subdirs(ms);
    }
  }

  int mc = MPI_Barrier(ms.get_comm());
  ms.check_mpi(mc);
}


/// Create the sub directories for each individual image under the root output directory.
void image_list::create_subdirs(const mpi_states& ms) const {
  std::unordered_set<std::string> subdirs;
  for (const auto& image_path : m_image_list) {
    std::string subdir;
    std::string basename;
    lbann::parse_path(image_path, subdir, basename);
    subdirs.insert(subdir);
  }
  for (const auto& subdir : subdirs) {
    //std::cout << "creating directory " + m_out_dir + subdir << std::endl;
    bool ok = lbann::create_dir(m_out_dir + subdir);
    if (!ok) {
      ms.abort("Could not create the out dir: " + (m_out_dir + subdir));
    }
  }
}


/**
 * Divide the set of input images among ranks, and find out which subset of
 * images for the current rank to process.
 */
void image_list::split_list(const mpi_states& ms) {
  if (m_image_list.size() < static_cast<size_t>(ms.get_num_ranks())) {
#if 1
    if (ms.is_root()) {
      std::cerr << "Needlessly large number of ranks compared to the number of images." << std::endl;
    }
#else
    ms.abort("Needlessly large number of ranks compared to the number of images.");
#endif
  }
  std::vector<unsigned int> num_per_rank;
  ms.split_over_ranks(m_image_list.size(), num_per_rank);
  std::cout << "rank " << ms.get_my_rank() << " has images ["
            << num_per_rank[ms.get_my_rank()] << " - " << num_per_rank[ms.get_my_rank()+1]
            << ')' << std::endl;

  // Scan m_image_list from the beginning to count how many ranks have zero.
  // Update the effective number of ranks
  unsigned int list_start = 0;

  for (unsigned int i = 1u; i < num_per_rank.size(); ++i) {
    unsigned int n = num_per_rank[i] - list_start;
    list_start = num_per_rank[i];
    if (n == 0u) {
      m_effective_num_ranks--;
    } else {
      break;
    }
  }

  // Select the subset of the list assigned to the current rank
  std::vector<std::string> selected(m_image_list.begin() + num_per_rank[ms.get_my_rank()],
                                    m_image_list.begin() + num_per_rank[ms.get_my_rank()+1]);
  m_image_list.swap(selected);
}


/// Obtain the new path to the image file to write with new file type extension.
std::string image_list::get_image_name_with_new_ext(const size_t i, const std::string new_ext) const {
  std::string subdir;
  std::string basename;
  lbann::parse_path(m_image_list.at(i), subdir, basename);
  return (m_out_dir + subdir + lbann::get_basename_without_ext(basename) + new_ext);
}


/// Print out the path prameters read, and the number of images for this rank to process.
void image_list::description() const {
  std::cout << " - m_root_data_path: " << m_root_data_path << std::endl;
  std::cout << " - m_data_list : " << m_data_list << std::endl;
  std::cout << " - m_out_dir : " << m_out_dir << std::endl;
  std::cout << " - m_image_list.size() : " << m_image_list.size() << std::endl;
}

} // end of namespace tools_compute_mean
