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
// lbann_file_io .hpp .cpp - Input / output utilities
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_IO_PERSIST_IMPL_H
#define LBANN_IO_PERSIST_IMPL_H

#include "lbann/comm_impl.hpp"
#include "lbann/io/persist.hpp"
#include "lbann/utils/serialize.hpp"

namespace lbann {

/** Archive for checkpoint and restart */
template <class Archive>
void persist::serialize(Archive & ar) {
  ar(CEREAL_NVP(ckpt_type));
}

template <typename C>
void write_cereal_archive(C& obj, const std::string& filename) {
  std::ofstream os(filename);
  if(!os.is_open()) {
    throw NonexistentArchiveFile(filename);
  }
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  cereal::XMLOutputArchive archive(os);
#else // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
  cereal::BinaryOutputArchive archive(os);
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  archive(obj);
}

template <typename C>
void write_cereal_archive(C& obj, persist& p, const std::string& filename) {
  write_cereal_archive<C>(obj, p.get_checkpoint_dir() + "/" + filename);
}

template <typename C>
void write_cereal_archive(C& obj, persist& p, persist_type pt, const std::string& suffix) {
  write_cereal_archive<C>(obj, p.get_filename(pt) + suffix);
}

template <typename C>
void write_cereal_archive(C& obj, persist& p, execution_mode mode, const std::string& suffix) {
  const persist_type pt = execution_mode_to_persist_type(mode);
  write_cereal_archive<C>(obj, p, pt, suffix);
}

template <typename C>
void read_cereal_archive(C& obj, const std::string& filename) {
  std::ifstream is(filename);
  if(!is.is_open()) {
    throw NonexistentArchiveFile(filename);
  }
#ifdef LBANN_HAS_CEREAL_XML_ARCHIVES
  cereal::XMLInputArchive archive(is);
#else // defined LBANN_HAS_CEREAL_BINARY_ARCHIVES
  cereal::BinaryInputArchive archive(is);
#endif // LBANN_HAS_CEREAL_XML_ARCHIVES
  archive(obj);
}

template <typename C>
void read_cereal_archive(C& obj, persist& p, const std::string& filename) {
  read_cereal_archive(obj, p.get_checkpoint_dir() + "/" + filename);
}

template <typename C>
void read_cereal_archive(C& obj, persist& p, persist_type pt, const std::string& suffix) {
  read_cereal_archive(obj, p.get_filename(pt) + suffix);
}

template <typename C>
void read_cereal_archive(C& obj, persist& p, execution_mode mode, const std::string& suffix) {
  const persist_type pt = execution_mode_to_persist_type(mode);
  read_cereal_archive<C>(obj, p, pt, suffix);
}

template <typename C>
std::string create_cereal_archive_binary_string(C& obj) {
  std::ostringstream ss;
  {
    cereal::BinaryOutputArchive archive(ss);
    archive(obj);
  } // archive goes out of scope, ensuring all contents are flushed
  return ss.str();
}

template <typename C>
void unpack_cereal_archive_binary_string(C& obj, const std::string& buf) {
  std::istringstream ss(buf);
  {
    cereal::BinaryInputArchive archive(ss);
    archive(obj);
  } // archive goes out of scope, ensuring all contents are flushed
}

template <typename C>
void load_from_shared_cereal_archive(C& obj,
                                     lbann_comm& comm,
                                     const std::string& filename) {
  std::string buf;
  if (comm.am_trainer_master()) {
    read_cereal_archive<C>(obj, filename);
    buf = create_cereal_archive_binary_string<C>(obj);
  }else {
    // If you are not the trainer master, still check to see if the file exists
    std::ifstream is(filename);
    if(!is.is_open()) {
      throw NonexistentArchiveFile(filename);
    }
  }

  // TODO: this assumes homogeneous processors
  // broadcast state from rank 0
  comm.trainer_broadcast(0, buf);

  if (!comm.am_trainer_master()) {
    unpack_cereal_archive_binary_string<C>(obj, buf);
  }
}

template <typename C>
void load_from_shared_cereal_archive(C& obj, persist& p,
                                     lbann_comm& comm,
                                     const std::string& filename) {
  load_from_shared_cereal_archive(obj, comm, p.get_checkpoint_dir() + filename);
}

template <typename C>
void load_from_shared_cereal_archive(C& obj, persist& p, persist_type pt,
                                     lbann_comm& comm,
                                     const std::string& suffix) {
  load_from_shared_cereal_archive(obj, comm, p.get_filename(pt) + suffix);
}

template <typename C>
void load_from_shared_cereal_archive(C& obj, persist& p, execution_mode mode,
                                     lbann_comm& comm,
                                     const std::string& suffix) {
  const persist_type pt = execution_mode_to_persist_type(mode);
  load_from_shared_cereal_archive<C>(obj, p, pt, comm, suffix);
}

} // namespace lbann
#endif // LBANN_IO_PERSIST_IMPL_H
