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
//
// lbann_data_reader_external .hpp .cpp - generic_data_reader class for data readers connected over IPC
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/data_reader_external/data_reader_external.hpp"
#include <cstdio>
#include <string>

// TODO all response sub-messages can probably be const &

namespace lbann {

external_reader::external_reader(bool shuffle)
  : generic_data_reader(shuffle) {}

external_reader::external_reader(const external_reader& other) :
  generic_data_reader(other) {}

external_reader& external_reader::operator=(const external_reader& other) {
  generic_data_reader::operator=(other);
  return *this;
}

external_reader* external_reader::copy() const {
  return new external_reader(*this);
}

external_reader::~external_reader() {
  disconnect();

  if (m_write_buffer) {
    free(m_write_buffer);
    m_write_buffer = nullptr;
    m_write_buffer_size = 0;
  }
  if (m_read_buffer) {
    free(m_read_buffer);
    m_read_buffer = nullptr;
    m_read_buffer_size = 0;
  }
}

void external_reader::connect() {
  // 1) create tmp dir, tmp fifo files
  // 2) mkfifo
  // 3) fork
  // 4) open fifos
  // if done out of that order, open will block or the external reader
  // won't be able to open the fifos

  // TODO: name these better, e.g. w/ tid
  char lbann_comm_dir[] = "/tmp/lbann_comm.XXXXXX";
  char lbann_to_external_file[] = "/tmp/lbann_comm.XXXXXX/lbann_out";
  char external_to_lbann_file[] = "/tmp/lbann_comm.XXXXXX/lbann_in";

  strcpy(lbann_comm_dir, mkdtemp(lbann_comm_dir));

  if (strcmp(lbann_comm_dir, "") == 0) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " external_reader::connect() - mkdtemp failed");
  }

  strncpy(lbann_to_external_file, lbann_comm_dir, strlen(lbann_comm_dir));
  strncpy(external_to_lbann_file, lbann_comm_dir, strlen(lbann_comm_dir));

  int retval1, retval2;
  retval1 = mkfifo(lbann_to_external_file, 0600);
  retval2 = mkfifo(external_to_lbann_file, 0600);
  if (retval1 < 0 || retval2 < 0) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " external_reader::connect() - mkfifo failed");
  }

  // TODO maybe https://stackoverflow.com/questions/1584956/how-to-handle-execvp-errors-after-fork

  pid_t child;

  switch (child = fork()) {
    case -1: {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " external_reader::connect() - fork failed");
      break;
    }
    case 0: {
      // TODO all of this should be configurable from a datareader prototext file
      char python[] = "python";
      char python_filename[] = "data_reader.py";

      char *argv[5];
      argv[0] = python;
      argv[1] = python_filename;
      argv[2] = lbann_to_external_file;
      argv[3] = external_to_lbann_file;
      argv[4] = nullptr;

      execvp("python", argv);

      // execvp is noreturn if successful
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " external_reader::connect() - exec failed, data reader not started");
      break;
    }
    default: {
      break;
    }
  }

  // open blocks until the other side picks up, do this after the fork
  m_lbann_to_external_fd = open(lbann_to_external_file, O_WRONLY);
  m_external_to_lbann_fd = open(external_to_lbann_file, O_RDONLY);

  if (m_lbann_to_external_fd < 0 || m_external_to_lbann_fd < 0) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " external_reader::connect() - could not open fifo");
  }

  // do these last, everything likes non-const c strings
  // keep them for cleanup in disconnect
  m_lbann_comm_dir = lbann_comm_dir;
  m_lbann_to_external_file = lbann_to_external_file;
  m_external_to_lbann_file = external_to_lbann_file;

  m_connected = true;
}

void external_reader::disconnect() {
  if (m_connected) {

    // kill external data reader
    Request request;
    request.mutable_exit_request();
    {
      std::lock_guard<std::mutex> lock{m_read_write_completion};
      message_write(request);
    }

    // clean up /tmp/lbann_comm
    close(m_lbann_to_external_fd);
    close(m_external_to_lbann_fd);

    unlink(m_lbann_to_external_file.c_str());
    unlink(m_external_to_lbann_file.c_str());

    rmdir(m_lbann_comm_dir.c_str());
  }
}

void external_reader::load() {
  if (m_connected) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " external_reader::load() - load called twice");
  }
  connect();

  // TODO this needs to have the path too
  std::string infile = get_data_filename();

  Request request;
  InitRequest* init_request = request.mutable_init_request();

  init_request->set_filename(infile);

  Response response = message_transaction(request);

  if (!response.has_init_response()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " external_reader::load() - incorrect message type received, expected init");
  }

  InitResponse init_response = response.init_response();

  m_num_samples = init_response.num_samples();
  m_num_features = init_response.num_features();
  m_num_labels = init_response.num_labels();

  m_has_labels = init_response.has_labels();
  m_has_responses = init_response.has_responses();

  // Reset indices.
  m_shuffled_indices.clear();
  m_shuffled_indices.resize(m_num_samples);
  std::iota(m_shuffled_indices.begin(), m_shuffled_indices.end(), 0);
  select_subset_of_data();

  // TODO make this a single message for all configuration data, instead of init+config
  get_config();
}

std::string external_reader::get_type() const {
  Request request;
  request.mutable_type_request();

  Response response = message_transaction(request);

  if (!response.has_type_response()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " external_reader::get_type() - incorrect message type received, expected type");
  }
  return std::string(response.type_response().type());
}

void external_reader::get_config() {
  Request request;
  request.mutable_config_request();

  Response response = message_transaction(request);

  if (!response.has_config_response()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " external_reader::get_config() - incorrect message type received, expected config");
  }
  ConfigResponse config_response = response.config_response();
  m_label_count = config_response.label_count();
  m_data_size = config_response.data_size();
  m_label_size = config_response.label_size();
  for (int i = 0; i < config_response.dims_size(); i++) {
    m_dims.push_back(config_response.dims(i));
  }
}

int external_reader::get_linearized_data_size() const {
  return m_data_size;
}

const std::vector<int> external_reader::get_data_dims() const {
  return m_dims;
}

// TODO test something that has labels/responses
int external_reader::get_num_labels() const {
  return m_label_count;
}

int external_reader::get_linearized_label_size() const {
  return m_label_size;
}

Response external_reader::message_transaction(const Request& request) const {
  Response response;
  {
    // TODO do we need this mutex
    bool locked = m_read_write_completion.try_lock();
    if (!locked) {
        std::cout << "SOMEONE'S BEEN EATING MY LOCKS" << std::endl;
    } else {
        m_read_write_completion.unlock();
    }
    
    std::lock_guard<std::mutex> lock{m_read_write_completion};
    message_write(request);
    response = message_read();
  }
  return response;
}


Response external_reader::message_read() const {
  // Read a message from the open named pipe

  size_t response_size = 0;
  uint8_t size_bytes[4];
  read(m_external_to_lbann_fd, size_bytes, 4);
  response_size |= size_bytes[0] << 24;
  response_size |= size_bytes[1] << 16;
  response_size |= size_bytes[2] <<  8;
  response_size |= size_bytes[3];

  if (response_size > m_read_buffer_size) {
    m_read_buffer = static_cast<uint8_t*>(realloc(m_read_buffer, response_size));
    m_read_buffer_size = response_size;
  }
  read(m_external_to_lbann_fd, m_read_buffer, response_size);

  Response response;
  response.ParseFromArray(m_read_buffer, response_size);

  return response;
}

bool external_reader::message_write(Request request) const {
  // Write a message to the open named pipe

  size_t request_size = request.ByteSizeLong();
  if (request_size > m_write_buffer_size) {
    m_write_buffer = static_cast<uint8_t*>(realloc(m_write_buffer, request_size));
    m_write_buffer_size = request_size;
  }

  request.SerializeToArray(m_write_buffer, request_size);

  uint8_t size_bytes[4];
  size_bytes[0] = request_size >> 24;
  size_bytes[1] = request_size >> 16;
  size_bytes[2] = request_size >>  8;
  size_bytes[3] = request_size;

  bool ok = (write(m_lbann_to_external_fd, size_bytes, 4) == 4);
  ok &=     (write(m_lbann_to_external_fd, m_write_buffer, request_size) == static_cast<ssize_t>(request_size));
  return ok;
}

bool external_reader::fetch_datum(Mat& X, int data_id, int mb_idx, int tid) {
  Request request;
  FetchDatumRequest* fetch_datum_request = request.mutable_fetch_datum_request();
  fetch_datum_request->set_data_id(data_id);
  fetch_datum_request->set_mb_idx(mb_idx);
  fetch_datum_request->set_tid(tid);

  Response response = message_transaction(request);

  if (!response.has_fetch_datum_response()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " external_reader::fetch_datum() - incorrect message type received, expected fetch_datum");
  }

  FetchDatumResponse fetch_datum_response = response.fetch_datum_response();

  if (fetch_datum_response.datum_size() != m_num_features) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " external_reader::fetch_datum() - incorrect dimensionality for received data" +
      " expected: " + std::to_string(m_num_features) +
      " got: " + std::to_string(fetch_datum_response.datum_size()));
  }

  for (int i = 0; i < fetch_datum_response.datum_size(); i++) {
    X(i, mb_idx) = fetch_datum_response.datum(i);
  }
  return true;
}

bool external_reader::fetch_response(Mat& Y, int data_id, int mb_idx, int tid) {
  if (!m_has_responses) {
    throw lbann_exception("external_reader: do not have responses");
  }
  Request request;
  FetchResponseRequest* fetch_response_request = request.mutable_fetch_response_request();
  fetch_response_request->set_data_id(data_id);
  fetch_response_request->set_mb_idx(mb_idx);
  fetch_response_request->set_tid(tid);

  Response response = message_transaction(request);

  if (!response.has_fetch_response_response()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " external_reader::fetch_response() - incorrect message type received, expected fetch_response");
  }

  FetchResponseResponse fetch_response_response = response.fetch_response_response();

  Y(0, mb_idx) = fetch_response_response.response();
  return true;
}

bool external_reader::fetch_label(Mat& Y, int data_id, int mb_idx, int tid) {
  if (!m_has_labels) {
    throw lbann_exception("external_reader: do not have labels");
  }

  Request request;
  FetchLabelRequest* fetch_label_request = request.mutable_fetch_label_request();
  fetch_label_request->set_data_id(data_id);
  fetch_label_request->set_mb_idx(mb_idx);
  fetch_label_request->set_tid(tid);

  Response response = message_transaction(request);

  if (!response.has_fetch_label_response()) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " external_reader::fetch_label() - incorrect message type received, expected fetch_label");
  }

  FetchLabelResponse fetch_label_response = response.fetch_label_response();

  Y(fetch_label_response.label(), mb_idx) = 1;
  return true;
}

}  // namespace lbann
