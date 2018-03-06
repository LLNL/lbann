#include "lbann/data_readers/data_reader_external/data_reader_external.hpp"
#include <cstdio>
#include <string>

namespace lbann {

external_reader::external_reader(bool shuffle)
  : generic_data_reader(shuffle) {}

external_reader::external_reader(const external_reader& other) :
  generic_data_reader(other) {}

external_reader& external_reader::operator=(const external_reader& other) {
  generic_data_reader::operator=(other);
  return *this;
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

  // Multiple data reader TODO: name these better, e.g. w/ tid
  char lbann_comm_dir[] = "/tmp/lbann_comm.XXXXXX";
  char lbann_to_external_file[] = "/tmp/lbann_comm.XXXXXX/lbann_out";
  char external_to_lbann_file[] = "/tmp/lbann_comm.XXXXXX/lbann_in";
  
  strcpy(lbann_comm_dir, mkdtemp(lbann_comm_dir));
  
  if (strcmp(lbann_comm_dir, "") == 0) {
    throw lbann_exception("mkdtemp failed");
  }

  strncpy(lbann_to_external_file, lbann_comm_dir, strlen(lbann_comm_dir));
  strncpy(external_to_lbann_file, lbann_comm_dir, strlen(lbann_comm_dir));
  
  int retval1, retval2;
  retval1 = mkfifo(lbann_to_external_file, 0600);
  retval2 = mkfifo(external_to_lbann_file, 0600);
  if (retval1 < 0 || retval2 < 0) {
    throw lbann_exception("mkfifo failed");
  }

  // TODO find some way to detect failure to launch data reader,
  // currently it just doesn't work and lbann has to be killed
  // https://stackoverflow.com/questions/1584956/how-to-handle-execvp-errors-after-fork

  pid_t child;

  switch (child = fork()) {
    case -1: {
      throw lbann_exception("fork");
      break;
    }
    case 0: {
      char python[] = "python";
      char python_filename[] = "data_reader.py";

      char *argv[5];
      argv[0] = python;
      argv[1] = python_filename;
      argv[2] = lbann_to_external_file;
      argv[3] = external_to_lbann_file;
      argv[4] = nullptr;

      execvp("python", argv);
      
      //execvp is noreturn if successful
      throw lbann_exception("failed to run external data reader");
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
    throw lbann_exception("could not open fifo");
  }

  // do these last, everything likes non-const c strings
  m_lbann_comm_dir = lbann_comm_dir;
  m_lbann_to_external_file = lbann_to_external_file;
  m_external_to_lbann_file = external_to_lbann_file;

  m_connected = true;
}

void external_reader::disconnect() {
  if (m_connected) {
    // TODO send exit message
    close(m_lbann_to_external_fd);
    close(m_external_to_lbann_fd);

    unlink(m_lbann_to_external_file.c_str());
    unlink(m_external_to_lbann_file.c_str());

    rmdir(m_lbann_comm_dir.c_str());
  }
}

void external_reader::load() {
  if (m_connected) {
    throw lbann_exception("double load");
  }
  connect();
    
  std::string infile = get_data_filename();

  Request request;
  InitRequest init_request = request.init_request();
  init_request.set_filename(infile);
  
  message_write(request);
  
  Response response = message_read();
  if (!response.has_init_response()) {
    throw lbann_exception("init response not received");
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
}

std::string external_reader::get_type() const {
  Request request;
  request.type_request();
  message_write(request);
  Response response = message_read();
  if (!response.has_type_response()) {
    throw lbann_exception("type response not received");
  }
  return std::string(response.type_response().type());
}

external_reader* external_reader::copy() const {
  // TODO this should be fixed;
  return new external_reader(*this);
}

void external_reader::get_config() const {
  Request request;
  request.config_request();
  message_write(request);
  Response response = message_read();
  if (!response.has_config_response()) {
    throw lbann_exception("config response not received");
  }
  ConfigResponse config_response = response.config_response();
  m_label_count = config_response.label_count();
  m_data_size = config_response.data_size();
  m_label_size = config_response.label_size();
  for (int i = 0; i < config_response.dims_size(); i++) {
    m_dims.push_back(config_response.dims(i));
  }
}

int external_reader::get_num_labels() const {
  if (m_label_count == -1) {
    get_config();
  }
  return m_label_count;
}

int external_reader::get_linearized_data_size() const {
  if (m_data_size == -1) {
    get_config();
  }
  return m_data_size;
}

int external_reader::get_linearized_label_size() const {
  if (m_label_size == -1) {
    get_config();
  }
  return m_label_size;
}

const std::vector<int> external_reader::get_data_dims() const {
  if (m_dims.size() == 0) {
    get_config();
  }
  return m_dims;
}

Response external_reader::message_read() const {
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
  FetchDatumRequest fetch_datum_request = request.fetch_datum_request();
  fetch_datum_request.set_data_id(data_id);
  fetch_datum_request.set_mb_idx(mb_idx);
  fetch_datum_request.set_tid(tid);
  
  message_write(request);
  
  Response response = message_read();
  if (!response.has_fetch_datum_response()) {
    throw lbann_exception("datum response not received");
  }
  
  FetchDatumResponse fetch_datum_response = response.fetch_datum_response();
  
  if (fetch_datum_response.datum_size() != m_num_features) {
    throw lbann_exception("datum of incorrect size");
  }
  
  for (int i = 0; i < fetch_datum_response.datum_size(); i++) {\
    X(i, mb_idx) = fetch_datum_response.datum(i);
  }
  return true;
}

bool external_reader::fetch_label(Mat& Y, int data_id, int mb_idx, int tid) {
  if (!m_has_labels) {
    throw lbann_exception("external_reader: do not have labels");
  }

  throw lbann_exception("external reader does not support label fetch yet");

  return true;
}

bool external_reader::fetch_response(Mat& Y, int data_id, int mb_idx, int tid) {
  if (!m_has_responses) {
    throw lbann_exception("external_reader: do not have responses");
  }

  throw lbann_exception("external reader does not support response fetch yet");

  return true;
}

}  // namespace lbann
