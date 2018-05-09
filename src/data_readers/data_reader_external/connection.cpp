#include "lbann/data_readers/data_reader_external/connection.hpp"

#include <cstdio>

namespace lbann {

connection::connection(const std::string &address) {
  m_address = address; // for copy constructor
  m_socket_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);

  if (m_socket_fd == -1) {
    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " external_reader::connect() - socket creation failed");
  }

  sockaddr_un external_socketaddr { 0, '\0' };

  external_socketaddr.sun_family = AF_UNIX;

  // the +1 leaves a null byte at the beginning on purpose
  std::copy(address.begin(), address.end(),
            external_socketaddr.sun_path+1);

  // sizeof(external_socketaddr) gets you the whole character buffer, but
  // the socket address is only '\0' + address, not '\0' + address + '\0'*95
  if (::connect(m_socket_fd,
    reinterpret_cast<sockaddr*>(&external_socketaddr),
    sizeof(external_socketaddr.sun_family) + address.length() + 1) == -1) {

    throw lbann_exception(
      std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
      " external_reader::connect() - connect failed");
  }
}

connection::~connection() {
  ::close(m_socket_fd);
  ::free(m_read_buffer);
  ::free(m_write_buffer);
}

void connection::recv_to_buffer(size_t data_size) {
  // TODO memset m_read_buffer?
  // TODO recv 8k at a time?
  if (data_size > m_read_buffer_size) {
    m_read_buffer = static_cast<uint8_t*>(realloc(m_read_buffer, data_size));
    m_read_buffer_size = data_size;
  }

  size_t offset = 0;
  while (offset < data_size) {
    ssize_t receive_size = recv(m_socket_fd, m_read_buffer + offset, data_size - offset, 0);
//    std::cout << "recv size: " << receive_size << std::endl;
    if (receive_size <= 0) {
      throw lbann_exception(
        std::string{} + __FILE__ + ":" + std::to_string(__LINE__) +
        " external_reader::recv_to_buffer() did not receive enough data\n"
        "\texpected: " + std::to_string(data_size) + " got: " + std::to_string(offset));
    }
    offset += static_cast<size_t>(receive_size);
  }
}

void connection::send_from_buffer(size_t data_size) {
  // m_write_buffer is already correct size

  // TODO send 8k at a time?

  size_t offset = 0;
  while (offset < m_write_buffer_size) {
    ssize_t send_size = send(m_socket_fd, m_write_buffer + offset, data_size - offset, 0);
    if (send_size <= 0) {
      throw lbann_exception(
        std::string{} + __FILE__ + " " + std::to_string(__LINE__) +
        " external_reader::send_from_buffer() did not send enough data");
    }
    offset += static_cast<size_t>(send_size);
  }
}

Response connection::message_read() {
  recv_to_buffer(4);

  size_t response_message_size = 0;
  response_message_size |= m_read_buffer[0] << 24;
  response_message_size |= m_read_buffer[1] << 16;
  response_message_size |= m_read_buffer[2] <<  8;
  response_message_size |= m_read_buffer[3];

  recv_to_buffer(response_message_size);

  Response response;
  response.ParseFromArray(m_read_buffer, response_message_size);
  return response;
}


void connection::message_write(const Request &request) {
  size_t request_size = std::max(request.ByteSizeLong(), 4ul);
  if (request_size > m_write_buffer_size) {
    m_write_buffer = static_cast<uint8_t*>(realloc(m_write_buffer, request_size));
    m_write_buffer_size = request_size;
  }

  m_write_buffer[0] = request_size >> 24;
  m_write_buffer[1] = request_size >> 16;
  m_write_buffer[2] = request_size >>  8;
  m_write_buffer[3] = request_size;
  send_from_buffer(4);

  request.SerializeToArray(m_write_buffer, request_size);
  send_from_buffer(request_size);
}
} // namespace lbann
