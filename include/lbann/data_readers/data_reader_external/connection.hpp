#ifndef LBANN_DATA_READER_EXTERNAL_CONNECTION_HPP
#define LBANN_DATA_READER_EXTERNAL_CONNECTION_HPP

#include <data_reader_communication.pb.h>

#include <cstddef> // size_t
#include <cstdint> // uint8_t
#include <sys/socket.h> // socket stuff
#include <sys/un.h>
#include <unistd.h> // close

#include <string>

 #include "lbann/utils/exception.hpp"

namespace lbann {

class connection {
  public:
    // TODO use ctor/dtor instead of connect/disconnect
    void connect(const std::string&);
    void disconnect();
    Response message_read();
    void message_write(const Request&);

  private:
    void recv_to_buffer(size_t data_size);
    void send_from_buffer(size_t data_size);

    uint8_t *m_read_buffer;
    size_t m_read_buffer_size;

    uint8_t *m_write_buffer;
    size_t m_write_buffer_size;

    int m_socket_fd;

    bool m_connected;
};
} // namespace lbann

#endif  // LBANN_DATA_READER_EXTERNAL_CONNECTION_HPP
