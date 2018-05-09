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
    connection(const std::string&);
    ~connection();
    Response message_read();
    void message_write(const Request&);

  private:
    void recv_to_buffer(size_t data_size);
    void send_from_buffer(size_t data_size);

    uint8_t *m_read_buffer = nullptr;
    size_t m_read_buffer_size = 0;

    uint8_t *m_write_buffer = nullptr;
    size_t m_write_buffer_size = 0;

    int m_socket_fd = -1;

    std::string m_address{};
};
} // namespace lbann

#endif  // LBANN_DATA_READER_EXTERNAL_CONNECTION_HPP
