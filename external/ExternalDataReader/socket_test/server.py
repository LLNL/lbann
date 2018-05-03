import select
import socket
import time
import os
import struct

class Connection(object):
    def __init__(self, manager):
        self.manager      = manager

        self.send_queue   = []
        self.recv_pending = 0
        self.recv_buffer  = []

        self.connection, self.address = self._connect()
        self.manager.register(self.fileno(), self)

    def _connect(self):
        # Accept a connection
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # abstract socket, closes when all connections are released
        s.bind("\0socket_test")
        s.listen(1)

        poller = select.epoll()
        poller.register(s.fileno(), select.EPOLLIN | select.EPOLLONESHOT)

        print('[ConnectionManager] waiting for accept')

        events = poller.poll()
        assert(len(events) == 1)
        fileno, event = events[0]
        assert(fileno == s.fileno())
        connection, address = s.accept()
        poller.unregister(s.fileno())
        s.close()
        poller.close()
        return connection, address

    def fileno(self):
        return self.connection.fileno()

    def send_message(self, message):
        length_bytes = struct.pack("!I", len(message))
        self.send_queue = length_bytes + message
        while send_queue != []:
            manager.process_event()

    def send(self):
        if self.send_queue == []:
            return
        sent = self.connection.send(self.send_data, self.send_size)
        self.send_data = self.send_data[sent:]

    def recv_message(self):
        self.recv_pending = 4
        self.recv_buffer = []
        while self.recv_pending != 0:
            manager.process_event()
        assert(len(self.recv_data) == 4)

        self.recv_pending = struct.unpack("!I", self.recv_data)[0]
        self.recv_buffer = []
        while self.recv_pending:
            manager.process_event()
        return self.recv_data

    def recv(self):
        if self.recv_pending <= 0:
            return
        data = self.connection.recv(self.recv_pending)
        self.recv_data.extend(data)
        self.recv_pending -= len(data)

    def close(self):
        self.connection.close()

class Connection(object):
    def __init__(self):
        self.poller = select.epoll()
        self.connections = {}

    def register(self, fd, connection):
        self.connections[fd] = connection
        self.poller.register(fd)

    def process_event(self):
        if not self.connected:
            return
        events = self.poller.poll()
        for fileno, event in events:
            if event & select.EPOLLIN:
                self.connections[fileno].recv()
            elif event & select.EPOLLOUT:
                self.connections[fileno].send()
            elif event & select.EPOLLHUP:
                # epoll should handle this?
                self.poller.unregister(fileno)
                self.connections[fileno].close()
                del self.connections[fileno]


# data_reader.py:
#   load in data
#   start accepting connections
#   socket(s1, cloexec)
#   bind(s1)
#   for each connection:
#       accept(s1)
#       fork()
#       handle_connection()
