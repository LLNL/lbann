import os
import struct
import select
import socket
import fcntl
import sys


class Server(object):
    'A server that accepts connecitions and passes them off to a runner'
    def __init__(self, runner):
        '''
        Create a server to run in the background

        runner: a subclass of Connection
        '''
        # open socket
        self.s_fd = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0)

        # set cloexec, so
        flags = fcntl.fcntl(self.s_fd, fcntl.F_GETFD)
        fcntl.fcntl(self.s_fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)

        # bind socket
        # http://man7.org/linux/man-pages/man7/unix.7.html
        #   section: Abstract sockets
        self.s_fd.bind("\0socket_test")
        # this process is only an epoll-driven accept loop,
        # shouldn't need a listen queue
        self.s_fd.listen(0)

        # setup epoll
        self.poller = select.epoll()
        self.poller.register(self.s_fd, select.EPOLLIN)

        self.runner = runner

        # done init, fork into the background
        pid = os.fork()
        if pid != 0:
            sys.exit(0)

    def run(self):
        'Spin, waiting for a connection that is passed off to a new runner'
        is_server = True
        while is_server:
            # wait for connection
            events = self.poller.poll()
            for fileno, event in events:
                connection, address = self.s_fd.accept()
                pid = os.fork()

                if pid != 0:
                    # allow process to close cleanly
                    is_server = False

                    runner = self.runner(connection, address)
                    runner.run()

                    # skip the rest of the events
                    # s_fd won't be open in this process
                    break


class Connection(object):
    'Connection handler for socket connections'
    def __init__(self, connection, address):
        self.poller = select.epoll()
        self.connection = connection
        self.address = address
        self.poller.register(self.connection.fileno())

        self.send_offset = 0
        self.send_queue = ''
        self.recv_pending = 0
        self.recv_buffer = ''

        self.closed = False

        self.epoll_default_eventmask = select.EPOLLPRI | select.EPOLLHUP
        self.epoll_in  = False
        self.epoll_out = True

    def update_epoll(self):
        'Set the epoll event flags to their correct values'
        eventmask = self.epoll_default_eventmask
        if self.epoll_in:
            eventmask |= select.EPOLLIN
        if self.epoll_out:
            eventmask |= select.EPOLLOUT
        self.poller.modify(self.connection.fileno(), eventmask)

    def send_message(self, message):
        'Send a string of bytes over the connection, with an added length prefix'
        # TODO variable-length prefixing?

        # tell epoll we want to write
        # TODO make this argument-based
        self.epoll_out = True
        self.update_epoll()

        length_bytes = struct.pack("!I", len(message))

        # TODO this can be better encapsulated
        self.send_offset = 0
        self.send_queue = length_bytes + message
        while self.send_offset != len(self.send_queue):
            self.process_event()

        # tell epoll we're done writing
        self.epoll_out = False
        self.update_epoll()

    def send(self):
        'Try to send the remainder of the send_queue, update send_offset by how many were actually sent'
        assert (self.send_offset <= len(self.send_queue))
        if self.send_offset == len(self.send_queue):
            return
        sent = self.connection.send(self.send_queue[self.send_offset:])
        if sent == 0:
            self.closed = True
        self.send_offset += sent

    def recv_message(self):
        'Receive a length-prefixed string of bytes over the connection'

        self.epoll_in = True
        self.update_epoll()

        self.recv_pending = 4
        self.recv_buffer = ''
        while self.recv_pending != 0:
            self.process_event()
        assert (len(self.recv_buffer) == 4)
        recv_size = struct.unpack("!I", self.recv_buffer)[0]
        self.recv_pending = recv_size
        self.recv_buffer = ''
        while self.recv_pending != 0:
            self.process_event()
        assert (len(self.recv_buffer) == recv_size)

        self.epoll_in = False
        self.update_epoll()

        return self.recv_buffer

    def recv(self):
        'Try to receive recv_pending bytes into recv_buffer'
        assert (self.recv_pending >= 0)
        if self.recv_pending == 0:
            return
        data = self.connection.recv(self.recv_pending)

        if len(data) == 0:
            self.closed = True

        self.recv_buffer += data
        self.recv_pending -= len(data)

    def process_event(self):
        'Wait on socket read/write availability, call recv/send as needed'
        if self.closed:
            raise RuntimeError("Connection hung up (1)")
        events = self.poller.poll()
        for fileno, event in events:
            assert (fileno == self.connection.fileno())

            if event & select.EPOLLIN:
                self.recv()
            if event & select.EPOLLOUT:
                self.send()
            if event & select.EPOLLHUP:
                raise RuntimeError("Connection hung up (2)")
