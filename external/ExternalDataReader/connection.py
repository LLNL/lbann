import os
import struct
import select
import socket
import fcntl


class Server(object):
    'A server that accepts connecitions and passes them off to a runner'
    def __init__(self, runner=None):
        '''
        Create server
        runner: a class
            should be constructable with (connection, address) and have a
            run() method that uses the connection socket
        '''
        # open socket
        self.s_fd = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0)

        # set cloexec
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

    def run(self):
        'Spin, waiting for a connection that is passed off to a new runner'
        is_server = True
        while is_server:
            # wait for connection
            events = self.poller.poll()
            for fileno, event in events:
                # event should only be from s_fd
                assert (fileno == self.s_fd.fileno())

                # event will only ever be EPOLLIN
                assert (event == select.EPOLLIN)

                # accept new connection
                connection, address = self.s_fd.accept()

                # create new process to handle connection
                pid = os.fork()

                if pid != 0:
                    # allow process to close cleanly
                    is_server = False

                    runner = self.runner(connection, address)
                    runner.run()

                    # skip the rest of the events
                    # s_fd shouldn't even be open in this process
                    break


class Connection(object):
    'Connection handler for socket connections'
    def __init__(self, connection, address):
        self.poller = select.epoll()
        self.connection = connection
        self.address = address
        self.poller.register(self.connection.fileno())

        self.send_offset = 0
        self.send_queue = []
        self.recv_pending = 0
        self.recv_buffer = []

        self.closed = False

    def send_message(self, message):
        print('send_message')
        length_bytes = struct.pack("!I", len(message))
        self.send_offset = 0
        self.send_queue = length_bytes + message
        while self.send_offset != len(self.send_queue):
            self.process_event()

    def send(self):
        print('send')
        assert (self.send_offset <= len(self.send_queue))
        if self.send_offset == len(self.send_queue):
            return
        sent = self.connection.send(self.send_queue[self.send_offset:])
        if sent == 0:
            self.closed = True
        self.send_offset += sent

    def recv_message(self):
        print('recv_message')
        self.recv_pending = 4
        self.recv_buffer = []
        while self.recv_pending != 0:
            self.process_event()
        assert (len(self.recv_buffer) == 4)

        recv_size = struct.unpack("!I", self.recv_buffer)[0]
        self.recv_pending = recv_size
        self.recv_buffer = []
        while self.recv_pending != 0:
            self.process_event()
        assert (len(self.recv_buffer) == recv_size)
        return self.recv_buffer

    def recv(self):
        print('recv')
        assert (self.recv_pending >= 0)
        if self.recv_pending == 0:
            return
        data = self.connection.recv(self.recv_pending)

        # recv_pending > 0, so this can only happen when other side hung up
        if len(data) == 0:
            self.closed = True

        self.recv_buffer.extend(data)
        self.recv_pending -= len(data)

    def process_event(self):
        if self.closed:
            raise RuntimeError("Connection hung up (1)")
        print('process_event')
        events = self.poller.poll()
        print(events)
        for fileno, event in events:
            assert (fileno == self.connection.fileno())
            if event & select.EPOLLIN:
                self.recv()
            if event & select.EPOLLOUT:
                self.send()
            if event & select.EPOLLHUP:
                # shouldn't happen, basically always waiting on send/recv
                raise RuntimeError("Connection hung up (2)")
