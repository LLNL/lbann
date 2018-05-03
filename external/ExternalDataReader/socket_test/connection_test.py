import select
import socket
import time
import os
#import queue
import struct

class Connection(object):
    def __init__(self, connection, address=None):
        self.connection = connection
        self.address    = address

    def fileno(self):
        return self.connection.fileno()

    def send(self, data):
        self.connection.sendall(data)

    def recv(self, size):
        result = []
        while len(result) < size:
            size_to_go = size - len(result)
            result.extend(self.connection.recv(self.buffer_size))
        assert(len(result) == size), (len(result), size)
        return result

socket_a, socket_b = socket.socketpair()

for i in range(1):
    data_size = 229248
    data = "0"*data_size
    print('sending length {}'.format(data_size))
    socket_a.send(data)
    socket_b.recv(data_size)
    socket_b.send(data)
    socket_a.recv(data_size)
