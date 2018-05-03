import socket
import struct

if __name__ == '__main__':
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect("\0socket_test")
#    length_bytes = struct.pack("!I", 256)

#    s.send('0'*50)
#    print(s.recv(500))
#    s.sendall(length_bytes)
#    s.sendall("a"*256)
#    print(s.recv(4))
#    print(s.recv(256))
    s.close()
