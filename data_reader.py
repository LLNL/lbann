#from __future__ import print_function

from data_reader_communication_pb2 import Request, Response
import os
import sys
import struct

class DataReader():
    def __init__(self, file_from_lbann, file_to_lbann):
        self.file_from_lbann = file_from_lbann
        self.file_to_lbann = file_to_lbann
        
        self.pipe_from_lbann = -1
        self.pipe_to_lbann = -1
        
        self._connect()

    def __del__(self):
        self._disconnect()

    def _connect(self):
        'Connect to the named pipes provided'
        self.pipe_from_lbann = os.open(self.file_from_lbann, os.O_RDONLY)
        self.pipe_to_lbann = os.open(self.file_to_lbann, os.O_WRONLY)
    
    def _disconnect(self):
        'Disconnect from the named pipes'
        os.close(self.pipe_from_lbann)
        os.close(self.pipe_to_lbann)

    def _read(self):
        bytes = os.read(self.pipe_from_lbann, 4)
        length = struct.unpack("!I", bytes)[0]
        data = os.read(self.pipe_from_lbann, length)
        message = Request() # from generated python
        message.ParseFromString(data)
        return message
        
    def _write(self, message):
        print('[DataReader] ' + message.WhichOneof('response'))
        message_bytes = message.SerializeToString()
        length_bytes = struct.pack("!I", len(message_bytes))
        os.write(self.pipe_to_lbann, length_bytes)
        os.write(self.pipe_to_lbann, message_bytes)
        
    def run(self):
        directory = {
            'init_request' : self.handle_init_request,
            'type_request' : self.handle_type_request,
            'config_request': self.handle_config_request,
            'fetch_datum_request': self.handle_fetch_datum_request,
            'fetch_label_request': self.handle_fetch_label_request,
            'fetch_response_request': self.handle_fetch_response_request
        }

        exit = False
        while not exit:
            message = self._read()
            message_type = message.WhichOneof('request')
            print('[DataReader] ' + message_type)
            directory[message_type](message)

    def handle_init_request(self, message):
        response = Response()
        response.init_response.num_samples = 1
        response.init_response.num_features = 3
        response.init_response.num_labels = 0
        response.init_response.has_labels = False
        response.init_response.has_responses = False
        self._write(response)

    def handle_type_request(self, message):
        response = Response()
        response.type_response.type = "external reader"
        self._write(response)

    def handle_config_request(self, message):
        response = Response()
        response.config_response.label_count = 5
        response.config_response.data_size = 5
        response.config_response.label_size = 5
        response.config_response.dims.append(1)
        self._write(response)

    def handle_fetch_datum_request(self, message):
        data_id = message.fetch_datum_request.data_id
        mb_idx = message.fetch_datum_request.mb_idx
        tid = message.fetch_datum_request.tid
        
        response = Response()
        response.fetch_datum_response.datum.append(data_id)
        response.fetch_datum_response.datum.append(mb_idx)
        response.fetch_datum_response.datum.append(tid)
        self._write(response)

    def handle_fetch_label_request(self, message):
        pass

    def handle_fetch_response_request(self, message):
        pass

def usage():
    'print the proper usage of this program'
    print('[DataReader] usage: python data_reader.py input_file output_file')

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    
    if len(args) == 0:
        usage()
        exit(-1)
    
    dr = DataReader(args[0], args[1])
    dr.run()