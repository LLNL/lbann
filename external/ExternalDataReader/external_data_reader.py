from data_reader_communication_pb2 import Request, Response
from connection import Connection

class ExternalDataReader(object):
    'External DataReader: provides data to lbann'

    def __init__(self, connection, address):
        'Initialize external data reader'
        self.connection = Connection(connection, address)

        self.running = True

        self.params = None

        #https://github.com/keras-team/keras/blob/adc321b4d7a4e22f6bdb00b404dfe5e23d4887aa/keras/engine/training.py

    def get_data_parameters(self):
        pass

    def get_data(self):
        pass

    def get_labels(self):
        pass

    def get_responses(self):
        pass

    def receive_init_request(self):
        data = self.connection.recv_message()
        message = Request()
        message.ParseFromString(data)
        if not message.has_init_request():
            raise RuntimeError("expected init request")
        return message.init_request

    def handle_init_request(self, message):
        if self.params is None:
            self.params = self.get_data_parameters()
        response = Response()
        response.init_response.num_samples = self.params["num_samples"]
        response.init_response.data_size = self.params["data_size"]
        response.init_response.has_labels = self.params["has_labels"]
        response.init_response.num_labels = self.params["num_labels"]
        response.init_response.label_size = self.params["label_size"]
        response.init_response.has_responses = self.params["has_responses"]
        response.init_response.num_responses = self.params["num_responses"]
        response.init_response.response_size = self.params["response_size"]
        response.init_response.data_dims = self.params["data_dims"]
        response.reader_type = self.params["reader_type"]
        return response

    def send_init_response(self, message):
        data = message.SerializeToString()
        self.connection.send_message(data)

    def receive_data_request(self):
        data = self.connection.recv_message()
        message = Request()
        message.ParseFromString(data)
        if not (message.has_fetch_data_request() or
                message.has_fetch_responses_request() or
                message.has_fetch_labels_request()):
            raise RuntimeError("expected data request")
        return message

    def handle_data_request(self, message):
        response = Response()
        if message.has_fetch_data_request():
            response.fetch_data_response.data = self.x
        elif message.has_fetch_labels_request():
            response.fetch_labels_response.labels = self.y
        elif message.has_fetch_responses_request():
            response.fetch_responses_response.responses = self.y
        return response

    def send_data_response(self, message):
        data = message.SerializeToString()
        self.connection.send_message(data)

    def run(self):
        # Init
        request = self.receive_init_request()
        response = self.handle_init_request(request)
        self.send_init_response(response)

        while self.running:
            # TODO add hangup request
            request = self.receive_data_request()
            response = self.handle_data_request(request)
            self.send_data_response(response)
