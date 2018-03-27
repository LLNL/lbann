################################################################################
# Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. <lbann-dev@llnl.gov>
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http:#software.llnl.gov/LBANN or
# https://github.com/LLNL/LBANN.
#
# Licensed under the Apache License, Version 2.0 (the "Licensee"); you
# may not use this file except in compliance with the License.  You may
# obtain a copy of the License at:
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the license.
#
# data_reader.py - data reader connected over IPC
################################################################################

from data_reader_communication_pb2 import Request, Response
import os
import sys
import struct

class DataReader():
    'External DataReader: connects to lbann over a pair of named pipes'
    def __init__(self, file_from_lbann, file_to_lbann):
        'Initialize data parameters and connect to the named pipes'

        # Named pipe files
        self.file_from_lbann = file_from_lbann
        self.file_to_lbann = file_to_lbann

        # Named pipe file descriptors
        self.pipe_from_lbann = -1
        self.pipe_to_lbann = -1

        # Whether an exit message has been received
        self.exit = False

        # Initial configuration
        self.num_samples = 0
        self.num_features = 0
        self.num_labels = 0
        self.has_labels = False
        self.has_responses = False
        self.label_count = 0
        self.data_size = 0
        self.label_size = 0
        self.type = ''
        self.dims = []

        # Perform reader-specific configuration
        self.configure()

        # Connect to named pipes
        self._connect()

    def __del__(self):
        'Destructor'
        self._disconnect()

    def configure():
        'Perform subclassed reader-specific configuration'
        pass

    def _connect(self):
        'Connect to the named pipes'
        self.pipe_from_lbann = os.open(self.file_from_lbann, os.O_RDONLY)
        self.pipe_to_lbann = os.open(self.file_to_lbann, os.O_WRONLY)

    def _disconnect(self):
        'Disconnect from the named pipes'
        # TODO decide how to clean up /tmp/lbann_comm
        if self.pipe_from_lbann != -1:
            os.close(self.pipe_from_lbann)
        if self.pipe_to_lbann != -1:
            os.close(self.pipe_to_lbann)

    def _read(self):
        'Read a length-prefixed protobuf message'
        bytes = os.read(self.pipe_from_lbann, 4)
        if len(bytes) < 4:
            raise IOError('[DataReader] did not recieve enough data')
        length = struct.unpack("!I", bytes)[0]
        data = os.read(self.pipe_from_lbann, length)
        if len(data) < length:
            raise IOError('[DataReader] did not recieve enough data')
        message = Request() # from generated python
        message.ParseFromString(data)
        return message

    def _write(self, message):
        'Write a length-prefixed protobuf message'
        #print('[DataReader] ' + message.WhichOneof('response'))
        message_bytes = message.SerializeToString()
        length_bytes = struct.pack("!I", len(message_bytes))
        os.write(self.pipe_to_lbann, length_bytes)
        os.write(self.pipe_to_lbann, message_bytes)

    def run(self):
        'Wait for a message then send a response'
        directory = {
            'init_request' : self.handle_init_request,
            'type_request' : self.handle_type_request,
            'config_request': self.handle_config_request,
            'fetch_datum_request': self.handle_fetch_datum_request,
            'fetch_label_request': self.handle_fetch_label_request,
            'fetch_response_request': self.handle_fetch_response_request,
            'exit_request': self.handle_exit_request
        }

        self.exit = False
        while not self.exit:
            message = self._read()
            message_type = message.WhichOneof('request')
            #print('[DataReader] ' + message_type)
            directory[message_type](message)

    def handle_exit_request(self, message):
        'Stop DataReader'
        self.exit = True

    def handle_init_request(self, message):
        response = Response()
        response.init_response.num_samples = self.num_samples
        response.init_response.num_features = self.num_features
        response.init_response.num_labels = self.num_labels
        response.init_response.has_labels = self.has_labels
        response.init_response.has_responses = self.has_responses
        self._write(response)

    def handle_type_request(self, message):
        'Get DataReader type'
        response = Response()
        response.type_response.type = self.type
        self._write(response)

    def handle_config_request(self, message):
        'Get DataReader configuration'
        response = Response()
        response.config_response.label_count = self.label_count
        response.config_response.data_size = self.data_size
        response.config_response.label_size = self.label_size
        for dim in self.dims:
            response.config_response.dims.append(dim)
        self._write(response)

    # Implement these per DataReader
    def handle_fetch_datum_request(self, message):
        'Get one item of data'
        raise NotImplementedError()

    def handle_fetch_label_request(self, message):
        'Get one label'
        raise NotImplementedError()

    def handle_fetch_response_request(self, message):
        'Get one response'
        raise NotImplementedError()

# copied originally from p2b1_mol_AE.py:Candle_Molecular_Train
# modifications:
#  * removed train_ac(), format_data()
#  * removed class members not required for datagen
#  * removed internal randomization
#  * removed debug print()s
#  * removed all customization except molecular_nbrs
#  * TODO make this able to take directory information

class Candle_Molecular_Datagen():
    def __init__(self):
        import helper
        self.numpylist, _ = helper.get_local_files('3k_run16')
        self.molecular_nbrs = 10

    def datagen(self):
        import numpy as np
        import helper
        X_all = np.array([])
        nbrs_all = np.array([])
        resnums_all = np.array([])
        files = self.numpylist
        # Training only on few files
        order = [0]
        # Randomize files after first training epoch

        for f_ind in order:
            (X, nbrs, resnums) = helper.get_data_arrays(files[f_ind])

            # normalizing the location coordinates and bond lengths and scale type encoding
            # Changed the xyz normalization from 255 to 350
            Xnorm = np.concatenate([X[:, :, :, 0:3]/320., X[:, :, :, 3:8], X[:, :, :, 8:]/10.], axis=3)

            num_frames = X.shape[0]
            input_feature_dim = np.prod(Xnorm.shape[2:])

            xt_all = np.array([])
            yt_all = np.array([])

            for i in range(num_frames):

                xt = Xnorm[i]
                xt = helper.get_neighborhood_features(xt, nbrs[i], self.molecular_nbrs)

                yt = xt.copy()
                #xt = xt.reshape(xt.shape[0], 1, xt.shape[1], 1)

                if not len(xt_all):
                    xt_all = np.expand_dims(xt, axis=0)
                    yt_all = np.expand_dims(yt, axis=0)
                else:
                    xt_all = np.append(xt_all, np.expand_dims(xt, axis=0), axis=0)
                    yt_all = np.append(yt_all, np.expand_dims(yt, axis=0), axis=0)

            yield files[f_ind], xt_all, yt_all

        return


class Pilot2DataReader(DataReader):
    def configure(self):
        cmd = Candle_Molecular_Datagen()
        
        print('[DataReader] Loading data into memory, this should take 4 minutes')
        for files, xt_all, yt_all in cmd.datagen():
            self.data = xt_all

            # TODO support multiple files
            break
        print('[DataReader] Finished loading data into memory')

        self.num_samples = self.data.shape[0]*self.data.shape[1]
        self.data_size = self.data.shape[2]
        self.num_features = self.data_size
        self.dims = [
            cmd.molecular_nbrs+1,
            12, # always 12 beads
            19]
        self.type = 'Pilot2 DataReader'
        print('num_samples: ', self.num_samples)
        print('data_size: ', self.data_size)
        print('num_features: ', self.num_features)
        print('dims: ', self.dims)
        print('data shape: ', self.data.shape)

    def handle_fetch_datum_request(self, message):
        data_id = message.fetch_datum_request.data_id # in range [0, self.num_samples-1]
        mb_idx = message.fetch_datum_request.mb_idx # ignored
        tid = message.fetch_datum_request.tid # ignored
        #print('[DataReader] fetch_datum({}, {}, {})'.format(data_id, mb_idx, tid))
        response = Response()
        
        frame_idx, particle_idx = divmod(data_id, self.data.shape[1])
        
        response.fetch_datum_response.datum.extend(self.data[frame_idx, particle_idx, :])

        self._write(response)

def usage():
    'Print the proper usage of this program'
    print('[DataReader] usage: python data_reader.py input_file output_file')

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]

    if len(args) == 0:
        usage()
        exit(-1)

    dr = Pilot2DataReader(args[0], args[1])
    dr.run()