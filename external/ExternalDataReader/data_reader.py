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

from external_data_reader import ExternalDataReader
from connection import Server

# copied originally from p2b1_mol_AE.py:Candle_Molecular_Train
# modifications:
#  * removed train_ac(), format_data()
#  * removed class members not required for datagen
#  * removed internal randomization
#  * removed debug print()s
#  * removed all customization except molecular_nbrs
#  * TODO make this able to take directory information
import helper
import numpy as np
class Candle_Molecular_Datagen():
    def __init__(self):
        self.numpylist, _ = helper.get_local_files('3k_run16')
        self.molecular_nbrs = 10

    def datagen(self):
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
            Xnorm = np.concatenate([X[:, :, :, 0:3] / 320., X[:, :, :, 3:8],
                                    X[:, :, :, 8:] / 10.], axis=3)

            num_frames = X.shape[0]
            input_feature_dim = np.prod(Xnorm.shape[2:])

            xt_all = np.array([])
            yt_all = np.array([])

            for i in range(num_frames):

                xt = Xnorm[i]
                xt = helper.get_neighborhood_features(xt, nbrs[i],
                                                      self.molecular_nbrs)

                yt = xt.copy()
                #xt = xt.reshape(xt.shape[0], 1, xt.shape[1], 1)

                if not len(xt_all):
                    xt_all = np.expand_dims(xt, axis=0)
                    yt_all = np.expand_dims(yt, axis=0)
                else:
                    xt_all = np.append(xt_all,
                                       np.expand_dims(xt, axis=0),
                                       axis=0)
                    yt_all = np.append(yt_all,
                                       np.expand_dims(yt, axis=0),
                                       axis=0)

            yield files[f_ind], xt_all, yt_all
        return

class Pilot2DataReader(ExternalDataReader):
    def __init__(self, connection, address):
        super(Pilot2DataReader, self).__init__(connection, address)

        # TODO abstract this a little more
        # eg take data, do samples = np.prod(shape[:-1]); data_size = shape[-1]
        # if there's labels xor responses set that and do the same
        # data_dims is ??
        self.params = {}

        self.params["num_samples"] = candle_data.shape[0] * candle_data.shape[1]
        self.params["data_size"] = candle_data.shape[2]
        self.params["has_labels"] = False
        self.params["num_labels"] = 0
        self.params["label_size"] = 0
        self.params["has_responses"] = False
        self.params["num_responses"] = 0
        self.params["response_size"] = 0
        self.params["data_dims"] = [
            11,  # cmd.molecular_nbrs + 1st bead
            12,  # always 12 beads per
            19   # 19 features
        ]
        self.params["reader_type"] = 'Pilot2 DataReader'

        self.params["data"] = flattened_candle_data
        self.params["labels"] = None
        self.params["responses"] = None

class Pilot2Server(Server):
    def __init__(self):
        super(Pilot2Server, self).__init__()
        self.runner = Pilot2DataReader

# print('loading data')
# candle_data = None
# cmd = Candle_Molecular_Datagen()
# for files, xt_all, yt_all in cmd.datagen():
#     # TODO support multiple files
#     candle_data = xt_all
#     break
# print('loaded data')

# messages are currently too large with 100 samples
# TODO take indices and give only what's requested
# also see if submessages will get around the size limit
# (one per sample?)
#candle_data = np.zeros((100, 3040, 2508))
#candle_data[:, :, 0] = np.ones((100, 3040))
candle_data = np.random.randn(*(1, 3040, 2508))
candle_data[:, :, 0] = np.ones((1, 3040))
flattened_candle_data = list(candle_data.flatten())

if __name__ == '__main__':
    s = Pilot2Server()
    print("[Pilot2 ExternalDataReader] running")
    s.run()
