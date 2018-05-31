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
# data_reader.py - data reader connected over unix socket IPC
################################################################################

from external_data_reader import ExternalDataReader
from util.connection import Server
from pilot2.data import Candle_Molecular_Datagen


class Pilot2DataReader(ExternalDataReader):
    def __init__(self, connection, address):
        super(Pilot2DataReader, self).__init__(connection, address)

        data_dims = [
            11,  # cmd.molecular_nbrs + 1st bead
            12,  # always 12 beads per
            19   # 19 features
        ]
        self.create_params(flattened_candle_data,
            data_dims=data_dims, title="Pilot2")

# this could probably be automatic
class Pilot2Server(Server):
    def __init__(self):
        super(Pilot2Server, self).__init__(Pilot2DataReader)

print("[EDR] loading data")
candle_data = None
cmd = Candle_Molecular_Datagen()
for files, xt_all, yt_all in cmd.datagen():
    # TODO support multiple files
    candle_data = xt_all
    break
print("[EDR] loaded data")
flattened_candle_data = candle_data.reshape(-1, candle_data.shape[-1])

if __name__ == '__main__':
    s = Pilot2Server()
    print("[EDR] Pilot2 running")
    s.run()
