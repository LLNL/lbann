////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC. 
// Produced at the Lawrence Livermore National Laboratory. 
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN. 
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
//
// lbann_dnn_imagenet.cpp - DNN application for image-net classification
////////////////////////////////////////////////////////////////////////////////

#include "lbann/lbann.hpp"
#include "lbann/regularization/lbann_dropout.hpp"
#include "lbann/data_readers/lbann_image_utils.hpp"

#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif
#include <iomanip>
#include <string>

#include <fstream>
#include <limits.h>

using namespace std;
using namespace lbann;
using namespace El;


// train/test data info
const string g_ImageNet_RootDir = "/p/lscratchf/brainusr/datasets/ILSVRC2012/";
const string g_ImageNet_TrainDir = "resized_256x256/train/";
const string g_ImageNet_LabelDir = "labels/";
const string g_ImageNet_TrainLabelFile = "train_c0-9.txt";

int main(int argc, char* argv[])
{
    // El initialization (similar to MPI_Init)
    Initialize(argc, argv);
    lbann_comm *comm = NULL;

    stringstream err;


/*
trainParams.DatasetRootDir + g_ImageNet_TrainDir,
trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile,
trainParams.PercentageTrainingSamples
*/

    try {
        //sanity check to ensure we're running with a single processor
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size); //couldn't find a call for this in comm??
        if (comm->am_world_master()) {
          if (size != 1) {
            err << "This driver may only be invoked with a single MPI process; you appear to be running with " << size << " processes";
            throw lbann_exception(err.str());
          }
        }

        //get input directory and filenames
        const string root_dir = Input("--root-dir", "Root dir for training data", g_ImageNet_RootDir);
        const string label_dir = Input("--label-dir", "dir for training labels", g_ImageNet_LabelDir);
        const string train_dir = Input("--train-dir", "dir for training data", g_ImageNet_TrainDir);
        const string label_file = Input("--label-file", "label filename", g_ImageNet_TrainLabelFile);
        const string output_file = Input("--output-file", "output filename", g_ImageNet_RootDir + g_ImageNet_TrainLabelFile + ".mean_data");
        size_t num_to_process = Input("--max", "max number of images to load; 0 will load all available", 0);
        ProcessInput();
        PrintInputReport();

        cout << endl << "num to process: " << num_to_process << endl;

        DataReader_ImageNet imagenet_trainset(0);
        cout << "calling load ...\n";
        imagenet_trainset.load(root_dir + train_dir,
                               root_dir + label_dir + label_file);
        std::vector<vector<unsigned char> > data;
        cout << "calling fetch_data ...\n";
        imagenet_trainset.fetch_data(data, num_to_process);

        cout << endl << "number of training samples: " << data.size() << endl
             << "num pixels, each sample: " << data[0].size() << endl;

        int num_pixels = imagenet_trainset.get_image_width()
                         * imagenet_trainset.get_image_height()
                         * imagenet_trainset.get_image_depth();

        cout << "num_pixels: " << num_pixels << endl;

        vector<unsigned int> d(num_pixels, 0);
        for (size_t i=0; i<data.size(); i++) {
          for (size_t j=0; j<num_pixels; j++) {
            d[j] += (int)data[i][j];
          }
        }

        size_t n = data.size();
        int max = 0;
        for (size_t i=0; i<d.size(); i++) {
          d[i] /= n;
          max = d[i] > max ? d[i] : max;
        }

        if (max > 255) {
          err << __FILE__ << " max value in mean_data is > 255: " << max;
          throw lbann_exception(err.str());
        }

        ofstream out(output_file.c_str());
        if (not out.good()) {
          err << __FILE__ << " failed to open file for output: " << output_file;
          throw lbann_exception(err.str());
        }

        //binary format would be more efficient -- but the output file
        //is small enough that it's worth writing in human-readable form
        for (size_t i=0; i<d.size(); i++) {
          out << d[i] << endl;
        }
        out.close();

        cout << "\noutput written to: " << output_file 
             << endl << "this file is human-readable, and contains "
             << num_pixels << " entries" << endl;
    }
    catch (lbann_exception& e) { lbann_report_exception(e, comm); }
    catch (exception& e) { ReportException(e); } /// Elemental exceptions

    // free all resources by El and MPI
    Finalize();

    return 0;
}
