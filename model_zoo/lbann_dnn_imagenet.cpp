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

//#include <algorithm>
//#include <random>

using namespace std;
using namespace lbann;
using namespace El;


// train/test data info
const int g_SaveImageIndex[1] = {0}; // for auto encoder
//const int g_SaveImageIndex[5] = {293, 2138, 3014, 6697, 9111}; // for auto encoder
//const int g_SaveImageIndex[5] = {1000, 2000, 3000, 4000, 5000}; // for auto encoder
const string g_ImageNet_TrainDir = "resized_256x256/train/";
const string g_ImageNet_ValDir = "resized_256x256/val/";
const string g_ImageNet_TestDir = "resized_256x256/val/"; //test/";
const string g_ImageNet_LabelDir = "labels/";
const string g_ImageNet_TrainLabelFile = "train_c0-9.txt";
const string g_ImageNet_ValLabelFile = "val.txt";
const string g_ImageNet_TestLabelFile = "val_c0-9.txt"; //"test.txt";
const uint g_ImageNet_Width = 256;
const uint g_ImageNet_Height = 256;


#if 0
void getTrainDataMB(CImageNet& ImageNet, const int* Indices, unsigned char* ImageBuf, Mat& X, Mat& Y, int MBSize, int inputDim)
{
    Zero(X);
    Zero(Y);
    for (uint k = 0; k < MBSize; k++) {
        // read train data/label
        int imagelabel;
        ImageNet.getTrainData(Indices[k], imagelabel, ImageBuf);
        //        printf("getTrainDataMB: Loading image at index Indices[%d]=%d label %d\n", k, Indices[k], imagelabel);

        for (uint n = 0; n < inputDim; n++)
            X.Set(n, k, ImageBuf[n] / 255.0);
        X.Set(inputDim, k, 1); // !!!!! Set the bias term in the last row of the input vector

        Y.Set(imagelabel, k, 1);
    }
}

void getTrainData(CImageNet& ImageNet, int Index, unsigned char* ImageBuf, CircMat& X, CircMat& Y, int inputDim)
{
    int imagelabel;
    ImageNet.getTrainData(Index, imagelabel, ImageBuf);

    //    printf("getTrainData: Loading image at index Index=%d label %d\n", Index, imagelabel);
    for (uint n = 0; n < inputDim; n++)
        X.SetLocal(n, 0, ImageBuf[n] / 255.0);
    X.SetLocal(inputDim, 0, 1); // !!!!! Set the bias term in the last row of the input vector

    Y.SetLocal(imagelabel, 0, 1);
}

void getValData(CImageNet& ImageNet, int Index, unsigned char* ImageBuf, CircMat& X, int& Label, int inputDim)
{
    int imagelabel;
    ImageNet.getValData(Index, imagelabel, ImageBuf);

    for (uint n = 0; n < inputDim; n++)
        X.SetLocal(n, 0, ImageBuf[n] / 255.0);
    X.SetLocal(inputDim, 0, 1); // !!!!!
    Label = imagelabel;

    //    printf("Image %d has label %d\n", Index, Label);
}

bool write_latest(const char* dir, const char* name, int epoch, int train)
{
   // define filename
   char filename[1024];
   sprintf(filename, "%s/%s", dir, name);

   // open the file for writing
   int fd = lbann::openwrite(filename);

   ssize_t write_rc = write(fd, &epoch, sizeof(epoch));
   if (write_rc != sizeof(epoch)) {
   }

   write_rc = write(fd, &train, sizeof(train));
   if (write_rc != sizeof(train)) {
   }

   // close our file
   lbann::closewrite(fd, filename);

   return true;
}

bool read_latest(const char* dir, const char* name, int* epochLast, int* trainLast)
{
   // assume we don't have a file
   *epochLast = -1;
   *trainLast = -1;

   // define filename
   char filename[1024];
   sprintf(filename, "%s/%s", dir, name);

   // open the file for reading
   int fd = openread(filename);

   // read epoch from file
   int epoch;
   ssize_t read_rc = read(fd, &epoch, sizeof(epoch));
   if (read_rc == sizeof(epoch)) {
       // got a value, overwrite return value
       *epochLast = epoch;
   }

   // read epoch from file
   int train;
   read_rc = read(fd, &train, sizeof(train));
   if (read_rc == sizeof(train)) {
       // got a value, overwrite return value
       *trainLast = train;
   }

   // close our file
   lbann::closeread(fd, filename);

   return true;
}

struct dnn_checkpoint {
    int epoch; // current epoch number
    int train; // current offset into list of training example indices array
    int size;  // size of training example indices array
    float learning_rate; // current learning rate
};

bool checkpointShared(int epoch, int train, vector<int>& indices, TrainingParams& trainParams,
                      deep_neural_network* dnn)
{
    // time how long this takes
    Timer timer;

    // get our rank and the number of ranks
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    // let user know we're saving a checkpoint
    uint64_t bytes_count = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        timer.Start();
        printf("Checkpoint: epoch %d, trainOffset %d ...\n", epoch, train);
        fflush(stdout);
    }

    // create top level directory
    const char* dir = trainParams.ParameterDir.c_str();
    lbann::makedir(dir);

    // create subdirectory for this epoch
    char epochdir[1024];
    sprintf(epochdir, "%s/shared.epoch.%d.train.%d", dir, epoch, train);
    lbann::makedir(epochdir);

    // rank 0 writes the training state file
    if (rank == 0) {
        // define filename for training state
        char filename[1024];
        sprintf(filename, "%s/train", epochdir);

        // open the file for writing
        int fd = lbann::openwrite(filename);

        // get size of list of training examples
        int size = indices.size();

        // checkpoint epoch number and learning rate
        dnn_checkpoint header;
        header.epoch = epoch;
        header.train = train;
        header.size  = size;
        header.learning_rate = trainParams.LearnRate;

        // write the header
        ssize_t write_rc = write(fd, &header, sizeof(header));
        if (write_rc != sizeof(header)) {
            // error!
        }
        bytes_count += write_rc;

        // write list of indices
        size_t bytes = size * sizeof(int);
        write_rc = write(fd, &indices[0], bytes);
        if (write_rc != bytes) {
            // error!
        }
        bytes_count += write_rc;

        // close our file
        lbann::closewrite(fd, filename);
    }

    // write network state
    dnn->save_to_checkpoint_shared(epochdir, &bytes_count);

    // write epoch number to current file
    if (rank == 0) {
        write_latest(dir, "shared.last", epoch, train);
    }

    // sum up bytes written across all procs
    uint64_t all_bytes_count;
    MPI_Allreduce(&bytes_count, &all_bytes_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    // stop timer and report cost
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double secs = timer.Stop();
        double bw = 0.0;
        if (secs > 0.0) {
            bw = ((double) all_bytes_count) / (secs * 1024.0 * 1024.0);
        }
        printf("Checkpoint complete: epoch %d, trainOffset %d (%f secs, %llu bytes, %f MB/sec)\n",
            epoch, train, secs, (unsigned long long) all_bytes_count, bw
        );
        fflush(stdout);
    }

    return true;
}

bool restartShared(int* epochStart,
                   int* trainStart,
                   vector<int>& indices,
                   TrainingParams& trainParams,
                   deep_neural_network* dnn)
{
    // create top level directory
    const char* dir = trainParams.ParameterDir.c_str();

    // get our rank and the number of ranks
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    // read epoch number from current file
    int epoch, train;
    if (rank == 0) {
        read_latest(dir, "shared.last", &epoch, &train);
    }
    MPI_Bcast(&epoch, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&train, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // if we couldn't find the latest epoch, just return
    if (epoch < 0) {
        return false;
    }

    // time how long this takes
    Timer timer;

    // let user know we're restarting from a checkpoint
    uint64_t bytes_count = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        timer.Start();
        printf("Restart: epoch %d, trainOffset %d ...\n", epoch, train);
        fflush(stdout);
    }

    // get subdirectory for this epoch
    char epochdir[1024];
    sprintf(epochdir, "%s/shared.epoch.%d.train.%d", dir, epoch, train);

    // rank 0 reads the training state file
    int success = 1;
    dnn_checkpoint header;
    if (rank == 0) {
        // define filename for training state
        char filename[1024];
        sprintf(filename, "%s/train", epochdir);

        // open the file for reading
        int fd = lbann::openread(filename);
        if (fd != -1) {
            // read header from checkpoint file
            ssize_t read_rc = read(fd, &header, sizeof(header));
            if (read_rc != sizeof(header)) {
                // process failed to read header
                success = 0;
            }
            bytes_count += read_rc;

            // read indices into training data
            int size = header.size;
            indices.resize(size);
            size_t bytes = size * sizeof(int);
            read_rc = read(fd, &indices[0], bytes);
            if (read_rc != bytes) {
                // process failed to read indices
                success = 0;
            }
            bytes_count += read_rc;

            // close our file
            lbann::closeread(fd, filename);
        } else {
            // failed to open the restart file
            success = 0;
        }
    }

    // broadcast whether rank 0 read training file
    MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (! success) {
        return false;
    }

    // TODO: this assumes homogeneous hardware
    // get header values from rank 0
    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

    // restore epoch number and learning rate from checkpoint file
    *epochStart           = header.epoch;
    *trainStart           = header.train;
    trainParams.LearnRate = header.learning_rate;

    // TODO: this assumes homogeneous hardware
    // get index values from rank 0
    int size = header.size;
    indices.resize(size);
    size_t bytes = size * sizeof(int);
    MPI_Bcast(&indices[0], bytes, MPI_BYTE, 0, MPI_COMM_WORLD);

    // restore model from checkpoint
    dnn->load_from_checkpoint_shared(epochdir, &bytes_count);

    // sum up bytes written across all procs
    uint64_t all_bytes_count;
    MPI_Allreduce(&bytes_count, &all_bytes_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    // let user know we've completed reading our restart
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double secs = timer.Stop();
        double bw = 0.0;
        if (secs > 0.0) {
            bw = ((double) all_bytes_count) / (secs * 1024.0 * 1024.0);
        }
        printf("Restart complete: %d, trainOffset %d (%f secs, %llu bytes, %f MB/sec)\n",
            epoch, train, secs, (unsigned long long) all_bytes_count, bw
        );
        fflush(stdout);
    }

    return true;
}

bool checkpoint(int epoch,
                int train,
                vector<int>& indices,
                TrainingParams& trainParams,
                deep_neural_network* dnn)
{
    // time how long this takes
    Timer timer;

    // get our rank and the number of ranks
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    // let user know we're saving a checkpoint
    uint64_t bytes_count = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        timer.Start();
        printf("Checkpoint: epoch %d, trainOffset %d ...\n", epoch, train);
        fflush(stdout);
    }

    // create top level directory
    const char* dir = trainParams.ParameterDir.c_str();
    lbann::makedir(dir);

    // create subdirectory for this epoch
    char epochdir[1024];
    sprintf(epochdir, "%s/epoch.%d.train.%d", dir, epoch, train);
    lbann::makedir(epochdir);

    // define filename for this rank
    char filename[1024];
    sprintf(filename, "%s/ckpt.%d", epochdir, rank);

    // open the file for writing
    int fd = lbann::openwrite(filename);

    // determine whether everyone opened their file
    int open_success = (fd != -1);
    int all_success;
    MPI_Allreduce(&open_success, &all_success, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (! all_success) {
        // someone failed to create their file
        // TODO: delete our file if we created one?
        return false;
    }

    // get size of list of training examples
    int size = indices.size();

    // checkpoint epoch number and learning rate
    dnn_checkpoint header;
    header.epoch = epoch;
    header.train = train;
    header.size  = size;
    header.learning_rate = trainParams.LearnRate;

    // track total number of bytes written
    //
    // write the header
    ssize_t write_rc = write(fd, &header, sizeof(header));
    if (write_rc != sizeof(header)) {
        // error!
    }
    bytes_count += write_rc;

    // write list of indices
    size_t bytes = size * sizeof(int);
    write_rc = write(fd, &indices[0], bytes);
    if (write_rc != bytes) {
        // error!
    }
    bytes_count += write_rc;

    // checkpoint model
    dnn->save_to_checkpoint(fd, filename, &bytes_count);

    // close our file
    lbann::closewrite(fd, filename);

    // write epoch number to current file
    if (rank == 0) {
        write_latest(dir, "last", epoch, train);
    }

    // sum up bytes written across all procs
    uint64_t all_bytes_count;
    MPI_Allreduce(&bytes_count, &all_bytes_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    // stop timer and report cost
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double secs = timer.Stop();
        double bw = 0.0;
        if (secs > 0.0) {
            bw = ((double) all_bytes_count) / (secs * 1024.0 * 1024.0);
        }
        printf("Checkpoint complete: epoch %d, trainOffset %d (%f secs, %llu bytes, %f MB/sec)\n",
            epoch, train, secs, (unsigned long long) all_bytes_count, bw
        );
        fflush(stdout);
    }

    return true;
}

bool restart(int* epochStart,
             int* trainStart,
             vector<int>& indices,
             TrainingParams& trainParams,
             deep_neural_network* dnn)
{
    // create top level directory
    const char* dir = trainParams.ParameterDir.c_str();

    // get our rank and the number of ranks
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    // read epoch number from current file
    int epoch, train;
    if (rank == 0) {
        read_latest(dir, "last", &epoch, &train);
    }
    MPI_Bcast(&epoch, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&train, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // if we couldn't find the latest epoch, just return
    if (epoch < 0) {
        return false;
    }

    // time how long this takes
    Timer timer;

    // let user know we're restarting from a checkpoint
    uint64_t bytes_count = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        timer.Start();
        printf("Restart: epoch %d, trainOffset %d ...\n", epoch, train);
        fflush(stdout);
    }

    // get subdirectory for this epoch
    char epochdir[1024];
    sprintf(epochdir, "%s/epoch.%d.train.%d", dir, epoch, train);

    // get filename for this rank
    char filename[1024];
    sprintf(filename, "%s/ckpt.%d", epochdir, rank);

    // open the file for reading
    int success = 1;
    int fd = lbann::openread(filename);
    if (fd != -1) {
        // read header from checkpoint file
        dnn_checkpoint header;
        ssize_t read_rc = read(fd, &header, sizeof(header));
        if (read_rc != sizeof(header)) {
            // process failed to read header
            success = 0;
        }
        bytes_count += read_rc;

        // restore epoch number and learning rate from checkpoint file
        *epochStart           = header.epoch;
        *trainStart           = header.train;
        trainParams.LearnRate = header.learning_rate;

        // read indices into training data
        int size = header.size;
        indices.resize(size);
        size_t bytes = size * sizeof(int);
        read_rc = read(fd, &indices[0], bytes);
        if (read_rc != bytes) {
            // process failed to read indices
            success = 0;
        }
        bytes_count += read_rc;

        // restore model from checkpoint
        dnn->load_from_checkpoint(fd, filename, &bytes_count);

        // close our file
        lbann::closeread(fd, filename);
    } else {
        // failed to open the restart file
        success = 0;
    }

    // determine whether everyone opened their file
    int all_success;
    MPI_Allreduce(&success, &all_success, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (! all_success) {
        // tried to read restart, but failed, at this point our state is ill-defined
        return false;
    }

    // sum up bytes written across all procs
    uint64_t all_bytes_count;
    MPI_Allreduce(&bytes_count, &all_bytes_count, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    // let user know we've completed reading our restart
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        double secs = timer.Stop();
        double bw = 0.0;
        if (secs > 0.0) {
            bw = ((double) all_bytes_count) / (secs * 1024.0 * 1024.0);
        }
        printf("Restart complete: %d, trainOffset %d (%f secs, %llu bytes, %f MB/sec)\n",
            epoch, train, secs, (unsigned long long) all_bytes_count, bw
        );
        fflush(stdout);
    }

    return true;
}
#endif

int main(int argc, char* argv[])
{
    // El initialization (similar to MPI_Init)
    Initialize(argc, argv);
    init_random(42);  // Deterministic initialization across every model.
    init_data_seq_random(42);
    lbann_comm *comm = NULL;

    try {
        ///////////////////////////////////////////////////////////////////
        // initalize grid, block
        ///////////////////////////////////////////////////////////////////
        TrainingParams trainParams;
        trainParams.DatasetRootDir = "/p/lscratchf/brainusr/datasets/ILSVRC2012/";
        trainParams.DropOut = 0.9;
        trainParams.ProcsPerModel = 0;
        trainParams.parse_params();
        trainParams.PercentageTrainingSamples = 0.80;
        trainParams.PercentageValidationSamples = 1.00;
        PerformanceParams perfParams;
        perfParams.parse_params();
        // Read in the user specified network topology
        NetworkParams netParams;
        netParams.parse_params();
        // Get some environment variables from the launch
        SystemParams sysParams;
        sysParams.parse_params();

        // regular dense neural network or auto encoder
        const bool g_AutoEncoder = Input("--mode", "DNN: false, AutoEncoder: true", false);

        // int inputDimension = 65536 * 3;
        // // Add in the imagenet specific part of the topology
        // std::vector<int>::iterator it;

        // it = netParams.Network.begin();
        // netParams.Network.insert(it, inputDimension);

        // training settings
        int decayIterations = 1;

        bool scale = Input("--scale", "scale data to [0,1], or [-1,1]", true);
        bool subtract_mean = Input("--subtract-mean", "subtract mean, per example", true);
        bool unit_variance = Input("--unit-variance", "standardize to unit-variance", true);

        //if set to true, above three settings have no effect
        bool z_score = Input("--z-score", "standardize to unit-variance; NA if not subtracting mean", false);

        ProcessInput();
        PrintInputReport();

        // set algorithmic blocksize
        SetBlocksize(perfParams.BlockSize);

        // create timer for performance measurement
        Timer timer_io;
        Timer timer_lbann;
        Timer timer_val;
        double sec_all_io = 0;
        double sec_all_lbann = 0;
        double sec_all_val = 0;

        // Set up the communicator and get the grid.
        comm = new lbann_comm(trainParams.ProcsPerModel);
        Grid& grid = comm->get_model_grid();
        if (comm->am_world_master()) {
          cout << "Number of models: " << comm->get_num_models() << endl;
          cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
          cout << endl;
        }

        int parallel_io = perfParams.MaxParIOSize;
        //        int io_offset = 0;
        if(parallel_io == 0) {
          if(comm->am_world_master()) {
             cout << "\tMax Parallel I/O Fetch: " << comm->get_procs_per_model() << " (Limited to # Processes)" << endl;
          }
          parallel_io = comm->get_procs_per_model();
          //          io_offset = comm->get_rank_in_model() *trainParams.MBSize;
        }else {
          if(comm->am_world_master()) {
            cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
          }
          //          parallel_io = grid.Size();
          // if(perfParams.MaxParIOSize > 1) {
          //   io_offset = comm->get_rank_in_model() *trainParams.MBSize;
          // }
        }

        parallel_io = 1;
        ///////////////////////////////////////////////////////////////////
        // load training data (ImageNet)
        ///////////////////////////////////////////////////////////////////
        DataReader_ImageNet imagenet_trainset(trainParams.MBSize, true);
        imagenet_trainset.set_file_dir(trainParams.DatasetRootDir + g_ImageNet_TrainDir);
        imagenet_trainset.set_data_filename(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile);
        imagenet_trainset.set_use_percent(trainParams.PercentageTrainingSamples);
        imagenet_trainset.load();

        if (comm->am_world_master()) {
          cout << "Training using " << (trainParams.PercentageTrainingSamples*100) << "% of the training data set, which is " << imagenet_trainset.getNumData() << " samples." << endl;
        }

        imagenet_trainset.scale(scale);
        imagenet_trainset.subtract_mean(subtract_mean);
        imagenet_trainset.unit_variance(unit_variance);
        imagenet_trainset.z_score(z_score);

        ///////////////////////////////////////////////////////////////////
        // create a validation set from the unused training data (ImageNet)
        ///////////////////////////////////////////////////////////////////
        DataReader_ImageNet imagenet_validation_set(imagenet_trainset); // Clone the training set object
        if (!imagenet_validation_set.swap_used_and_unused_index_sets()) { // Swap the used and unused index sets so that it validates on the remaining data
          if (comm->am_world_master()) {
             cerr << __FILE__ << " " << __LINE__ << " ImageNet validation data error" << endl;
          }
          return -1;
        }

        if(trainParams.PercentageValidationSamples == 1.00) {
          if (comm->am_world_master()) {
            cout << "Validating training using " << ((1.00 - trainParams.PercentageTrainingSamples)*100) << "% of the training data set, which is " << imagenet_validation_set.getNumData() << " samples." << endl;
          }
        }else {
          size_t preliminary_validation_set_size = imagenet_validation_set.getNumData();
          size_t final_validation_set_size = imagenet_validation_set.trim_data_set(trainParams.PercentageValidationSamples);
          if (comm->am_world_master()) {
            cout << "Trim the validation data set from " << preliminary_validation_set_size << " samples to " << final_validation_set_size << " samples." << endl;
          }
        }

        ///////////////////////////////////////////////////////////////////
        // load testing data (ImageNet)
        ///////////////////////////////////////////////////////////////////
        DataReader_ImageNet imagenet_testset(trainParams.MBSize, true);
        imagenet_testset.set_file_dir(trainParams.DatasetRootDir + g_ImageNet_TestDir);
        imagenet_testset.set_data_filename(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TestLabelFile);
        imagenet_testset.set_use_percent(trainParams.PercentageTestingSamples);
        imagenet_testset.load();

        if (comm->am_world_master()) {
          cout << "Testing using " << (trainParams.PercentageTestingSamples*100) << "% of the testing data set, which is " << imagenet_testset.getNumData() << " samples." << endl;
        }

        imagenet_testset.scale(scale);
        imagenet_testset.subtract_mean(subtract_mean);
        imagenet_testset.unit_variance(unit_variance);
        imagenet_testset.z_score(z_score);


        ///////////////////////////////////////////////////////////////////
        // initalize neural network (layers)
        ///////////////////////////////////////////////////////////////////
        Optimizer_factory *optimizer;
        if (trainParams.LearnRateMethod == 1) { // Adagrad
          optimizer = new Adagrad_factory(comm, trainParams.LearnRate);
        }else if (trainParams.LearnRateMethod == 2) { // RMSprop
          optimizer = new RMSprop_factory(comm/*, trainParams.LearnRate*/);
        }else {
          optimizer = new SGD_factory(comm, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
        }

        layer_factory* lfac = new layer_factory();
        deep_neural_network *dnn = NULL;
        dnn = new deep_neural_network(trainParams.MBSize, comm, new objective_functions::categorical_cross_entropy(comm), lfac, optimizer);
        metrics::categorical_accuracy acc(comm);
        dnn->add_metric(&acc);
        std::map<execution_mode, DataReader*> data_readers = {std::make_pair(execution_mode::training,&imagenet_trainset), 
                                                              std::make_pair(execution_mode::validation, &imagenet_validation_set), 
                                                              std::make_pair(execution_mode::testing, &imagenet_testset)};
        //input_layer *input_layer = new input_layer_distributed_minibatch(comm, (int) trainParams.MBSize, &imagenet_trainset, &imagenet_testset);
        input_layer *input_layer = new input_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io, (int) trainParams.MBSize, data_readers);
        dnn->add(input_layer);
        int NumLayers = netParams.Network.size();
        // initalize neural network (layers)
        for (int l = 0; l < (int)NumLayers; l++) {
          string networkType;
          if(l < (int)NumLayers-1) {
            dnn->add("FullyConnected", data_layout::MODEL_PARALLEL, netParams.Network[l],
                     trainParams.ActivationType,
                     weight_initialization::glorot_uniform,
                     {new dropout(data_layout::MODEL_PARALLEL, comm, trainParams.DropOut)});
          }else {
            // Add a softmax layer to the end
            dnn->add("Softmax", data_layout::MODEL_PARALLEL, netParams.Network[l],
                     activation_type::ID,
                     weight_initialization::glorot_uniform,
                     {});
          }
        }
        //target_layer *target_layer = new target_layer_distributed_minibatch(comm, (int) trainParams.MBSize, &imagenet_trainset, &imagenet_testset, true);
        target_layer *target_layer = new target_layer_distributed_minibatch_parallel_io(data_layout::MODEL_PARALLEL, comm, parallel_io, (int) trainParams.MBSize, data_readers, true);
        dnn->add(target_layer);

        lbann_summary summarizer(trainParams.SummaryDir, comm);
        // Print out information for each epoch.
        lbann_callback_print print_cb;
        dnn->add_callback(&print_cb);
        // Record training time information.
        lbann_callback_timer timer_cb(&summarizer);
        dnn->add_callback(&timer_cb);
        // Summarize information to Tensorboard.
        lbann_callback_summary summary_cb(&summarizer, 25);
        dnn->add_callback(&summary_cb);
        // lbann_callback_io io_cb({0});
        // dnn->add_callback(&io_cb);
        lbann_callback_adaptive_learning_rate lrsched(4, 0.1f);
        dnn->add_callback(&lrsched);

        dnn->setup();

        if (grid.Rank() == 0) {
	        cout << "Layer initialized:" << endl;
                for (uint n = 0; n < dnn->get_layers().size(); n++)
                  cout << "\tLayer[" << n << "]: " << dnn->get_layers()[n]->NumNeurons << endl;
            cout << endl;
        }

        if (grid.Rank() == 0) {
	        cout << "Parameter settings:" << endl;
            cout << "\tBlock size: " << perfParams.BlockSize << endl;
            cout << "\tEpochs: " << trainParams.EpochCount << endl;
            cout << "\tMini-batch size: " << trainParams.MBSize << endl;
            // if(trainParams.MaxMBCount == 0) {
            //   cout << "\tMini-batch count (max): " << "unlimited" << endl;
            // }else {
            //   cout << "\tMini-batch count (max): " << trainParams.MaxMBCount << endl;
            // }
            cout << "\tLearning rate: " << trainParams.LearnRate << endl;
            cout << "\tEpoch count: " << trainParams.EpochCount << endl << endl;
            if(perfParams.MaxParIOSize == 0) {
              cout << "\tMax Parallel I/O Fetch: " << grid.Size() << " (Limited to # Processes)" << endl;
            }else {
              cout << "\tMax Parallel I/O Fetch: " << perfParams.MaxParIOSize << endl;
            }
            cout << "\tDataset: " << trainParams.DatasetRootDir << endl;
        }

        // load parameters from file if available
        if (trainParams.LoadModel && trainParams.ParameterDir.length() > 0) {
          dnn->load_from_file(trainParams.ParameterDir);
        }

        ///////////////////////////////////////////////////////////////////
        // load ImageNet label list file
        ///////////////////////////////////////////////////////////////////
#if 0
        CImageNet imagenet;
        if (!imagenet.loadList(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile,
                               trainParams.DatasetRootDir + g_ImageNet_TrainDir,
                               trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_ValLabelFile,
                               trainParams.DatasetRootDir + g_ImageNet_ValDir,
                               trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TestLabelFile,
                               trainParams.DatasetRootDir + g_ImageNet_TestDir)) {
            cout << "ImageNet list file error: " << grid.Rank() << endl;
            return -1;
        }
        if (grid.Rank() == 0) {
            cout << "ImageNet training/validating/testing list loaded: ";
            cout << imagenet.getNumTrainData() << ", ";
            cout << imagenet.getNumValData()   << ", ";
            cout << imagenet.getNumTestData()  << endl;
            cout << endl;
        }
        /* Limit the number to training data samples to the size of
           the data set or the user-specified maximum */
        int numTrainData;
        if(trainParams.MaxMBCount != 0 && trainParams.MaxMBCount * trainParams.MBSize < imagenet.getNumTrainData()) {
          numTrainData = trainParams.MaxMBCount * trainParams.MBSize;
        }else {
          numTrainData = imagenet.getNumTrainData();
        }
        int numValData;
        if(trainParams.MaxValidationSamples != 0 && trainParams.MaxValidationSamples < imagenet.getNumValData()) {
          numValData = trainParams.MaxValidationSamples;
        }else {
          numValData = imagenet.getNumValData();
        }
        int numTestData;
        if(trainParams.MaxTestSamples != 0 && trainParams.MaxTestSamples < imagenet.getNumTestData()) {
          numTestData = trainParams.MaxTestSamples;
        }else {
          numTestData = imagenet.getNumTestData();
        }
        int MBCount = numTrainData / trainParams.MBSize;
        if (grid.Rank() == 0) {
          cout << "Processing " << numTrainData << " ImageNet training images in " << MBCount << " batches." << endl;
        }
#endif
        mpi::Barrier(grid.Comm());


        ///////////////////////////////////////////////////////////////////
        // main loop for training/testing
        ///////////////////////////////////////////////////////////////////

        int last_layer_size;
        last_layer_size = netParams.Network[netParams.Network.size()-1];

        //************************************************************************
        // read training state from checkpoint file if we have one
        //************************************************************************
        int epochStart = 0; // epoch number we should start at
        int trainStart; // index into indices we should start at

        //************************************************************************
        // mainloop for train/validate
        //************************************************************************
        for (uint epoch = epochStart; epoch < trainParams.EpochCount; epoch++) {
            // if (grid.Rank() == 0) {
            //     cout << "-----------------------------------------------------------" << endl;
            //     cout << "[" << epoch << "] Epoch (learning rate = " << trainParams.LearnRate << ")"<< endl;
            //     cout << "-----------------------------------------------------------" << endl;
            // }

            // if (!restarted && !g_AutoEncoder) {
            //   ((SoftmaxLayer*)dnn->get_layers()[dnn->get_layers().size()-1])->resetCost();
            // //              dnn->Softmax->resetCost();
            // }

            // TODO: need to save this in checkpoint?
            decayIterations = 1;

            //************************************************************************
            // training epoch loop
            //************************************************************************

            dnn->train(1, true);

            dnn->evaluate(execution_mode::testing);
        }

        delete dnn;
    }
    catch (lbann_exception& e) { lbann_report_exception(e, comm); }
    catch (exception& e) { ReportException(e); } /// Elemental exceptions

    // free all resources by El and MPI
    Finalize();

    return 0;
}

















#if 0
int main(int argc, char* argv[])
{
    // El initialization (similar to MPI_Init)
    Initialize(argc, argv);

    try {
        ///////////////////////////////////////////////////////////////////
        // initalize grid, block
        ///////////////////////////////////////////////////////////////////
        TrainingParams trainParams("/p/lscratchf/brainusr/datasets/ILSVRC2012/");
        PerformanceParams perfParams;
        // Read in the user specified network topology
        NetworkParams netParams;
        // Get some environment variables from the launch
        SystemParams sysParams;

        // regular dense neural network or auto encoder
        const bool g_AutoEncoder = Input("--mode", "DNN: false, AutoEncoder: true", false);

        // int inputDimension = 65536 * 3;
        // // Add in the imagenet specific part of the topology
        // std::vector<int>::iterator it;

        // it = netParams.Network.begin();
        // netParams.Network.insert(it, inputDimension);

        // training settings
        int decayIterations = 1;

        ProcessInput();
        PrintInputReport();

        // set algorithmic blocksize
        SetBlocksize(perfParams.BlockSize);

        // create a Grid: convert MPI communicators into a 2-D process grid
        Grid grid(mpi::COMM_WORLD);
        if (grid.Rank() == 0) {
            cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
            cout << endl;
        }

        // create timer for performance measurement
        Timer timer_io;
        Timer timer_lbann;
        Timer timer_val;
        double sec_all_io = 0;
        double sec_all_lbann = 0;
        double sec_all_val = 0;

        ///////////////////////////////////////////////////////////////////
        // load training data (ImageNet)
        ///////////////////////////////////////////////////////////////////
        DataReader_ImageNet imagenet_trainset(trainParams.MBSize, true, grid.Rank()*trainParams.MBSize, parallel_io*trainParams.MBSize);
        if (!imagenet_trainset.load(trainParams.DatasetRootDir, g_MNIST_TrainImageFile, g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile)) {
          if (comm->am_world_master()) {
            cout << "ImageNet train data error" << endl;
          }
          return -1;
        }

        ///////////////////////////////////////////////////////////////////
        // load testing data (ImageNet)
        ///////////////////////////////////////////////////////////////////
        DataReader_MNIST imagenet_testset(trainParams.MBSize, true, grid.Rank()*trainParams.MBSize, parallel_io*trainParams.MBSize);
        if (!imagenet_testset.load(g_MNIST_Dir, g_MNIST_TestImageFile, g_MNIST_TestLabelFile)) {
          if (comm->am_world_master()) {
            cout << "ImageNet Test data error" << endl;
          }
          return -1;
        }

        ///////////////////////////////////////////////////////////////////
        // initalize neural network (layers)
        ///////////////////////////////////////////////////////////////////
        Optimizer_factory *optimizer;
        if (trainParams.LearnRateMethod == 1) { // Adagrad
          optimizer = new Adagrad_factory(grid, trainParams.LearnRate);
        }else if (trainParams.LearnRateMethod == 2) { // RMSprop
          optimizer = new RMSprop_factory(grid/*, trainParams.LearnRate*/);
        }else {
          optimizer = new SGD_factory(grid, trainParams.LearnRate, 0.9, trainParams.LrDecayRate, true);
        }

        deep_neural_network *dnn = NULL;
        AutoEncoder *autoencoder = NULL;
        if (g_AutoEncoder) {
			// need to fix later!!!!!!!!!!!!!!!!!!!!!!!  netParams.Network should be separated into encoder and decoder parts
			//autoencoder = new AutoEncoder(netParams.Network, netParams.Network, false, trainParams.MBSize, trainParams.ActivationType, trainParams.DropOut, trainParams.Lambda, grid);
            autoencoder = new AutoEncoder(optimizer, trainParams.MBSize, grid);
          // autoencoder.add("FullyConnected", 784, g_ActivationType, g_DropOut, trainParams.Lambda);
          // autoencoder.add("FullyConnected", 100, g_ActivationType, g_DropOut, trainParams.Lambda);
          // autoencoder.add("FullyConnected", 30, g_ActivationType, g_DropOut, trainParams.Lambda);
          // autoencoder.add("Softmax", 10);
        }else {
          dnn = new deep_neural_network(optimizer, trainParams.MBSize, grid);
          int NumLayers = netParams.Network.size();
          // initalize neural network (layers)
          for (int l = 0; l < (int)NumLayers; l++) {
            string networkType;
            if(l < (int)NumLayers-1) {
              networkType = "FullyConnected";
            }else {
              // Add a softmax layer to the end
              networkType = "Softmax";
            }
            dnn->add(networkType, netParams.Network[l], trainParams.ActivationType, {new dropout(trainParams.DropOut)});
          }
        }

        if (grid.Rank() == 0) {
	        cout << "Layer initialized:" << endl;
            if (g_AutoEncoder) {
              for (size_t n = 0; n < autoencoder->get_layers().size(); n++)
                cout << "\tLayer[" << n << "]: " << autoencoder->get_layers()[n]->NumNeurons << endl;
            }
            else {
              for (uint n = 0; n < dnn->get_layers().size(); n++)
                cout << "\tLayer[" << n << "]: " << dnn->get_layers()[n]->NumNeurons << endl;
            }
            cout << endl;
        }

        if (grid.Rank() == 0) {
	        cout << "Parameter settings:" << endl;
            cout << "\tBlock size: " << perfParams.BlockSize << endl;
            cout << "\tEpochs: " << trainParams.EpochCount << endl;
            cout << "\tMini-batch size: " << trainParams.MBSize << endl;
            if(trainParams.MaxMBCount == 0) {
              cout << "\tMini-batch count (max): " << "unlimited" << endl;
            }else {
              cout << "\tMini-batch count (max): " << trainParams.MaxMBCount << endl;
            }
            cout << "\tLearning rate: " << trainParams.LearnRate << endl;
            cout << "\tEpoch count: " << trainParams.EpochCount << endl << endl;
            if(perfParams.MaxParIOSize == 0) {
              cout << "\tMax Parallel I/O Fetch: " << grid.Size() << " (Limited to # Processes)" << endl;
            }else {
              cout << "\tMax Parallel I/O Fetch: " << perfParams.MaxParIOSize << endl;
            }
            cout << "\tDataset: " << trainParams.DatasetRootDir << endl;
        }

        // load parameters from file if available
        if (trainParams.LoadModel && trainParams.ParameterDir.length() > 0) {
            if (g_AutoEncoder)
                autoencoder->load_from_file(trainParams.ParameterDir);
            else
                dnn->load_from_file(trainParams.ParameterDir);
        }

        ///////////////////////////////////////////////////////////////////
        // load ImageNet label list file
        ///////////////////////////////////////////////////////////////////
#if 0
        CImageNet imagenet;
        if (!imagenet.loadList(trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TrainLabelFile,
                               trainParams.DatasetRootDir + g_ImageNet_TrainDir,
                               trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_ValLabelFile,
                               trainParams.DatasetRootDir + g_ImageNet_ValDir,
                               trainParams.DatasetRootDir + g_ImageNet_LabelDir + g_ImageNet_TestLabelFile,
                               trainParams.DatasetRootDir + g_ImageNet_TestDir)) {
            cout << "ImageNet list file error: " << grid.Rank() << endl;
            return -1;
        }
        if (grid.Rank() == 0) {
            cout << "ImageNet training/validating/testing list loaded: ";
            cout << imagenet.getNumTrainData() << ", ";
            cout << imagenet.getNumValData()   << ", ";
            cout << imagenet.getNumTestData()  << endl;
            cout << endl;
        }
        /* Limit the number to training data samples to the size of
           the data set or the user-specified maximum */
        int numTrainData;
        if(trainParams.MaxMBCount != 0 && trainParams.MaxMBCount * trainParams.MBSize < imagenet.getNumTrainData()) {
          numTrainData = trainParams.MaxMBCount * trainParams.MBSize;
        }else {
          numTrainData = imagenet.getNumTrainData();
        }
        int numValData;
        if(trainParams.MaxValidationSamples != 0 && trainParams.MaxValidationSamples < imagenet.getNumValData()) {
          numValData = trainParams.MaxValidationSamples;
        }else {
          numValData = imagenet.getNumValData();
        }
        int numTestData;
        if(trainParams.MaxTestSamples != 0 && trainParams.MaxTestSamples < imagenet.getNumTestData()) {
          numTestData = trainParams.MaxTestSamples;
        }else {
          numTestData = imagenet.getNumTestData();
        }
        int MBCount = numTrainData / trainParams.MBSize;
        if (grid.Rank() == 0) {
          cout << "Processing " << numTrainData << " ImageNet training images in " << MBCount << " batches." << endl;
        }
#endif
        mpi::Barrier(grid.Comm());


        ///////////////////////////////////////////////////////////////////
        // main loop for training/testing
        ///////////////////////////////////////////////////////////////////

        int last_layer_size;
        if(g_AutoEncoder) {
          last_layer_size = netParams.Network[netParams.Network.size()-1]+1;
        }else {
          last_layer_size = netParams.Network[netParams.Network.size()-1];
        }

        // create a local matrix on each process for holding an input image
        Mat X_local(netParams.Network[0] + 1, trainParams.MBSize);
        Mat Y_local(last_layer_size, trainParams.MBSize);
        // create a distributed matrix on each process for input and output that stores the data on a single root node
        CircMat Xs(netParams.Network[0] + 1, trainParams.MBSize, grid);
        CircMat X(netParams.Network[0] + 1, 1, grid);
        CircMat XP(netParams.Network[0] + 1, 1, grid);

        CircMat Ys(last_layer_size, trainParams.MBSize, grid);
        CircMat Y(last_layer_size, 1, grid);
        CircMat YP(last_layer_size, 1, grid);

        vector<int> indices(numTrainData);

        // create a buffer for image data
        unsigned char* imagedata = new unsigned char[g_ImageNet_Width * g_ImageNet_Height * 3];

        //************************************************************************
        // read training state from checkpoint file if we have one
        //************************************************************************
        int epochStart; // epoch number we should start at
        int trainStart; // index into indices we should start at
        //bool restarted = restartShared(&epochStart, &trainStart, indices, trainParams, dnn);
        bool restarted = restartShared(&epochStart, &trainStart, indices, trainParams, dnn);
        if (! restarted) {
            // brand new run, set both starting values to 0
            epochStart = 0;
            trainStart = 0;

            // Note: libelemental intializes model params above with random values
            // seed the random number generator
            std::srand(trainParams.RandomSeed + 0);

            // Create a random ordering of the training set
            int tmpNumTrainData = numTrainData; //imagenet.getNumTrainData()-1;
            vector<int> trainingSet(tmpNumTrainData /*imagenet.getNumTrainData()-1*/);
            for (int n = 0; n < tmpNumTrainData/*imagenet.getNumTrainData()-1*/; n++) {
                trainingSet[n] = n;
            }
            if(trainParams.ShuffleTrainingData) {
                std::random_shuffle(trainingSet.begin(), trainingSet.end());
            }

            // select the first N from the randomly ordered training samples - initialize indices
            for (int n = 0; n < numTrainData; n++) {
                indices[n] = trainingSet[n];
            }
        }

        //************************************************************************
        // mainloop for train/validate
        //************************************************************************
        for (uint epoch = epochStart; epoch < trainParams.EpochCount; epoch++) {
            if (grid.Rank() == 0) {
                cout << "-----------------------------------------------------------" << endl;
                cout << "[" << epoch << "] Epoch (learning rate = " << trainParams.LearnRate << ")"<< endl;
                cout << "-----------------------------------------------------------" << endl;
            }

            if (!restarted && !g_AutoEncoder) {
              ((SoftmaxLayer*)dnn->get_layers()[dnn->get_layers().size()-1])->resetCost();
            //              dnn->Softmax->resetCost();
            }

            // TODO: need to save this in checkpoint?
            decayIterations = 1;

            //************************************************************************
            // training epoch loop
            //************************************************************************
            if (! restarted) {
                // randomly shuffle indices into training data at start of each epoch
                std::srand(trainParams.RandomSeed + epoch);
	        std::random_shuffle(indices.begin(), indices.end());
            }

            // Determine how much parallel I/O
            int TargetMaxIOSize = 1;
            if (perfParams.MaxParIOSize > 0) {
                TargetMaxIOSize = (grid.Size() < perfParams.MaxParIOSize) ? grid.Size() : perfParams.MaxParIOSize;
            }

            // if (grid.Rank() == 0) {
            //   cout << "\rTraining:      " << endl; //flush;
            // }

            int trainOffset = trainStart;
            while (trainOffset < numTrainData) {
                Zero(X_local);
                Zero(Y_local);

                // assume each reader can fetch a whole minibatch of training data
                int trainBlock = TargetMaxIOSize * trainParams.MBSize;
                int trainRemaining = numTrainData - trainOffset;
                if (trainRemaining < trainBlock) {
                    // not enough training data left for all readers to fetch a full batch
                    // compute number of readers needed
                    trainBlock = trainRemaining;
                }

                // How many parallel I/O streams can be fetched
                int IO_size = ceil((double)trainBlock / trainParams.MBSize);

                if (trainParams.EnableProfiling && grid.Rank() == 0) {
                  timer_io.Start();
                }

                // read training data/label mini batch
                if (grid.Rank() < IO_size) {
                  int myOffset = trainOffset + (grid.Rank() * trainParams.MBSize);
                  int numImages = std::min(trainParams.MBSize, numTrainData - myOffset);
                  getTrainDataMB(imagenet, &indices[myOffset], imagedata, X_local, Y_local, numImages, netParams.Network[0]);
                }
                mpi::Barrier(grid.Comm());

                if (grid.Rank() == 0) {
                  if (trainParams.EnableProfiling) {
                    sec_all_io += timer_io.Stop();
                    timer_lbann.Start();
                  }
                  cout << "\rTraining: " << trainOffset << endl;
                  //                  cout << "\b\b\b\b\b" << setw(5) << trainOffset << flush;
                  //                  cout << "\r" << setw(5) << trainOffset << "\t" << std::flush;
                  //                  cout << "\t" << setw(5) << trainOffset << "\t" << std::flush;
                  //                  flush(cout);
#if 0
                  {
                  float progress = 0.0;
                  while (progress < 1.0) {
                    int barWidth = 70;

                    std::cout << "[";
                    int pos = barWidth * progress;
                    for (int i = 0; i < barWidth; ++i) {
                      if (i < pos) std::cout << "=";
                      else if (i == pos) std::cout << ">";
                      else std::cout << " ";
                    }
                    std::cout << "] " << int(progress * 100.0) << " %\r";
                    std::cout.flush();

                    progress += 0.16; // for demonstration only
                  }
                  std::cout << std::endl;
                  }
#endif
                }

                // train mini batch
                for(int r = 0; r < IO_size; r++) {
                  Zero(Xs);
                  Zero(Ys);
                  Xs.SetRoot(r);
                  Ys.SetRoot(r);
                  //if (grid.Rank() == r) {
                  //  Xs.CopyFromRoot(X_local);
                  //  Ys.CopyFromRoot(Y_local);
                  //}else {
                  //  Xs.CopyFromNonRoot();
                  //  Ys.CopyFromNonRoot();
                  //}
                  //mpi::Barrier(grid.Comm());
                  if (grid.Rank() == r) {
                      CopyFromRoot(X_local, Xs);
                      CopyFromRoot(Y_local, Ys);
                  }else {
                      CopyFromNonRoot(Xs);
                      CopyFromNonRoot(Ys);
                  }


                  if (g_AutoEncoder)
                    autoencoder->train(Xs, trainParams.LearnRate);
                  else
                    dnn->train(Xs, Ys, trainParams.LearnRate, trainParams.LearnRateMethod);

#if 0
                  if(/*n*/trainOffset + r * trainParams.MBSize > decayIterations * trainParams.LrDecayCycles) {
                    trainParams.LearnRate = trainParams.LearnRate * trainParams.LrDecayRate;
                    decayIterations++;
                    if(grid.Rank() == 0) {
                      cout << "Changing the learning rate to " << trainParams.LearnRate << " after processing " << (/*n*/trainOffset + r * trainParams.MBSize) << " dataums" << endl;
                    }
                  }
#endif
                  mpi::Barrier(grid.Comm());
                }
                if (trainParams.EnableProfiling && grid.Rank() == 0) {
                  sec_all_lbann += timer_lbann.Stop();
                }
                // Finished training on single pass of data
                mpi::Barrier(grid.Comm());

                // increment our offset into the training data
                trainOffset += trainBlock;

                //************************************************************************
                // checkpoint our training state
                //************************************************************************
                // TODO: checkpoint
                bool ckpt_epoch = ((trainOffset == numTrainData) && (trainParams.Checkpoint > 0) && (epoch % trainParams.Checkpoint == 0));
                if (trainParams.SaveModel && trainParams.ParameterDir.length() > 0 && ckpt_epoch)
                {
                    checkpoint(epoch, trainOffset, indices, trainParams, dnn);
                    checkpointShared(epoch, trainOffset, indices, trainParams, dnn);
                }
            }

            // reset our training offset for the next epoch
            restarted = false;
            trainStart = 0;

            if (grid.Rank() == 0) {
                cout << " ... done" << endl;
                if (trainParams.EnableProfiling) {
                  double sec_all_total = sec_all_io + sec_all_lbann;

                  double sec_mbatch_io = sec_all_io / (MBCount * (epoch+1));
                  double sec_mbatch_lbann = sec_all_lbann / (MBCount * (epoch+1));
                  double sec_mbatch_total = (sec_all_io + sec_all_lbann) / (MBCount * (epoch+1));

                  double sec_each_io = sec_mbatch_io / trainParams.MBSize;
                  double sec_each_lbann = sec_mbatch_lbann / trainParams.MBSize;
                  double sec_each_total = (sec_mbatch_io + sec_mbatch_lbann) / trainParams.MBSize;

                  if(!g_AutoEncoder) {
                    double avg_cost = ((SoftmaxLayer*)dnn->get_layers()[dnn->get_layers().size()-1])->avgCost();
                    //                    double avg_cost = dnn->Softmax->avgCost();
                    cout << "Average Softmax Cost: " << avg_cost << endl;
                  }
                  cout << "#, Host, Nodes, Processes, Cores, TasksPerNode, Epoch, Training Samples, Mini-Batch Size, Mini-Batch Count, Total Time, Total I/O, Total lbann, MB Time, MB I/O, MB lbann, Sample Time, Sample I/O, Sample lbann" << endl;
                  cout << "# [RESULT], " << sysParams.HostName << ", " << sysParams.NumNodes << ", " << grid.Size() << ", " << sysParams.NumCores << ", " << sysParams.TasksPerNode << ", " << epoch << ", ";
                  cout << numTrainData << ", " << trainParams.MBSize << ", " << MBCount << ", ";
                  cout << sec_all_total    << ", " << sec_all_io    << ", " << sec_all_lbann    << ", ";
                  cout << sec_mbatch_total << ", " << sec_mbatch_io << ", " << sec_mbatch_lbann << ", ";
                  cout << sec_each_total   << ", " << sec_each_io   << ", " << sec_each_lbann   << endl;
#if 0
                  cout << "Training time (sec): ";
                  cout << "total: "      << sec_all_total    << " (I/O: " << sec_all_io    << ", lbann: " << sec_all_lbann    << ")" << endl;
                  cout << "mini-batch: " << sec_mbatch_total << " (I/O: " << sec_mbatch_io << ", lbann: " << sec_mbatch_lbann << ")" << endl;
                  cout << "each: "       << sec_each_total   << " (I/O: " << sec_each_io   << ", lbann: " << sec_each_lbann   << ")" << endl;
                  cout << endl;
#endif
                }
            }

#if 1
            // Update the learning rate on each epoch
            trainParams.LearnRate = trainParams.LearnRate * trainParams.LrDecayRate;
            if(grid.Rank() == 0) {
              cout << "Changing the learning rate to " << trainParams.LearnRate << " after processing " << (epoch+1) << " epochs" << endl;
            }
#endif

            //************************************************************************
            // validating/testing loop
            //************************************************************************
            int numTopOneErrors = 0, numTopFiveErrors = 0;
            double sumerrors = 0;
            if (trainParams.EnableProfiling && grid.Rank() == 0) {
              timer_val.Start();
            }
            for (int n = 0; n < numValData; n++) {

                // read validating data/label
                int imagelabel;
                if (grid.Rank() == 0) {
                  if(trainParams.TestWithTrainData) {
                    getTrainData(imagenet, indices[n], imagedata, X, Y, netParams.Network[0]);
                    for(int i = 0; i < Y.Height(); i++) {
                      if(Y.GetLocal(i,0) == 1) {
                        imagelabel = i;
                      }
                    }
                  }else {
                    getValData(imagenet, n, imagedata, X, imagelabel, netParams.Network[0]);
                  }
                }
                mpi::Barrier(grid.Comm());

                if (g_AutoEncoder) {
                    autoencoder->test(X, XP);

                    // validate
                    if (grid.Rank() == 0) {
                        for (uint m = 0; m < netParams.Network[0]; m++)
                            sumerrors += ((X.GetLocal(m, 0) - XP.GetLocal(m, 0)) * (X.GetLocal(m, 0) - XP.GetLocal(m, 0)));

                        cout << "\rTesting: " << n;
                    }
                }
                else {
                    // test dnn
                    dnn->test(X, Y);

                    // validate
                    if (grid.Rank() == 0) {
                        int labelidx[5] = {-1, -1, -1, -1, -1};
                        double labelmax[5] = {-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
                        //                        cout << endl;
                        for (int m = 0; m < netParams.Network[netParams.Network.size()-1]; m++) {
                          for(int k = 0; k < 5; k++) {
                            if (labelmax[k] <= Y.GetLocal(m, 0)) {
                                for(int i = 4; i > k; i--) {
                                  labelmax[i] = labelmax[i-1];
                                  labelidx[i] = labelidx[i-1];
                                }
                                labelmax[k] = Y.GetLocal(m, 0);
                                labelidx[k] = m;
                                break;
                            }
                          }
                        }
                        if (imagelabel != labelidx[0]) {
                            numTopOneErrors++;
                        }
                        bool topFiveMatch = false;
                        for(int i = 0; i < 5; i++) {
                          if(imagelabel == labelidx[i]) {
                            topFiveMatch = true;
                            break;
                          }
                        }
                        if(!topFiveMatch) {
                          numTopFiveErrors++;
                        }
                            // Print(Y);
#if 0
                        if(!topFiveMatch) {
                          cout << "\rTesting: " << n << "th sample, " << numTopOneErrors << " top one errors and " << numTopFiveErrors
                               << " top five errors - image label " << imagelabel << " =?= ";
                          for(int i = 0; i < 5; i++) {
                            cout << labelidx[i] << "(" << labelmax[i] << ") ";
                          }
                          cout << endl;
                          int bad_val = 0;
                          for(int i = 0; i < Y.Height(); i++) {
                            if(Y.GetLocal(i,0) < 0.00001) {
                              bad_val++;
                            }else {
                              cout << i << "=" << Y.GetLocal(i,0) << " ";
                            }
                          }
                          cout << endl;
                          cout << bad_val << " insignificant values"<< endl << endl;
                        }
#endif
                    }
                }
            }
            if (grid.Rank() == 0) {
                if (trainParams.EnableProfiling) {
                  sec_all_val += timer_val.Stop();
                }
                cout << endl;
                if (trainParams.EnableProfiling) {
                  //                  double sec_all_vall_total = sec_all_io + sec_all_lbann;

                  double sec_val_each_total = sec_all_val / (numValData * (epoch+1));

                  cout << "Validation time (sec): ";
                  cout << "total: "      << sec_all_val << endl;
                  cout << "each: "       << sec_val_each_total << endl;
                  cout << endl;
                }
            }

            if (g_AutoEncoder) {
				if (grid.Rank() == 0)
                	cout << "Sum. square errors: " << sumerrors << endl;

                // save a couple of reconstructed outputs as image files
                int imagecount = sizeof(g_SaveImageIndex) / sizeof(int);
                uchar* pixels_gt = new uchar[netParams.Network[0] * imagecount];
                uchar* pixels_rc = new uchar[netParams.Network[0] * imagecount];

                for (int n = 0; n < imagecount; n++) {
                    int imagelabel;
                    if (grid.Rank() == 0) {
                        if (1 || numValData <= 0)
                          getTrainData(imagenet, g_SaveImageIndex[n], imagedata, X, Y, netParams.Network[0]);
                        else
	                        getValData(imagenet, g_SaveImageIndex[n], imagedata, X, imagelabel, netParams.Network[0]);

                        for (int y = 0; y < g_ImageNet_Height; y++)
                            for (int x = 0; x < g_ImageNet_Width; x++)
                                for (int ch = 0; ch < 3; ch++)
                                    pixels_gt[(y * g_ImageNet_Width * imagecount + x + g_ImageNet_Width * n) * 3 + ch] = imagedata[(y * g_ImageNet_Width + x) * 3 + ch];
                    }
                    mpi::Barrier(grid.Comm());
                    autoencoder->test(X, XP);

                    if (grid.Rank() == 0) {
                        for (uint m = 0; m < netParams.Network[0]; m++)
                            imagedata[m] = XP.GetLocal(m, 0) * 255;

                        for (int y = 0; y < g_ImageNet_Height; y++)
                            for (int x = 0; x < g_ImageNet_Width; x++)
                                for (int ch = 0; ch < 3; ch++)
                                    pixels_rc[(y * g_ImageNet_Width * imagecount + x + g_ImageNet_Width * n) * 3 + ch] = imagedata[(y * g_ImageNet_Width + x) * 3 + ch];
                    }
                }

                if (grid.Rank() == 0 && trainParams.SaveImageDir.length() > 0) {
                    char imagepath_gt[512];
                    char imagepath_rc[512];
                    sprintf(imagepath_gt, "%s/lbann_autoencoder_imagenet_gt.png", trainParams.SaveImageDir.c_str());
                    sprintf(imagepath_rc, "%s/lbann_autoencoder_imagenet_%04d.png", trainParams.SaveImageDir.c_str(), epoch);
                    CImageUtil::savePNG(imagepath_gt, g_ImageNet_Width * imagecount, g_ImageNet_Height, true, pixels_gt);
                    CImageUtil::savePNG(imagepath_rc, g_ImageNet_Width * imagecount, g_ImageNet_Height, true, pixels_rc);
                }

                delete [] pixels_gt;
                delete [] pixels_rc;
            }
            else {
                float topOneAccuracy = (float)(numValData - numTopOneErrors) / numValData * 100.0f;
                float topFiveAccuracy = (float)(numValData - numTopFiveErrors) / numValData * 100.0f;
                if (grid.Rank() == 0) {
                    cout << "Top One Accuracy:  " << topOneAccuracy << "%" << endl;
                    cout << "Top Five Accuracy: " << topFiveAccuracy << "%" << endl << endl;
                }
            }

            //************************************************************************
            // checkpoint our training state
            //************************************************************************
/*
            if (trainParams.SaveModel && trainParams.ParameterDir.length() > 0 &&
                trainParams.Checkpoint > 0 && (epoch % trainParams.Checkpoint == 0))
            {
                checkpoint(epoch+1, trainParams, dnn);
            }
*/
        }
        delete [] imagedata;

        // save final model parameters
        if (trainParams.SaveModel && trainParams.ParameterDir.length() > 0) {
            if (g_AutoEncoder)
                autoencoder->save_to_file(trainParams.ParameterDir);
            else
                dnn->save_to_file(trainParams.ParameterDir);
        }

        if (g_AutoEncoder)
            delete autoencoder;
        else
            delete dnn;
    }
    catch (exception& e) { ReportException(e); }

    // free all resources by El and MPI
    Finalize();

    return 0;
}
#endif
