////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC. 
// Produced at the Lawrence Livermore National Laboratory. 
// Written by:
//         Brian Van Essen <vanessen1@llnl.gov>
//         Sam Jacobs <jacobs32@llnl.gov>
//         Hyojin Kim <kim63@llnl.gov>
//         Nikoli Dryden <dryden1@llnl.gov>
//         Tim Moon <moon13@llnl.gov>
//
// LLNL-CODE-XXXXXX.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network Toolkit, Version 0.9
//
// lbann_autoencoder_mnist.cpp - Autoencoder application for mnist
////////////////////////////////////////////////////////////////////////////////

#include "core/lbann_model_autoencoder.hpp"
#include "tools/mnist.h"
#include "tools/imageutil.h"

using namespace std;
using namespace lbann;
#ifdef __LIB_ELEMENTAL
using namespace El;
#endif


// layer definition
const std::vector<int> g_LayerDim = {784, 100, 10}; // layer dimension
const uint g_NumLayers = g_LayerDim.size(); // # layers

// distribution setting
const int g_BlockSize = 256; // block size

// training setting
const uint g_MBSize = 10; // mini-batch size to be trained
const uint g_EpochCount = 5; // # epochs
const float g_LearnRate = .5; // learning rate
const int  g_ActivationType = 2;
const int  g_DropOut = 0;

// training data path
//const string g_MNIST_Dir = "/g/g90/kim63/src/lbann/MNIST/";
const string g_MNIST_Dir = "/Users/kim63/LLNL/Data/MNIST/";
const string g_MNIST_TrainLabelFile = "train-labels.idx1-ubyte";
const string g_MNIST_TrainImageFile = "train-images.idx3-ubyte";
const string g_MNIST_TestLabelFile = "t10k-labels.idx1-ubyte";
const string g_MNIST_TestImageFile = "t10k-images.idx3-ubyte";



int main(int argc, char* argv[])
{
#ifdef __LIB_ELEMENTAL

    // El initialization (similar to MPI_Init)
    Initialize(argc, argv);
    
    try {
        //const Int m = Input("--height","height of matrix", 100);
        //const Int n = Input("--width","width of matrix", 100);
        //const Int g_BlockSize = Input("--nb","algorithmic blocksize", 256);
        //const bool print = Input("--print","print matrices?", false);
        
        ///////////////////////////////////////////////////////////////////
        // initalize grid, block
        ///////////////////////////////////////////////////////////////////
        ProcessInput();
        PrintInputReport();
        
        // set algorithmic blocksize
        SetBlocksize(g_BlockSize);
        
        // create a Grid: convert MPI communicators into a 2-D process grid
        Grid grid(mpi::COMM_WORLD);
        if (mpi::Rank() == 0) {
            cout << "Grid is " << grid.Height() << " x " << grid.Width() << endl;
            cout << endl;
		}


        ///////////////////////////////////////////////////////////////////
        // initalize neural network (layers)
        ///////////////////////////////////////////////////////////////////
        
        AutoEncoder autoencoder(g_LayerDim, g_LayerDim, true, g_MBSize, g_ActivationType, g_DropOut, 0.0, grid);
        if (mpi::Rank() == 0) {
            cout << "Layer initialized:" << endl;
            for (size_t n = 0; n < autoencoder.Layers.size(); n++)
                cout << "\tLayer[" << n << "]: " << autoencoder.Layers[n]->NumNeurons << endl;
            cout << endl;
        }

        if (mpi::Rank() == 0) {
	        cout << "Parameter settings:" << endl;
            cout << "\tMini-batch size: " << g_MBSize << endl;
            cout << "\tLearning rate: " << g_LearnRate << endl << endl;
            cout << "\tEpoch count: " << g_EpochCount << endl;
        }


        ///////////////////////////////////////////////////////////////////
        // load training/testing data (MNIST)
        ///////////////////////////////////////////////////////////////////
        
		CMNIST mnist;
        if (mpi::Rank() == 0) {
            if (!mnist.load((g_MNIST_Dir + g_MNIST_TrainLabelFile).c_str(), (g_MNIST_Dir + g_MNIST_TrainImageFile).c_str(), \
                            (g_MNIST_Dir + g_MNIST_TestLabelFile).c_str(), (g_MNIST_Dir + g_MNIST_TestImageFile).c_str())) {
                if (mpi::Rank() == 0)
                    cout << "MNIST error" << endl;
                return -1;
            }
            cout << "MNIST training/testing data loaded: " << mnist.getNumTrainData() << ", " << mnist.getNumTestData() << endl;
            cout << endl;
        }
        mpi::Barrier(grid.Comm());

        
        ///////////////////////////////////////////////////////////////////
        // main loop for training/testing
        ///////////////////////////////////////////////////////////////////

        // create input and output matrix residing on a single process
        CircMat Xs(g_LayerDim[0] + 1, g_MBSize, grid);
        CircMat X(g_LayerDim[0] + 1, 1, grid);
        CircMat XP(g_LayerDim[0] + 1, 1, grid);
        CircMat YP(g_LayerDim[g_NumLayers-1] + 1, 1, grid);

        // initialize indices
        vector<int> indices(mnist.getNumTrainData());
        for (int n = 0; n < mnist.getNumTrainData(); n++)
            indices[n] = n;
        std::srand(time(NULL));

        // train/test
        for (uint t = 0; t < g_EpochCount; t++) {
            if (mpi::Rank() == 0) {
                cout << "-----------------------------------------------------------" << endl;
                cout << "[" << t << "] Epoch "<< endl;
                cout << "-----------------------------------------------------------" << endl;
            }

            // training
            if (mpi::Rank() == 0)
	            std::random_shuffle(indices.begin(), indices.end());

            for (int n = 0; n < mnist.getNumTrainData(); n+= g_MBSize) {
                Zero(Xs);
                if (mpi::Rank() == 0) {
                    for (uint k = 0; k < g_MBSize; k++) {
                        int index = indices[n + k];

                        // read train data/label
                        uchar label;
                        uchar* pixels = mnist.getTrainData(index, label);

                        for (uint m = 0; m < g_LayerDim[0]; m++)
                            Xs.SetLocal(m, k, pixels[m] / 255.0);
                        Xs.SetLocal(g_LayerDim[0], k, 1);
                    }
                }
                mpi::Barrier(grid.Comm());
                
                if (mpi::Rank() == 0)
                    cout << "\rTraining: " << n;

                autoencoder.train(Xs, g_LearnRate);
            }
            if (mpi::Rank() == 0)
                cout << endl;


            // testing (reconstruction error)
            double sumerrors = 0;
            for (int n = 0; n < mnist.getNumTestData(); n++) {
                // read test data/label
                uchar label;
                if (mpi::Rank() == 0) {
                    uchar* pixels = mnist.getTestData(n, label);

                    for (uint m = 0; m < g_LayerDim[0]; m++)
                        X.SetLocal(m, 0, pixels[m] / 255.0);
                    X.SetLocal(g_LayerDim[0], 0, 1);
                }
                mpi::Barrier(grid.Comm());

                // test dnn
                autoencoder.test(X, YP, XP);

                // validate
                if (mpi::Rank() == 0) {
                    for (uint m = 0; m < g_LayerDim[0]; m++)
                        sumerrors += ((X.GetLocal(m, 0) - XP.GetLocal(m, 0)) * (X.GetLocal(m, 0) - XP.GetLocal(m, 0)));

                    cout << "\rTesting: " << n;
                }
            }
            if (mpi::Rank() == 0) {
                cout << endl;
                cout << "Sum. square errors: " << sumerrors << endl << endl;
            }
            
            
            // save a couple of reconstructed outputs as an image file (10 images)
            int imageindex[10] = {0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000};
            uchar* pixels_gt = new uchar[28 * 10 * 28];
            uchar* pixels_rc = new uchar[28 * 10 * 28];
            
            for (int n = 0; n < 10; n++) {
                uchar label;
                uchar* pixels;
                if (mpi::Rank() == 0) {
	                pixels = mnist.getTestData(imageindex[n], label);
    	            for (int y = 0; y < 28; y++)
        	            for (int x = 0; x < 28; x++)
            	            pixels_gt[y * 28 * 10 + x + 28 * n] = pixels[y * 28 + x];
                
                    for (uint m = 0; m < g_LayerDim[0]; m++)
                        X.SetLocal(m, 0, pixels[m] / 255.0);
                    X.SetLocal(g_LayerDim[0], 0, 1);
				}
                mpi::Barrier(grid.Comm());
                autoencoder.test(X, YP, XP);
                
                if (mpi::Rank() == 0) {
	                for (uint m = 0; m < g_LayerDim[0]; m++)
    	                pixels[m] = XP.GetLocal(m, 0) * 255;
                
        	        for (int y = 0; y < 28; y++)
            	        for (int x = 0; x < 28; x++)
                	        pixels_rc[y * 28 * 10 + x + 28 * n] = pixels[y * 28 + x];
                }
            }
            
            if (mpi::Rank() == 0) {
	            char imagepath_gt[512];
    	        char imagepath_rc[512];
        	    sprintf(imagepath_gt, "%slbann_autoencoder_mnist_gt.pgm", g_MNIST_Dir.c_str());
            	sprintf(imagepath_rc, "%slbann_autoencoder_mnist_%02d.pgm", g_MNIST_Dir.c_str(), t);
	            CImageUtil::savePGM(imagepath_gt, 28 * 10, 28, 1, false, pixels_gt);
    	        CImageUtil::savePGM(imagepath_rc, 28 * 10, 28, 1, false, pixels_rc);
            }
            
            delete [] pixels_gt;
            delete [] pixels_rc;

        }
    }
    catch (exception& e) { ReportException(e); }

    // free all resources by El and MPI
    Finalize();

#else

    // initalize neural network (layers)
    AutoEncoder autoencoder(g_NumLayers, g_LayerDim, g_MBSize, 0);
    cout << "Layer initialized:" << endl;
    for (size_t n = 0; n < autoencoder.Layers.size(); n++)
        cout << "\tLayer[" << n << "]: " << autoencoder.Layers[n]->NumNeurons << endl;
    cout << endl;

    cout << "Parameter settings:" << endl;
    cout << "\tMini-batch size: " << g_MBSize << endl;
    cout << "\tLearning rate: " << g_LearnRate << endl;
    cout << "\tEpoch count: " << g_EpochCount << endl;
    cout << endl;


    // load training/testing data (MNIST)
    CMNIST mnist;
    if (!mnist.load((g_MNIST_Dir + g_MNIST_TrainLabelFile).c_str(), (g_MNIST_Dir + g_MNIST_TrainImageFile).c_str(), \
                    (g_MNIST_Dir + g_MNIST_TestLabelFile).c_str(), (g_MNIST_Dir + g_MNIST_TestImageFile).c_str())) {
        cout << "MNIST error" << endl;
        return -1;
    }
    cout << "MNIST training/testing data loaded: " << mnist.getNumTrainData() << ", " << mnist.getNumTestData() << endl;
    cout << endl;


    // initialize input output matrix
    Mat Xs(g_LayerDim[0] + 1, g_MBSize);
    Mat X(g_LayerDim[0] + 1, 1);
    Mat XP(g_LayerDim[0] + 1, 1);
    Mat YP(g_LayerDim[g_NumLayers - 1] + 1, 1);


    // initialize indices
    vector<int> indices(mnist.getNumTrainData());
    for (int n = 0; n < mnist.getNumTrainData(); n++)
        indices[n] = n;


    // main loop for training/testing
    for (uint t = 0; t < g_EpochCount; t++) {
        cout << "-----------------------------------------------------------" << endl;
        cout << "[" << t << "] Epoch "<< endl;
        cout << "-----------------------------------------------------------" << endl;

        // training
        std::srand(time(NULL));
        std::random_shuffle(indices.begin(), indices.end());

        for (int n = 0; n < mnist.getNumTrainData(); n+= g_MBSize) {
            Xs.zero();
            for (uint k = 0; k < g_MBSize; k++) {
                int index = indices[n + k];

                // read train data/label
                uchar label;
                uchar* pixels = mnist.getTrainData(index, label);

                for (uint m = 0; m < g_LayerDim[0]; m++)
                    Xs[m][k] = pixels[m] / 255.0;
                Xs[g_LayerDim[0]][k] = 1;
            }

            cout << "\rTraining: " << n;
            autoencoder.train(Xs, g_LearnRate);
        }
        cout << endl;

        // testing (reconstruction error)
        double sumerrors = 0;
        for (int n = 0; n < mnist.getNumTestData(); n++) {
            // read test data/label
            uchar label;
            uchar* pixels = mnist.getTestData(n, label);

            for (uint m = 0; m < g_LayerDim[0]; m++)
                X[m][0] = pixels[m] / 255.0;
            X[g_LayerDim[0]][0] = 1;

            autoencoder.test(X, YP, XP);

            for (uint m = 0; m < g_LayerDim[0]; m++)
                sumerrors += ((X[m][0] - XP[m][0]) * (X[m][0] - XP[m][0]));

            cout << "\rTesting: " << n;
        }
        cout << endl;
        cout << "Sum. square errors: " << sumerrors << endl << endl;

        // save a couple of reconstructed outputs as an image file (10 images)
        int imageindex[10] = {0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000};
        uchar* pixels_gt = new uchar[28 * 10 * 28];
        uchar* pixels_rc = new uchar[28 * 10 * 28];

        for (int n = 0; n < 10; n++) {
            uchar label;
            uchar* pixels = mnist.getTestData(imageindex[n], label);
            for (int y = 0; y < 28; y++)
                for (int x = 0; x < 28; x++)
                    pixels_gt[y * 28 * 10 + x + 28 * n] = pixels[y * 28 + x];

            for (uint m = 0; m < g_LayerDim[0]; m++)
                X[m][0] = pixels[m] / 255.0;
            X[g_LayerDim[0]][0] = 1;

            autoencoder.test(X, YP, XP);

            for (uint m = 0; m < g_LayerDim[0]; m++)
                pixels[m] = XP[m][0] * 255;

            for (int y = 0; y < 28; y++)
                for (int x = 0; x < 28; x++)
                    pixels_rc[y * 28 * 10 + x + 28 * n] = pixels[y * 28 + x];
        }

        char imagepath_gt[512];
        char imagepath_rc[512];
        sprintf(imagepath_gt, "%slbann_autoencoder_mnist_gt.pgm", g_MNIST_Dir.c_str());
        sprintf(imagepath_rc, "%slbann_autoencoder_mnist_%02d.pgm", g_MNIST_Dir.c_str(), t);
        CImageUtil::savePGM(imagepath_gt, 28 * 10, 28, 1, false, pixels_gt);
        CImageUtil::savePGM(imagepath_rc, 28 * 10, 28, 1, false, pixels_rc);

        delete [] pixels_gt;
        delete [] pixels_rc;
    }

#endif

    return 0;
}
