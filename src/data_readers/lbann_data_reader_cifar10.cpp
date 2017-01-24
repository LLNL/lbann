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
// lbann_data_reader_cifar10 .hpp .cpp - DataReader class for CIFAR10 dataset
////////////////////////////////////////////////////////////////////////////////

#include "lbann/data_readers/lbann_data_reader_cifar10.hpp"

using namespace std;
using namespace El;



lbann::DataReader_CIFAR10::DataReader_CIFAR10(const EGrid& grid, int batchSize)
	: DataReader(grid, batchSize)
{
    ImageWidth = 32;
    ImageHeight = 32;	
    setName("CIFAR10");
}

lbann::DataReader_CIFAR10::~DataReader_CIFAR10()
{
	
}

bool lbann::DataReader_CIFAR10::load(string FileDir, string FileName)
{
#if 0
	this->free();
	
	string trainfiles[5] = { "data_batch_1.bin", "data_batch_2.bin",
		"data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin"};
	string testfile = "test_batch.bin";
	int imgsize = 1 + ImageWidth * ImageHeight * 3;

	// read training data
	for (int n = 0; n < 5; n++) {
		string trainpath = string(DatasetDir) + __DIR_DELIMITER + trainfiles[n];
		FILE* fptrain = fopen(trainpath.c_str(), "rb");
		if (!fptrain)
			return false;
		
		for (int i = 0; i < 10000; i++) {
			unsigned char* data = new unsigned char[imgsize];
			fread(data, imgsize, 1, fptrain);
			TrainData.push_back(data);
		}

		fclose(fptrain);
	}
	
	// read testing data
	string testpath = string(DatasetDir) + __DIR_DELIMITER + testfile;
	FILE* fptest = fopen(testpath.c_str(), "rb");
	if (!fptest)
		return false;
	
	for (int i = 0; i < 10000; i++) {
		unsigned char* data = new unsigned char[imgsize];
		fread(data, imgsize, 1, fptest);
		TestData.push_back(data);
	}
	
    return true;
#endif	
}

void lbann::DataReader_CIFAR10::free()
{
	
}

bool lbann::DataReader_CIFAR10::begin(bool shuffle, int seed)
{
	
}

bool lbann::DataReader_CIFAR10::next()
{
	
}

