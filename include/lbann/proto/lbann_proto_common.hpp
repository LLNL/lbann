#ifndef LBANN_PROTO__INCLUDED
#define LBANN_PROTO__INCLUDED

#include "lbann/data_readers/lbann_data_reader.hpp"
#include "lbann/lbann.hpp"
#include "lbann/proto/lbann.pb.h"

/// returns mini_batch_size
int init_data_readers(bool master, const lbann_data::LbannPB &p, std::map<execution_mode, lbann::DataReader*> &data_readers, int &mini_batch_size);

void readPrototextFile(string fn, lbann_data::LbannPB &pb);
void writePrototextFile(string fn, lbann_data::LbannPB &pb);


#endif
