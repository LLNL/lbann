#ifndef LBANN_PROTO__INCLUDED
#define LBANN_PROTO__INCLUDED

#include "lbann/data_readers/lbann_data_reader.hpp"
#include "lbann/lbann.hpp"
#include <lbann.pb.h>
#include "lbann/models/lbann_model_sequential.hpp"
#include "lbann/optimizers/lbann_optimizer.hpp"

/** returns mini_batch_size;
 *  instantiates one or more DataReaders and inserts them in &data_readers
 */
int init_data_readers(
  bool master, 
  const lbann_data::LbannPB &p, 
  std::map<execution_mode, lbann::DataReader*> &data_readers, 
  int &mini_batch_size);

/** returns a sequential model that is on of: dnn, stacked_autoencoder, greedy_layerwise_autoencoder
 */
lbann::sequential_model * init_model(
  lbann::lbann_comm *comm, 
  lbann::Optimizer_factory *optimizer_fac, 
  const lbann_data::LbannPB &p); 

/** returns a optimizer factory that is one of: adagrad, rmsprop, adam, sgd
 */
lbann::Optimizer_factory * init_optimizer_factory(
  lbann::lbann_comm *comm, 
  const lbann_data::LbannPB &p); 

void init_callbacks(
  lbann::lbann_comm *comm,
  lbann::sequential_model *model,
  const lbann_data::LbannPB &p);

///
void readPrototextFile(
  string fn, 
  lbann_data::LbannPB &pb);

///
void writePrototextFile(
  string fn, 
  lbann_data::LbannPB &pb);


#endif
