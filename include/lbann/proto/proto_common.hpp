#ifndef LBANN_PROTO__INCLUDED
#define LBANN_PROTO__INCLUDED

#include "lbann/lbann.hpp"
#include <lbann.pb.h>
#include "lbann/utils/cudnn_wrapper.hpp"

/// Returns true if the Model contains at least one MotifLayer
bool has_motifs(lbann::lbann_comm *comm, const lbann_data::LbannPB& p);

void expand_motifs(lbann::lbann_comm *comm, lbann_data::LbannPB& pb);

/// instantiates one or more generic_data_readers and inserts them in &data_readers
void init_data_readers(
  bool master,
  const lbann_data::LbannPB& p,
  std::map<execution_mode, lbann::generic_data_reader *>& data_readers);

/// adjusts the number of parallel data readers
void set_num_parallel_readers(lbann::lbann_comm *comm, lbann_data::LbannPB& p);

/// adjusts the values in p by querying the options db
void get_cmdline_overrides(lbann::lbann_comm *comm, lbann_data::LbannPB& p);

/// print various params (learn_rate, etc) to cout
void print_parameters(lbann::lbann_comm *comm, lbann_data::LbannPB& p);

/// prints usage information
void print_help(lbann::lbann_comm *comm);

/// prints prototext file, cmd line, etc to file
void save_session(lbann::lbann_comm *comm, int argc, char **argv, lbann_data::LbannPB& p);

///returns a model that is on of: dnn, stacked_autoencoder, greedy_layerwise_autoencoder
lbann::model *init_model(
  lbann::lbann_comm *comm,
  lbann::optimizer *default_optimizer,
  const lbann_data::LbannPB& p);

void add_layers(
  lbann::model *model,
  std::map<execution_mode, lbann::generic_data_reader *>& data_readers,
  lbann::cudnn::cudnn_manager *cudnn,
  const lbann_data::LbannPB& p);

/// returns a optimizer factory that is one of: adagrad, rmsprop, adam, sgd
lbann::optimizer *init_default_optimizer(
  lbann::lbann_comm *comm,
  lbann::cudnn::cudnn_manager *cudnn,
  const lbann_data::LbannPB& p);

void init_callbacks(
  lbann::lbann_comm *comm,
  lbann::model *model,
  std::map<execution_mode, lbann::generic_data_reader *>& data_readers,
  const lbann_data::LbannPB& p);

///
void read_prototext_file(
  std::string fn,
  lbann_data::LbannPB& pb,
  bool master);

///
void write_prototext_file(
  std::string fn,
  lbann_data::LbannPB& pb);


#endif
