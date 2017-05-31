#include "lbann/proto/lbann_proto_common.hpp"

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"

#include "lbann/data_readers/lbann_data_reader_cnpy.hpp"
#include "lbann/data_readers/lbann_data_reader_nci.hpp"
#include "lbann/data_readers/lbann_data_reader_nci_regression.hpp"
#include "lbann/data_readers/lbann_data_reader_imagenet.hpp"
#include "lbann/data_readers/lbann_data_reader_mnist.hpp"

#include "lbann/optimizers/lbann_optimizer_adagrad.hpp"
#include "lbann/optimizers/lbann_optimizer_adam.hpp"
#include "lbann/optimizers/lbann_optimizer_rmsprop.hpp"
#include "lbann/optimizers/lbann_optimizer_sgd.hpp"

#include "lbann/callbacks/lbann_callback_dump_weights.hpp"
#include "lbann/callbacks/lbann_callback_dump_activations.hpp"
#include "lbann/callbacks/lbann_callback_dump_gradients.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>



using namespace lbann;

void init_callbacks(lbann_comm *comm, lbann::sequential_model *model, const lbann_data::LbannPB &p) {
  stringstream err;
  bool master = comm->am_world_master();

  const lbann_data::Model &m = p.model();

  //loop over the callbacks
  int size = m.callback_size();
  for (int j=0; j<size; j++) {
    const lbann_data::Callback &callback = m.callback(j);

    if (callback.has_print()) {
      const lbann_data::CallbackPrint &c = callback.print();
      if (c.interval() > 0) {
        if (master) {
          cout << "adding print callback with interval: " << c.interval() << endl;
        }
        lbann_callback_print *print_cb = new lbann_callback_print(c.interval());
        model->add_callback(print_cb);
      }  
    }

    if (callback.has_timer()) {
      const lbann_data::CallbackTimer &c = callback.timer();
      if (master) {
        cout << "adding timer callback with dir: " << c.dir() << endl;
      }  
      lbann_summary *summarizer = nullptr;
      if (c.dir() != "none") {
        summarizer = new lbann_summary(c.dir(), comm);
      }
      lbann_callback_timer *timer_cb = new lbann_callback_timer(summarizer);
      model->add_callback(timer_cb);
    }

    if (callback.has_summary()) {
      const lbann_data::CallbackSummary &c = callback.summary();
      if (master) {
        cout << "adding summary callback with dir: " << c.dir() << endl;
      }  
      lbann_summary *summarizer = nullptr;
      if (c.dir() != "none") {
        summarizer = new lbann_summary(c.dir(), comm);
      }
      lbann_callback_summary *summary_cb = new lbann_callback_summary(summarizer, c.interval());
      model->add_callback(summary_cb);
    }

    if (callback.has_dump_weights()) {
      const lbann_data::CallbackDumpWeights &c = callback.dump_weights();
      if (master) {
        cout << "adding dump weights callback with basename: " << c.basename() 
             << " and interval: " << c.interval() << endl;
      }  
      lbann_callback_dump_weights *weights_cb = new lbann_callback_dump_weights(c.basename(), c.interval());
      model->add_callback(weights_cb);
    }

    if (callback.has_dump_activations()) {
      const lbann_data::CallbackDumpActivations &c = callback.dump_activations();
      if (master) {
        cout << "adding dump activations callback with basename: " << c.basename() 
             << " and interval: " << c.interval() << endl;
      }  
      lbann_callback_dump_activations *activations_cb = new lbann_callback_dump_activations(c.basename(), c.interval());
      model->add_callback(activations_cb);
    }

    if (callback.has_dump_gradients()) {
      const lbann_data::CallbackDumpGradients &c = callback.dump_gradients();
      if (master) {
        cout << "adding dump gradients callback with basename: " << c.basename() 
             << " and interval: " << c.interval() << endl;
      }  
      lbann_callback_dump_gradients *gradients_cb = new lbann_callback_dump_gradients(c.basename(), c.interval());
      model->add_callback(gradients_cb);
    }

    if (callback.has_imcomm()) {
      //TODO todo
    }
  }

}


sequential_model * init_model(lbann_comm *comm, optimizer_factory *optimizer_fac, const lbann_data::LbannPB &p) {
  stringstream err;

  sequential_model * model;

  layer_factory* lfac = new layer_factory();

#if __LIB_CUDNN
  //cudnn::cudnn_manager* cudnn = new cudnn::cudnn_manager(comm, num_gpus);
  // num_gpus is a user input and '-1' by default (using all available gpus)
  cudnn::cudnn_manager* cudnn = new cudnn::cudnn_manager(comm);
#else // __LIB_CUDNN
  cudnn::cudnn_manager* cudnn = NULL;
#endif // __LIB_CUDNN

  const lbann_data::Model &m = p.model();
  const string name = m.name();
  const string objective_function = m.objective_function();
  uint mini_batch_size = m.mini_batch_size();

  //instantiate the objective function
  objective_functions::objective_fn * obj;
  if (objective_function == "categorical_cross_entropy") {
    obj = new objective_functions::categorical_cross_entropy(comm);
  } else if (objective_function == "mean_squared_error") {
    obj = new objective_functions::mean_squared_error(comm);
  } else {
    err << __FILE__ << " " << __LINE__ 
        << " :: init_model() - unknown objective function name: " << name << endl
        << "; should be one of: categorical_cross_entropy, mean_squared_error";
    throw lbann_exception(err.str());
  }

  //instantiate the network; layers will be added in a separate function call
  if (name == "dnn") {
    model = new deep_neural_network(mini_batch_size, comm, obj, lfac, optimizer_fac); 
  } else if (name == "stacked_autoencoder") {
    model = new stacked_autoencoder(mini_batch_size, comm, obj, lfac, optimizer_fac); 
  } else if (name == "greedy_layerwise_autoencoder") {
    model = new greedy_layerwise_autoencoder(mini_batch_size, comm, obj, lfac, optimizer_fac); 
  } else {
    err << __FILE__ << " " << __LINE__ 
        << " :: init_model() - unknown model name: " << name << endl
        << "; should be one of: dnn, stacked_autoencoder, greedy_layerwise_autoencoder";
    throw lbann_exception(err.str());
  }

  //add the metrics
  int size = m.metric_size();
  for (int j=0; j<size; j++) {
    string metric = m.metric(j);
    if (metric == "categorical_accuracy") {
      model->add_metric(new metrics::categorical_accuracy(data_layout::DATA_PARALLEL, comm));
    } else if (metric == "mean_squared_error") {
      model->add_metric(new metrics::mean_squared_error(data_layout::DATA_PARALLEL, comm));
    } else {
      err << __FILE__ << " " << __LINE__ 
          << " :: init_model() - unknown metric name: " << metric << endl
          << "; should be one of: categorical_accuracy, mean_squared_error";
      throw lbann_exception(err.str());
    }
  }

  //set checkpoint values
  model->set_checkpoint_dir(m.checkpoint_dir());
  model->set_checkpoint_epochs(m.checkpoint_epochs());
  model->set_checkpoint_steps(m.checkpoint_steps());
  model->set_checkpoint_secs(m.checkpoint_secs());

  return model;
}

optimizer_factory * init_optimizer_factory(lbann_comm *comm, const lbann_data::LbannPB &p) {
  const lbann_data::Model &model = p.model();
  const lbann_data::Optimizer &optimizer = model.optimizer();

  const string name = optimizer.name();
  double learn_rate = optimizer.learn_rate();
  double momentum = optimizer.momentum();
  double decay_rate = optimizer.decay();
  bool nesterov = optimizer.nesterov();

  //note: learn_rate, momentum, decay are DataType in LBANN, which is
  //      probably float. They'll be properly cast in the following

  optimizer_factory *factory;

  if (name == "adagrad") {
    factory = new adagrad_factory(comm, learn_rate);
  } else if (name == "rmsprop") {
    factory = new rmsprop_factory(comm, learn_rate);
  } else if (name == "adam") {
    factory = new adam_factory(comm, learn_rate);
  } else if (name == "sgd") {
    factory = new sgd_factory(comm, learn_rate, momentum, decay_rate, nesterov);
  } else {
    stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: unknown name for optimizer; should be one of: adagrad, rmsprop, adam, sgd\n"
        << "instead we found: " << name;
    throw lbann_exception(err.str());
  }

  return factory;
}

int init_data_readers(bool master, const lbann_data::LbannPB &p, std::map<execution_mode, DataReader*> &data_readers, int &mb_size)
{
  stringstream err;

  const lbann_data::DataReader &d_reader = p.data_reader();
  int size = d_reader.reader_size();

  int mini_batch_size = 0;
  if (mb_size != 0) {
    mini_batch_size = mb_size;
  }
  if (master) {
    cout << "mini_batch_size: " << mini_batch_size << " mb_size: " << mb_size << endl;
  }

  for (int j=0; j<size; j++) {
    const lbann_data::Reader &readme = d_reader.reader(j);
    const lbann_data::ImagePreprocessor &preprocessor = readme.image_preprocessor();

    const string &name = readme.name();

    if (mb_size == 0) {
      int this_mini_batch_size = readme.mini_batch_size();
      if (this_mini_batch_size != mini_batch_size and mini_batch_size > 0) {
        stringstream err;
        err << __FILE__ << " " << __LINE__
            << " :: mini_batch sizes don't match; one reader has "
            << this_mini_batch_size << " the other has " << mini_batch_size;
        throw lbann_exception(err.str());
      }
      mini_batch_size = this_mini_batch_size;
    }

    bool shuffle = readme.shuffle();

    DataReader *reader = 0;
    DataReader *reader_validation = 0;

    if (name == "mnist") {
      reader = new DataReader_MNIST(mini_batch_size, shuffle);
    } else if (name == "imagenet") {
      reader = new DataReader_ImageNet(mini_batch_size, shuffle);
    } else if (name == "nci") {
      reader = new data_reader_nci(mini_batch_size, shuffle);
    } else if (name == "nci_regression") {
      reader = new data_reader_nci_regression(mini_batch_size, shuffle);
    } else if (name == "cnpy") {
      reader = new DataReader_cnpy(mini_batch_size, shuffle);
    } else {
      err << __FILE__ << " " << __LINE__ << " :: unknown name for data reader: "
          << name;
      throw lbann_exception(err.str());
    }

    reader->set_data_filename( readme.data_filename() );
    if (readme.label_filename() != "") {
      reader->set_label_filename( readme.label_filename() );
    }
    reader->set_use_percent( readme.train_or_test_percent() );
    reader->set_firstN( readme.firstn() );
    if (readme.max_sample_count()) {
      reader->set_max_sample_count( readme.max_sample_count() );
    }
    if (readme.percent_of_data_to_use()) {
      reader->set_use_percent( readme.percent_of_data_to_use() );
    }
    reader->set_use_percent( readme.train_or_test_percent() );

    reader->horizontal_flip( preprocessor.horizontal_flip() );
    reader->vertical_flip( preprocessor.vertical_flip() );
    reader->rotation( preprocessor.rotation() );
    reader->horizontal_shift( preprocessor.horizontal_shift() );
    reader->vertical_shift( preprocessor.vertical_shift() );
    reader->shear_range( preprocessor.shear_range() );
    reader->subtract_mean( preprocessor.subtract_mean() );
    reader->unit_variance( preprocessor.unit_variance() );
    reader->scale( preprocessor.scale() );
    reader->z_score( preprocessor.z_score() );
    if (preprocessor.disable_augmentation()) {
      reader->disable_augmentation();
    }
    if (readme.role() == "train") {
      reader->set_role("train");
    } else if (readme.role() == "test") {
      reader->set_role("test");
    } else {
      reader->set_role("error");
    }

    if (readme.role() == "train") {
      reader->set_validation_percent( readme.validation_percent() );
      if (master) {
        cout << "Training using " << (readme.train_or_test_percent()*100)
             << "% of the training data set, which is " << reader->getNumData()
             << " samples.\n";
      }
    }

    reader->set_master(master);

    reader->load();

    if (readme.role() == "train") {
      data_readers[execution_mode::training] = reader;
    } else if (readme.role() == "test") {
      data_readers[execution_mode::testing] = reader;
    }

    //double validation_percent = readme.validation_percent();
    //TODO remove this hack!
    double validation_percent = 1.0 - readme.validation_percent();

    if (readme.role() == "train") {
      if (name == "mnist") {
        reader_validation = new DataReader_MNIST(mini_batch_size, shuffle);
        (*(DataReader_MNIST*)reader_validation) = (*(DataReader_MNIST*)reader);
      } else if (name == "imagenet") {
        reader_validation = new DataReader_MNIST(mini_batch_size, shuffle);
        (*(DataReader_ImageNet*)reader_validation) = (*(DataReader_ImageNet*)reader);
      } else if (name == "nci") {
        reader_validation = new data_reader_nci(mini_batch_size, shuffle);
        (*(data_reader_nci*)reader_validation) = (*(data_reader_nci*)reader);
      } else if (name == "nci_regression") {
        reader_validation = new data_reader_nci_regression(mini_batch_size, shuffle);
        (*(data_reader_nci_regression*)reader_validation) = (*(data_reader_nci_regression*)reader);
      } else if (name == "cnpy") {
        reader_validation = new DataReader_cnpy(mini_batch_size, shuffle);
        (*(DataReader_cnpy*)reader_validation) = (*(DataReader_cnpy*)reader);
      }
      reader_validation->set_role("validate");
      reader_validation->use_unused_index_set();

      if (master) {
          size_t num_train = reader->getNumData();
          size_t num_validate = reader_validation->getNumData();
          double validate_percent = num_validate / (num_train+num_validate)*100.0;
          double train_percent = num_train / (num_train+num_validate)*100.0;
          cout << "Training using " << train_percent << "% of the training data set, which is " << reader->getNumData() << " samples." << endl
               << "Validating training using " << validate_percent << "% of the training data set, which is " << reader_validation->getNumData() << " samples." << endl;
      }

      data_readers[execution_mode::validation] = reader_validation;
    }
  }
  if (mb_size == 0) {
    mb_size = mini_batch_size;
  }

/*
  if (master) {
    for (auto it : data_readers) {
      cerr << ">>>> leaving int_data_readers; data reader; role: " << it.second->get_role() << " num data: " << it.second->getNumData() << endl;
    }  
  }  
*/

  return mini_batch_size;
}

void readPrototextFile(string fn, lbann_data::LbannPB &pb)
{
  stringstream err;
  int fd = open(fn.c_str(), O_RDONLY);
  if (fd == -1) {
    err <<  __FILE__ << " " << __LINE__ << " :: failed to open " << fn << " for reading";
    throw lbann_exception(err.str());
  }
  google::protobuf::io::FileInputStream* input = new google::protobuf::io::FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, &pb);
  if (not success) {
    err <<  __FILE__ << " " << __LINE__ << " :: failed to read or parse prototext file: " << fn << endl;
    throw lbann_exception(err.str());
  }
}

bool writePrototextFile(const char *fn, lbann_data::LbannPB &pb)
{
  int fd = open(fn, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd == -1) {
    return false;
  }
  google::protobuf::io::FileOutputStream* output = new google::protobuf::io::FileOutputStream(fd);
  if (not google::protobuf::TextFormat::Print(pb, output)) {
    close(fd);
    delete output;
    return false;
  }
  delete output;
  close(fd);
  return true;
}
