#if 1
#include "lbann/proto/lbann_proto_common.hpp"

#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"

#include "lbann/data_readers/lbann_data_reader_cnpy.hpp"
#include "lbann/data_readers/lbann_data_reader_cifar10.hpp"
#include "lbann/data_readers/lbann_data_reader_nci.hpp"
#include "lbann/data_readers/lbann_data_reader_nci_regression.hpp"
#include "lbann/data_readers/lbann_data_reader_imagenet.hpp"
#include "lbann/data_readers/lbann_data_reader_imagenet_single.hpp"
#include "lbann/data_readers/lbann_data_reader_imagenet_single_cv.hpp"
#include "lbann/data_readers/lbann_data_reader_imagenet_cv.hpp"
#include "lbann/data_readers/lbann_data_reader_mnist.hpp"
#include "lbann/data_readers/lbann_data_reader_synthetic.hpp"
#include "lbann/data_readers/lbann_opencv.hpp"

#include "lbann/optimizers/lbann_optimizer_adagrad.hpp"
#include "lbann/optimizers/lbann_optimizer_adam.hpp"
#include "lbann/optimizers/lbann_optimizer_rmsprop.hpp"
#include "lbann/optimizers/lbann_optimizer_sgd.hpp"

#include "lbann/callbacks/lbann_callback_dump_weights.hpp"
#include "lbann/callbacks/lbann_callback_dump_activations.hpp"
#include "lbann/callbacks/lbann_callback_dump_gradients.hpp"
#include "lbann/callbacks/lbann_callback_save_images.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

using namespace lbann;

pool_mode get_pool_mode(const string& s) {
  if (s == "max") {
    return pool_mode::max;
  } else if (s == "average") {
    return pool_mode::average;
  } else if (s == "average_no_pad") {
    return pool_mode::average_no_pad;
  } else {
    std::stringstream err;
    err << __FILE__ << " " <<__LINE__
        << " :: unkown pool_mode: " << s
        << " should be one of: max, average, average_no_pad";
    throw lbann_exception(err.str());
  }

}

void get_prev_neurons_and_index( lbann::sequential_model *model, int& prev_num_neurons, int& cur_index) {
  std::vector<Layer *>& layers = model->get_layers();
  prev_num_neurons = -1;
  if(layers.size() != 0) {
    Layer *prev_layer = layers.back();
    prev_num_neurons = prev_layer->get_num_neurons();
  }
  cur_index = layers.size();
}

activation_type get_activation_type(const string& s) {
  if (s == "sigmoid") {
    return activation_type::SIGMOID;
  } else if (s == "tanh") {
    return activation_type::TANH;
  } else if (s == "relu") {
    return activation_type::RELU;
  } else if (s == "id") {
    return activation_type::ID;
  } else if (s == "leaky_relu") {
    return activation_type::LEAKY_RELU;
  } else if (s == "softplus") {
    return activation_type::SOFTPLUS;
  } else if (s == "smooth_relu") {
    return activation_type::SMOOTH_RELU;
  } else if (s == "elu") {
    return activation_type::ELU;
  } else {
    std::stringstream err;
    err << __FILE__ << " " <<__LINE__
        << " :: unkown activation_type: " << s
        << " should be one of: sigmoid tanh relu id leaky_relu softplus smooth_relu elu";
    throw lbann_exception(err.str());
  }
}

weight_initialization get_weight_initialization(const string& s) {
  if (s == "zero") {
    return weight_initialization::zero;
  } else if (s == "uniform") {
    return weight_initialization::uniform;
  } else if (s == "normal") {
    return weight_initialization::normal;
  } else if (s == "glorot_normal") {
    return weight_initialization::glorot_normal;
  } else if (s == "glorot_uniform") {
    return weight_initialization::glorot_uniform;
  } else if (s == "he_normal") {
    return weight_initialization::he_normal;
  } else if (s == "he_uniform") {
    return weight_initialization::he_uniform;
  } else {
    std::stringstream err;
    err << __FILE__ << " " <<__LINE__
        << " :: unkown weight_initialization: " << s
        << " should be one of: zero uniform normal glorot_normal glorot_uniform he_normal he_uniform";
    throw lbann_exception(err.str());
  }
}

data_layout get_data_layout(const string& s, const char *file, int line) {
  if (s == "model_parallel") {
    return data_layout::MODEL_PARALLEL;
  } else if (s == "data_parallel") {
    return data_layout::DATA_PARALLEL;
  } else {
    std::stringstream err;
    err << file << " " << line
        << " :: unknown value for data_layout; should be model_parallel"
        << " or data_parallel; we got: " << s;
    throw lbann_exception(err.str());
  }
}

void init_regularizers(
  vector<regularizer *>& regs,
  lbann_comm *comm,
  const ::google::protobuf::RepeatedPtrField< ::lbann_data::Regularizer >& r) {
  for (int i=0; i<r.size(); i++) {
    const lbann_data::Regularizer r2 = r[i];
    if (r[i].has_batch_normalization()) {
      const lbann_data::BatchNormalization& b = r[i].batch_normalization();
      batch_normalization *b2 = new batch_normalization(
        get_data_layout(b.data_layout(), __FILE__, __LINE__),
        comm,
        b.decay(),
        b.gamma(),
        b.beta());
      regs.push_back(b2);
    }
    if (r[i].has_dropout()) {
      const lbann_data::Dropout& b = r[i].dropout();
      dropout *b2 = new dropout(
        get_data_layout(b.data_layout(), __FILE__, __LINE__),
        comm,
        b.keep_prob());
      regs.push_back(b2);
    }
    if (r[i].has_l2_regularization()) {
      const lbann_data::L2Regularization& b = r[i].l2_regularization();
      l2_regularization *b2 = new l2_regularization(b.lambda());
      regs.push_back(b2);
    }
  }
}

void add_layers(
  lbann::sequential_model *model,
  std::map<execution_mode, generic_data_reader *>& data_readers,
  cudnn::cudnn_manager *cudnn,
  const lbann_data::LbannPB& p) {
  std::stringstream err;
  lbann_comm *comm = model->get_comm();

  const lbann_data::Model& m = p.model();
  int mb_size = m.mini_batch_size();
  int size = m.layer_size();

  for (int j=0; j<size; j++) {
    const lbann_data::Layer& layer = m.layer(j);

    int layer_id;
    int prev_num_neurons;
    get_prev_neurons_and_index(model, prev_num_neurons, layer_id);

    //////////////////////////////////////////////////////////////////
    // LAYER: input_distributed_minibatch_parallel_io
    //////////////////////////////////////////////////////////////////
    if (layer.has_input_distributed_minibatch_parallel_io()) {
      const lbann_data::InputDistributedMiniBatchParallelIO& ell = layer.input_distributed_minibatch_parallel_io();
      vector<regularizer *> regs;
      init_regularizers(regs, comm, ell.regularizer());
      input_layer<data_layout> *d = new input_layer_distributed_minibatch_parallel_io<data_layout>(
          get_data_layout(ell.data_layout(), __FILE__, __LINE__),
          comm,
          m.num_parallel_readers(),
          mb_size,
          data_readers,
          regs);
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: fully_connected
    //////////////////////////////////////////////////////////////////
    if (layer.has_fully_connected()) {
      const lbann_data::FullyConnected& ell = layer.fully_connected();
      vector<regularizer *> regs;
      init_regularizers(regs, comm, ell.regularizer());
      Layer *new_layer = new fully_connected_layer<data_layout>(
          get_data_layout(ell.data_layout(), __FILE__, __LINE__),
          layer_id,
          prev_num_neurons,
          ell.num_neurons(),
          mb_size,
          get_activation_type(ell.activation_type()),
          get_weight_initialization(ell.weight_initialization()),
          comm,
          model->create_optimizer(),
          regs);
      model->add(new_layer);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: pooling
    //////////////////////////////////////////////////////////////////
    if (layer.has_pooling()) {
      const lbann_data::Pooling& ell = layer.pooling();

      vector<int> input_dims;
      int i;
      std::stringstream ss(ell.input_dims());
      while (ss >> i) {
        input_dims.push_back(i);
      }

      vector<int> pool_dims;
      ss.clear();
      ss.str(ell.pool_dims());
      while (ss >> i) {
        pool_dims.push_back(i);
      }

      vector<int> pool_pads;
      ss.clear();
      ss.str(ell.pool_pads());
      while (ss >> i) {
        pool_pads.push_back(i);
      }

      vector<int> pool_strides;
      ss.clear();
      ss.str(ell.pool_strides());
      while (ss >> i) {
        pool_strides.push_back(i);
      }

      pooling_layer<data_layout> *new_layer = new pooling_layer<data_layout>(
        layer_id,
        ell.num_dims(),
        ell.num_channels(),
        &input_dims[0],
        &pool_dims[0],
        &pool_pads[0],
        &pool_strides[0],
        get_pool_mode(ell.pool_mode()),
        mb_size,
        comm,
        cudnn
      );

      model->add(new_layer);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: Convolution
    //////////////////////////////////////////////////////////////////
    if (layer.has_convolution()) {
      const lbann_data::Convolution& ell = layer.convolution();

      vector<Int> input_dims;
      std::stringstream ss(ell.input_dims());
      int i;
      while (ss >> i) {
        input_dims.push_back(i);
      }

      vector<Int> filter_dims;
      ss.clear();
      ss.str(ell.filter_dims());
      while (ss >> i) {
        filter_dims.push_back(i);
      }

      vector<Int> conv_pads;
      ss.clear();
      ss.str(ell.conv_pads());
      while (ss >> i) {
        conv_pads.push_back(i);
      }

      vector<Int> conv_strides;
      ss.clear();
      ss.str(ell.conv_strides());
      while (ss >> i) {
        conv_strides.push_back(i);
      }

      Int num_dims = ell.num_dims();
      Int num_input_channels = ell.num_input_channels();
      Int num_output_channels = ell.num_output_channels();

      vector<regularizer *> regs;
      init_regularizers(regs, comm, ell.regularizer());

      convolutional_layer<data_layout> *new_layer = new convolutional_layer<data_layout>(
        layer_id,
        num_dims,
        num_input_channels,
        &input_dims[0],
        num_output_channels,
        &filter_dims[0],
        &conv_pads[0],
        &conv_strides[0],
        mb_size,
        get_activation_type(ell.activation_type()),
        get_weight_initialization(ell.weight_initialization()),
        comm,
        model->create_optimizer(),
        regs,
        cudnn
      );

      model->add(new_layer);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: softmax
    //////////////////////////////////////////////////////////////////
    if (layer.has_softmax()) {
      const lbann_data::Softmax& ell = layer.softmax();
      Layer *new_layer = new softmax_layer<data_layout>(
          get_data_layout(ell.data_layout(), __FILE__, __LINE__),
          layer_id,
          prev_num_neurons,
          ell.num_neurons(),
          mb_size,
          get_weight_initialization(ell.weight_initialization()),
          comm,
          model->create_optimizer()
        );
      model->add(new_layer);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: target_distributed_minibatch_parallel_io
    //////////////////////////////////////////////////////////////////
    if (layer.has_target_distributed_minibatch_parallel_io()) {
      const lbann_data::TargetDistributedMinibatchParallelIO& ell = layer.target_distributed_minibatch_parallel_io();
      Layer *t = new  target_layer_distributed_minibatch_parallel_io<data_layout>(
        get_data_layout(ell.data_layout(), __FILE__, __LINE__),
        comm,
        m.num_parallel_readers(),
        mb_size,
        data_readers,
        ell.shared_data_reader(),
        ell.for_regression());
      model->add(t);
    }
  }
}

void init_callbacks(
  lbann_comm *comm,
  lbann::sequential_model *model,
  std::map<execution_mode, lbann::generic_data_reader *>& data_readers,
  const lbann_data::LbannPB& p) {
  std::stringstream err;
  bool master = comm->am_world_master();

  const lbann_data::Model& m = p.model();

  cerr << endl << "STARTING init_callbacks; size: " << m.callback_size() << endl;

  //loop over the callbacks
  int size = m.callback_size();
  for (int j=0; j<size; j++) {
    const lbann_data::Callback& callback = m.callback(j);

    if (callback.has_save_images()) {
      const lbann_data::CallbackSaveImages& c = callback.save_images();
      string image_dir = c.image_dir();
      string extension = c.extension();
      generic_data_reader *reader = data_readers[execution_mode::training];
      lbann_callback_save_images *image_cb = new lbann_callback_save_images(reader, image_dir, extension);
      model->add_callback(image_cb);
    }

    if (callback.has_print()) {
      const lbann_data::CallbackPrint& c = callback.print();
      if (c.interval() > 0) {
        if (master) {
          cout << "adding print callback with interval: " << c.interval() << endl;
        }
        lbann_callback_print *print_cb = new lbann_callback_print(c.interval());
        model->add_callback(print_cb);
      }
    }

    if (callback.has_timer()) {
      const lbann_data::CallbackTimer& c = callback.timer();
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
      const lbann_data::CallbackSummary& c = callback.summary();
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
      const lbann_data::CallbackDumpWeights& c = callback.dump_weights();
      if (master) {
        cout << "adding dump weights callback with basename: " << c.basename()
             << " and interval: " << c.interval() << endl;
      }
      lbann_callback_dump_weights *weights_cb = new lbann_callback_dump_weights(c.basename(), c.interval());
      model->add_callback(weights_cb);
    }

    if (callback.has_dump_activations()) {
      const lbann_data::CallbackDumpActivations& c = callback.dump_activations();
      if (master) {
        cout << "adding dump activations callback with basename: " << c.basename()
             << " and interval: " << c.interval() << endl;
      }
      lbann_callback_dump_activations *activations_cb = new lbann_callback_dump_activations(c.basename(), c.interval());
      model->add_callback(activations_cb);
    }

    if (callback.has_dump_gradients()) {
      const lbann_data::CallbackDumpGradients& c = callback.dump_gradients();
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


sequential_model *init_model(lbann_comm *comm, optimizer_factory *optimizer_fac, const lbann_data::LbannPB& p) {
  std::stringstream err;

  sequential_model *model;

  const lbann_data::Model& m = p.model();
  const string name = m.name();
  const string objective_function = m.objective_function();
  uint mini_batch_size = m.mini_batch_size();

  //instantiate the objective function
  objective_functions::objective_fn *obj;
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
    model = new deep_neural_network(mini_batch_size, comm, obj, optimizer_fac);
  } else if (name == "stacked_autoencoder") {
    model = new stacked_autoencoder(mini_batch_size, comm, obj, optimizer_fac);
  } else if (name == "greedy_layerwise_autoencoder") {
    model = new greedy_layerwise_autoencoder(mini_batch_size, comm, obj, optimizer_fac);
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

optimizer_factory *init_optimizer_factory(lbann_comm *comm, const lbann_data::LbannPB& p) {
  const lbann_data::Model& model = p.model();
  const lbann_data::Optimizer& optimizer = model.optimizer();

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
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: unknown name for optimizer; should be one of: adagrad, rmsprop, adam, sgd\n"
        << "instead we found: " << name;
    throw lbann_exception(err.str());
  }

  return factory;
}

void init_data_readers(bool master, const lbann_data::LbannPB& p, std::map<execution_mode, generic_data_reader *>& data_readers, int mini_batch_size) {
  std::stringstream err;

  const lbann_data::DataReader & d_reader = p.data_reader();
  int size = d_reader.reader_size();

  for (int j=0; j<size; j++) {
    const lbann_data::Reader& readme = d_reader.reader(j);
    const lbann_data::ImagePreprocessor& preprocessor = readme.image_preprocessor();

    const string& name = readme.name();

    bool shuffle = readme.shuffle();

    generic_data_reader *reader = 0;
    generic_data_reader *reader_validation = 0;

    if (name == "mnist") {
      reader = new mnist_reader(mini_batch_size, shuffle);
    } else if (name == "imagenet") {
      reader = new imagenet_reader(mini_batch_size, shuffle);
      /*
      } else if (name == "imagenet_cv") {
      std::shared_ptr<cv_process> pp = std::make_shared<cv_process>();
      pp->set_normalizer(std::move(normalizer));
      pp->set_custom_transform2(std::move(colorizer));
      reader = new imagenet_reader_cv(mini_batch_size, pp, shuffle);
      } else if (name == "imagenet_single") {
      reader = new imagenet_reader_single(mini_batch_size, shuffle);
      } else if (name == "imagenet_single_cv") {
      reader = new imagenet_reader_single_cv(mini_batch_size, shuffle);
      */
    } else if (name == "nci") {
      reader = new data_reader_nci(mini_batch_size, shuffle);
    } else if (name == "nci_regression") {
      reader = new data_reader_nci_regression(mini_batch_size, shuffle);
    } else if (name == "cnpy") {
      reader = new cnpy_reader(mini_batch_size, shuffle);
    } else if (name == "cifar10") {
      reader = new cifar10_reader(mini_batch_size, shuffle);
      /*
      } else if (name == "synthetic") {
      reader = new data_reader_synthetic(mini_batch_size, shuffle);
      */
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

    if (readme.role() == "train") {
      if (name == "mnist") {
        reader_validation = new mnist_reader(mini_batch_size, shuffle);
        (*(mnist_reader *)reader_validation) = (*(mnist_reader *)reader);
      } else if (name == "imagenet") {
        reader_validation = new mnist_reader(mini_batch_size, shuffle);
        (*(imagenet_reader *)reader_validation) = (*(imagenet_reader *)reader);
      } else if (name == "nci") {
        reader_validation = new data_reader_nci(mini_batch_size, shuffle);
        (*(data_reader_nci *)reader_validation) = (*(data_reader_nci *)reader);
      } else if (name == "nci_regression") {
        reader_validation = new data_reader_nci_regression(mini_batch_size, shuffle);
        (*(data_reader_nci_regression *)reader_validation) = (*(data_reader_nci_regression *)reader);
      } else if (name == "cnpy") {
        reader_validation = new cnpy_reader(mini_batch_size, shuffle);
        (*(cnpy_reader *)reader_validation) = (*(cnpy_reader *)reader);
      } else if (name == "cifar10") {
        reader_validation = new cifar10_reader(mini_batch_size, shuffle);
        /*
        } else if (name == "synthetic") {
        reader_validation = new data_reader_synthetic(mini_batch_size, shuffle);
        */
      }
      /*
      } else if (name == "imagenet_cv") {
      std::shared_ptr<cv_process> pp = std::make_shared<cv_process>();
      pp->set_normalizer(std::move(normalizer));
      pp->set_custom_transform2(std::move(colorizer));
      reader = new imagenet_reader_cv(mini_batch_size, pp, shuffle);
      reader_validation = new imagenet_reader_cv(mini_batch_size, pp, shuffle);
      } else if (name == "imagenet_single") {
      reader_validation = new imagenet_reader_single(mini_batch_size, shuffle);
      } else if (name == "imagenet_single_cv") {
      reader_validation = new imagenet_reader_single_cv(mini_batch_size, shuffle);
      */

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
}

void readPrototextFile(string fn, lbann_data::LbannPB& pb) {
  std::stringstream err;
  int fd = open(fn.c_str(), O_RDONLY);
  if (fd == -1) {
    err <<  __FILE__ << " " << __LINE__ << " :: failed to open " << fn << " for reading";
    throw lbann_exception(err.str());
  }
  google::protobuf::io::FileInputStream *input = new google::protobuf::io::FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, &pb);
  if (not success) {
    err <<  __FILE__ << " " << __LINE__ << " :: failed to read or parse prototext file: " << fn << endl;
    throw lbann_exception(err.str());
  }
}

bool writePrototextFile(const char *fn, lbann_data::LbannPB& pb) {
  int fd = open(fn, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd == -1) {
    return false;
  }
  google::protobuf::io::FileOutputStream *output = new google::protobuf::io::FileOutputStream(fd);
  if (not google::protobuf::TextFormat::Print(pb, output)) {
    close(fd);
    delete output;
    return false;
  }
  delete output;
  close(fd);
  return true;
}
#endif
