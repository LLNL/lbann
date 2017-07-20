#include "lbann/proto/proto_common.hpp"

#include "lbann/lbann.hpp"
#include "lbann/base.hpp"
#include "lbann/comm.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <unordered_map>

using namespace lbann;

lbann_callback_imcomm::comm_type get_comm_type(const string &s)
{
  if (s == "none") {
    return lbann_callback_imcomm::comm_type::NONE;
  } else if (s == "normal") {
    return lbann_callback_imcomm::comm_type::NORMAL;
  } else if (s == "onebit_quantization") {
    return lbann_callback_imcomm::comm_type::ONEBIT_QUANTIZATION;
  } else if (s == "thresh_quantization") {
    return lbann_callback_imcomm::comm_type::THRESH_QUANTIZATION;
  } else if (s == "adaptive_quantization") {
    return lbann_callback_imcomm::comm_type::ADAPTIVE_QUANTIZATION;
  } else {
    std::stringstream err;
    err << __FILE__ << " " <<__LINE__
        << " :: unkown comm_type: " << s
        << " should be one of: none, normal, onebit_quantization, thresh_quantization, adaptive_quantization";
    throw lbann_exception(err.str());
  }
}

pool_mode get_pool_mode(const string& s)
{
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

void get_prev_neurons_and_index( lbann::sequential_model *model, int& prev_num_neurons, int& cur_index)
{
  std::vector<Layer *>& layers = model->get_layers();
  prev_num_neurons = -1;
  if(layers.size() != 0) {
    Layer *prev_layer = layers.back();
    prev_num_neurons = prev_layer->get_num_neurons();
  }
  cur_index = layers.size();
}

weight_initialization get_weight_initialization(const string& s)
{
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

data_layout get_data_layout(const string& s, const char *file, int line)
{
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

void add_layers(
  lbann::sequential_model *model,
  std::map<execution_mode, generic_data_reader *>& data_readers,
  cudnn::cudnn_manager *cudnn,
  const lbann_data::LbannPB& p,
  std::unordered_map<uint,uint> &layer_mapping)
{
  std::stringstream err;
  lbann_comm *comm = model->get_comm();
  //bool master = comm->am_world_master();

  std::unordered_map<int, Layer*> all_layers;

  const lbann_data::Model& m = p.model();
  int mb_size = m.mini_batch_size();
  int size = m.layer_size();

  Layer *d;

  for (int j=0; j<size; j++) {
    const lbann_data::Layer& layer = m.layer(j);
    int layer_id;
    int prev_num_neurons;
    get_prev_neurons_and_index(model, prev_num_neurons, layer_id);
    data_layout dl = get_data_layout(layer.data_layout(), __FILE__, __LINE__);
    bool num_neurons_from_data_reader = layer.num_neurons_from_data_reader();

    //////////////////////////////////////////////////////////////////
    // LAYER: Relu
    //////////////////////////////////////////////////////////////////
    if (layer.has_relu()) {
      //const lbann_data::Relu &ell = layer.relu();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new relu_layer<data_layout::MODEL_PARALLEL>(layer_id, comm, cudnn);
      } else {
        d = new relu_layer<data_layout::DATA_PARALLEL>(layer_id, comm, cudnn);
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: sigmoid
    //////////////////////////////////////////////////////////////////
    if (layer.has_sigmoid()) {
      //const lbann_data::Sigmoid &ell = layer.sigmoid();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new sigmoid_layer<data_layout::MODEL_PARALLEL>(layer_id, comm);
      } else {
        d = new sigmoid_layer<data_layout::DATA_PARALLEL>(layer_id, comm);
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: reconstruction
    //////////////////////////////////////////////////////////////////
    if (layer.has_reconstruction()) {
      const lbann_data::TargetReconstruction & ell = layer.reconstruction();
      int original_layer = ell.original_layer();
      if (all_layers.find(original_layer) == all_layers.end()) {
        err << __FILE__ << " " << __LINE__ << " :: the original_field in the "
            << " Reconstruction layer has index " << original_layer
            << " but we don't have a layer with that index. Something may be "
            << " wrong in your prototext file";
        throw lbann_exception(err.str());
      }
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new reconstruction_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm,
          all_layers[original_layer]
        );
      } else {
        d = new reconstruction_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          all_layers[original_layer]
        );
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: input_distributed_minibatch
    //////////////////////////////////////////////////////////////////
    if (layer.has_input_distributed_minibatch()) {
      //const lbann_data::InputDistributedMiniBatch& ell = layer.input_distributed_minibatch();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new input_layer_distributed_minibatch<data_layout::MODEL_PARALLEL>(
          comm,
          m.num_parallel_readers(),
          data_readers);
      } else {
        d = new input_layer_distributed_minibatch<data_layout::DATA_PARALLEL>(
          comm,
          m.num_parallel_readers(),
          data_readers);
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: input_partitioned_minibatch
    //////////////////////////////////////////////////////////////////
    if (layer.has_input_partitioned_minibatch()) {
      //const lbann_data::InputPartitionedMiniBatch& ell = layer.input_partitioned_minibatch();
      if (dl == data_layout::MODEL_PARALLEL) {
        err << __FILE__ << " " << __LINE__ << " :: input_layer_partitioned_minibatch "
            << "does not support MODEL_PARALLEL layouts";
        throw lbann_exception(err.str());
      } else {
        d = new input_layer_partitioned_minibatch<data_layout::DATA_PARALLEL>(
          comm,
          m.num_parallel_readers(),
          data_readers);
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: fully_connected
    //////////////////////////////////////////////////////////////////
    if (layer.has_fully_connected()) {
      const lbann_data::FullyConnected& ell = layer.fully_connected();
      int num_neurons;
      if (num_neurons_from_data_reader) {
        num_neurons = data_readers[execution_mode::training]->get_linearized_data_size();
      } else {
        num_neurons = ell.num_neurons();
      }
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm,
          num_neurons,
          get_weight_initialization(ell.weight_initialization()),
          model->create_optimizer(),
          ell.has_bias());
      } else {
        d = new fully_connected_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          num_neurons,
          get_weight_initialization(ell.weight_initialization()),
          model->create_optimizer(),
          ell.has_bias());
      }
      double l2_regularization_factor = ell.l2_regularization_factor();
      if(l2_regularization_factor != double(0.0)) {
        ((learning *) d)->set_l2_regularization_factor(l2_regularization_factor);
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
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
      if (dl == data_layout::MODEL_PARALLEL) {
        err << __FILE__ << " " << __LINE__ << " :: local_response_normalization "
            << "does not support MODEL_PARALLEL layouts";
        throw lbann_exception(err.str());
      } else {
        d = new pooling_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          ell.num_dims(),
          &pool_dims[0],
          &pool_pads[0],
          &pool_strides[0],
          get_pool_mode(ell.pool_mode()),
          cudnn
        );
      }

      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: Convolution
    //////////////////////////////////////////////////////////////////
    if (layer.has_convolution()) {
      const lbann_data::Convolution& ell = layer.convolution();

      vector<int> conv_dims;
      std::stringstream ss;
      int i;
      ss.str(ell.conv_dims());
      while (ss >> i) {
        conv_dims.push_back(i);
      }

      vector<int> conv_pads;
      ss.clear();
      ss.str(ell.conv_pads());
      while (ss >> i) {
        conv_pads.push_back(i);
      }

      vector<int> conv_strides;
      ss.clear();
      ss.str(ell.conv_strides());
      while (ss >> i) {
        conv_strides.push_back(i);
      }

      int num_dims = ell.num_dims();
      //int num_input_channels = ell.num_input_channels();
      int num_output_channels = ell.num_output_channels();
      bool has_bias = ell.has_bias();
      if (dl == data_layout::MODEL_PARALLEL) {
        err << __FILE__ << " " << __LINE__ << " :: convolution "
            << "does not support MODEL_PARALLEL layouts";
        throw lbann_exception(err.str());
      } else {
        d = new convolution_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          num_dims,
          num_output_channels,
          &conv_dims[0],
          &conv_pads[0],
          &conv_strides[0],
          get_weight_initialization(ell.weight_initialization()),
          model->create_optimizer(),
          has_bias,
          cudnn
        );
      }

      double l2_regularization_factor = ell.l2_regularization_factor();
      if(l2_regularization_factor != double(0.0)) {
        ((learning *) d)->set_l2_regularization_factor(l2_regularization_factor);
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: local_response_normalization
    //////////////////////////////////////////////////////////////////
    if (layer.has_local_response_normalization()) {
      const lbann_data::LocalResponseNormalization& ell = layer.local_response_normalization();

      DataType lrn_alpha = ell.lrn_alpha();
      DataType lrn_beta = ell.lrn_beta();
      DataType lrn_k = ell.lrn_k();
      int window_width = ell.window_width();
      if (dl == data_layout::MODEL_PARALLEL) {
        err << __FILE__ << " " << __LINE__ << " :: local_response_normalization "
            << "does not support MODEL_PARALLEL layouts";
        throw lbann_exception(err.str());
      } else {
        d = new local_response_normalization_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          window_width,
          lrn_alpha,
          lrn_beta,
          lrn_k,
          cudnn);
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: selu_dropout (regularizer)
    //////////////////////////////////////////////////////////////////
    if (layer.has_selu_dropout()) {
      const lbann_data::SeluDropout& ell = layer.selu_dropout();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new selu_dropout<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm,
          ell.keep_prob(),
          ell.alpha(),
          ell.scale()
        );
      } else {
        d = new selu_dropout<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          ell.keep_prob(),
          ell.alpha(),
          ell.scale()
        );
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: batch_normalization
    //////////////////////////////////////////////////////////////////
    if (layer.has_batch_normalization()) {
      const lbann_data::BatchNormalization& ell = layer.batch_normalization();
      if (dl == data_layout::MODEL_PARALLEL) {
        err << __FILE__ << " " << __LINE__ << " :: batch_normalization "
            << "does not support MODEL_PARALLEL layouts";
        throw lbann_exception(err.str());
      } else {
        d = new batch_normalization<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          ell.decay(),
          ell.scale_init(),
          ell.bias_init(),
          ell.epsilon(),
          ell.use_global_stats());
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: selu (activation)
    //////////////////////////////////////////////////////////////////
    if (layer.has_selu()) {
      const lbann_data::Selu& ell = layer.selu();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new selu_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm,
          ell.alpha(),
          ell.scale()
        );
      } else {
        d = new selu_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          ell.alpha(),
          ell.scale()
        );
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: tanh
    //////////////////////////////////////////////////////////////////
    if (layer.has_tanh()) {
      //const lbann_data::Tanh& ell = layer.tanh();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new tanh_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm
        );
      } else {
        d = new tanh_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm
        );
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: softplus
    //////////////////////////////////////////////////////////////////
    if (layer.has_softplus()) {
      //const lbann_data::Softplus& ell = layer.softplus();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new softplus_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm
        );
      } else {
        d = new softplus_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm
        );
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: smooth_relu
    //////////////////////////////////////////////////////////////////
    if (layer.has_smooth_relu()) {
      //const lbann_data::SmoothRelu& ell = layer.smooth_relu();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new smooth_relu_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm
        );
      } else {
        d = new smooth_relu_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm
        );
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: leaky_relu
    //////////////////////////////////////////////////////////////////
    if (layer.has_leaky_relu()) {
      const lbann_data::LeakyRelu& ell = layer.leaky_relu();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new leaky_relu_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm,
          ell.leak()
        );
      } else {
        d = new leaky_relu_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          ell.leak()
        );
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: id
    //////////////////////////////////////////////////////////////////
    if (layer.has_id()) {
      //const lbann_data::ID& ell = layer.id();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new id_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm
        );
      } else {
        d = new id_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm
        );
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: elu
    //////////////////////////////////////////////////////////////////
    if (layer.has_elu()) {
      const lbann_data::ELU& ell = layer.elu();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new elu_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm,
          ell.alpha()
        );
      } else {
        d = new elu_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          ell.alpha()
        );
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: dropout
    //////////////////////////////////////////////////////////////////
    if (layer.has_dropout()) {
      const lbann_data::Dropout& ell = layer.dropout();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new dropout<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm,
          ell.keep_prob());
      } else {
        d = new dropout<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          ell.keep_prob());
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: softmax
    //////////////////////////////////////////////////////////////////
    if (layer.has_softmax()) {
      //const lbann_data::Softmax& ell = layer.softmax();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new softmax_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm
        );
      } else {
        d = new softmax_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm
        );
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: target_partitioned_minibatch
    //////////////////////////////////////////////////////////////////
    if (layer.has_target_partitioned_minibatch()) {
      const lbann_data::TargetPartitionedMinibatch& ell = layer.target_partitioned_minibatch();
      if (dl == data_layout::MODEL_PARALLEL) {
        err << __FILE__ << " " << __LINE__ << " :: target_layer_partitioned_minibatch "
            << "does not support MODEL_PARALLEL layouts";
        throw lbann_exception(err.str());
      } else {
        d = new  target_layer_partitioned_minibatch<data_layout::DATA_PARALLEL>(
          comm,
          m.num_parallel_readers(),
          data_readers,
          ell.shared_data_reader(),
          ell.for_regression());
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: target_distributed_minibatch
    //////////////////////////////////////////////////////////////////
    if (layer.has_target_distributed_minibatch()) {
      const lbann_data::TargetDistributedMinibatch& ell = layer.target_distributed_minibatch();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new  target_layer_distributed_minibatch<data_layout::MODEL_PARALLEL>(
          comm,
          m.num_parallel_readers(),
          data_readers,
          ell.shared_data_reader(),
          ell.for_regression());
      } else {
        d = new  target_layer_distributed_minibatch<data_layout::DATA_PARALLEL>(
          comm,
          m.num_parallel_readers(),
          data_readers,
          ell.shared_data_reader(),
          ell.for_regression());
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }
  }
}

void init_callbacks(
  lbann_comm *comm,
  lbann::sequential_model *model,
  std::map<execution_mode, lbann::generic_data_reader *>& data_readers,
  const lbann_data::LbannPB& p,
  const std::unordered_map<uint,uint> &layer_mapping)
{
  std::stringstream err;
  bool master = comm->am_world_master();

  const lbann_data::Model& m = p.model();
  lbann_summary *summarizer = nullptr;

  if (master) cerr << endl << "starting init_callbacks; size: " << m.callback_size() << endl;

  //loop over the callbacks
  int size = m.callback_size();
  for (int j=0; j<size; j++) {
    const lbann_data::Callback& callback = m.callback(j);

    //////////////////////////////////////////////////////////////////
    // CALLBACK: save_images
    //////////////////////////////////////////////////////////////////
    if (callback.has_save_images()) {
      const lbann_data::CallbackSaveImages& c = callback.save_images();
      string image_dir = c.image_dir();
      string extension = c.extension();
      generic_data_reader *reader = data_readers[execution_mode::training];
      lbann_callback_save_images *image_cb = new lbann_callback_save_images(reader, image_dir, extension);
      model->add_callback(image_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: print
    //////////////////////////////////////////////////////////////////
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

    //////////////////////////////////////////////////////////////////
    // CALLBACK: timer
    //////////////////////////////////////////////////////////////////
    if (callback.has_timer()) {
      const lbann_data::CallbackTimer& c = callback.timer();
      if (master) {
        cout << "adding timer callback with dir: " << c.dir() << endl;
      }
      if (c.dir() != "none" && !summarizer) {
        summarizer = new lbann_summary(c.dir(), comm);
      }
      lbann_callback_timer *timer_cb = new lbann_callback_timer(summarizer);
      model->add_callback(timer_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: summary
    //////////////////////////////////////////////////////////////////
    if (callback.has_summary()) {
      const lbann_data::CallbackSummary& c = callback.summary();
      if (master) {
        cout << "adding summary callback with dir: " << c.dir() << endl;
      }
      if (c.dir() != "none" && !summarizer) {
        summarizer = new lbann_summary(c.dir(), comm);
      }
      if (!summarizer) {
        throw lbann_exception(
          std::string {} + __FILE__ + " " + std::to_string(__LINE__) + " :: " +
          "summary callback requires a valid summarizer directory");
      }
      lbann_callback_summary *summary_cb = new lbann_callback_summary(summarizer, c.interval());
      model->add_callback(summary_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: dump_weights
    //////////////////////////////////////////////////////////////////
    if (callback.has_dump_weights()) {
      const lbann_data::CallbackDumpWeights& c = callback.dump_weights();
      if (master) {
        cout << "adding dump weights callback with basename: " << c.basename()
             << " and interval: " << c.interval() << endl;
      }
      lbann_callback_dump_weights *weights_cb = new lbann_callback_dump_weights(c.basename(), c.interval());
      model->add_callback(weights_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: dump_activations
    //////////////////////////////////////////////////////////////////
    if (callback.has_dump_activations()) {
      const lbann_data::CallbackDumpActivations& c = callback.dump_activations();
      if (master) {
        cout << "adding dump activations callback with basename: " << c.basename()
             << " and interval: " << c.interval() << endl;
      }
      lbann_callback_dump_activations *activations_cb = new lbann_callback_dump_activations(c.basename(), c.interval());
      model->add_callback(activations_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: dump_gradients
    //////////////////////////////////////////////////////////////////
    if (callback.has_dump_gradients()) {
      const lbann_data::CallbackDumpGradients& c = callback.dump_gradients();
      if (master) {
        cout << "adding dump gradients callback with basename: " << c.basename()
             << " and interval: " << c.interval() << endl;
      }
      lbann_callback_dump_gradients *gradients_cb = new lbann_callback_dump_gradients(c.basename(), c.interval());
      model->add_callback(gradients_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: dump_mb_indices
    //////////////////////////////////////////////////////////////////
    if (callback.has_dump_mb_indices()) {
      const lbann_data::CallbackDumpMBIndices& c = callback.dump_mb_indices();
      if (master) {
        cout << "adding dump I/O callback with basename: " << c.basename()
             << " and interval: " << c.interval() << endl;
      }
      lbann_callback_dump_minibatch_sample_indices *mb_indices_cb = new lbann_callback_dump_minibatch_sample_indices(c.basename(), c.interval());
      model->add_callback(mb_indices_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: disp_io_stats
    //////////////////////////////////////////////////////////////////
    if (callback.has_disp_io_stats()) {
      const lbann_data::CallbackDispIOStats& c = callback.disp_io_stats();
      std::stringstream s(c.layers());
      std::unordered_set<uint> which;
      uint a;
      bool all_layers = false;
      while (s >> a) {
        if (a == 10000) {
          all_layers = true;
        } else {
          if (layer_mapping.find(a) == layer_mapping.end()) {
            err << __FILE__ << " " << __LINE__
                << " :: callback disp_io_stats: you specified the layer index " << a
                << " wrt the prototext file, but we don't have a layer with that"
                << " index; please check your prototext file";
            throw lbann_exception(err.str());
          }
          which.insert(layer_mapping.find(a)->second);
          if (master) {
            cout << "adding display I/O stats callback: index " << a << " from prototext file maps to model layer " << layer_mapping.find(a)->second << endl;
          }
        }
      }
      lbann_callback_io *io_cb = new lbann_callback_io(which);
      model->add_callback(io_cb);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: imcomm
    //////////////////////////////////////////////////////////////////
    if (callback.has_imcomm()) {
      const lbann_data::CallbackImComm& c = callback.imcomm();
      if (master) {
        cout << "adding imcomm callback\n";
      }
      std::stringstream s(c.layers());
      std::unordered_set<uint> which;
      uint a;
      bool all_layers = false;
      while (s >> a) {
        if (a == 10000) {
          all_layers = true;
        } else {
          if (layer_mapping.find(a) == layer_mapping.end()) {
            err << __FILE__ << " " << __LINE__
                << " :: callback imcomm: you specified the layer index " << a
                << " wrt the prototext file, but we don't have a layer with that"
                << " index; please check your prototext file";
            throw lbann_exception(err.str());
          }
          which.insert(layer_mapping.find(a)->second);
          if (master) {
            cout << "CALLBACK: imcomm: index " << a << " from prototext file maps to model layer " << layer_mapping.find(a)->second << endl;
          }
        }
      }
      lbann_callback_imcomm::comm_type c_type  = get_comm_type(c.intermodel_comm_method());
      lbann_callback_imcomm *im;
      if (all_layers) {
        im = new lbann_callback_imcomm(c_type);
      } else {
        im = new lbann_callback_imcomm(c_type, which);
      }
      model->add_callback(im);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: step_learning_rate
    //////////////////////////////////////////////////////////////////
    if (callback.has_step_learning_rate()) {
      const lbann_data::CallbackStepLearningRate &c = callback.step_learning_rate();
      std::stringstream s(c.layers());
      std::unordered_set<uint> which;
      uint a;
      bool all_layers = false;
      while (s >> a) {
        if (a == 10000) {
          all_layers = true;
        } else {
          if (layer_mapping.find(a) == layer_mapping.end()) {
            err << __FILE__ << " " << __LINE__
                << " :: callback step_learning_rate: you specified the layer index "
                << a << " wrt the prototext file, but we don't have a layer with that"
                << " index; please check your prototext file";
            throw lbann_exception(err.str());
          }
          which.insert(layer_mapping.find(a)->second);
        }
      }
      lbann_callback_adaptive_learning_rate *learn;
      if (all_layers) {
        learn = new lbann_callback_adaptive_learning_rate(c.step(), c.amt());
      } else {
        learn = new lbann_callback_adaptive_learning_rate(c.step(), c.amt(), which);
      }
      model->add_callback(learn);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: adaptive_learning_rate
    //////////////////////////////////////////////////////////////////
    if (callback.has_adaptive_learning_rate()) {
      const lbann_data::CallbackAdaptiveLearningRate &c = callback.adaptive_learning_rate();
      std::stringstream s(c.layers());
      std::unordered_set<uint> which;
      uint a;
      bool all_layers = false;
      while (s >> a) {
        if (a == 10000) {
          all_layers = true;
        } else {
          if (layer_mapping.find(a) == layer_mapping.end()) {
            err << __FILE__ << " " << __LINE__
                << " :: callback adaptive_learning_rate: you specified the layer index "
                << a << " wrt the prototext file, but we don't have a layer with that"
                << " index; please check your prototext file";
            throw lbann_exception(err.str());
          }
          which.insert(layer_mapping.find(a)->second);
        }
      }
      lbann_callback_adaptive_learning_rate *learn;
      if (all_layers) {
        learn = new lbann_callback_adaptive_learning_rate(c.patience(), c.amt());
      } else {
        learn = new lbann_callback_adaptive_learning_rate(c.patience(), c.amt(), which);
      }
      model->add_callback(learn);
    }

    //////////////////////////////////////////////////////////////////
    // CALLBACK: debug
    //////////////////////////////////////////////////////////////////
    if (callback.has_debug()) {
      const lbann_data::CallbackDebug& c = callback.debug();
      if (master) {
        cout << "adding debugging callback for phase: " << c.phase() << endl;
      }
      lbann_callback_debug *debug_cb = nullptr;
      if(c.phase() == "train") {
        debug_cb = new lbann_callback_debug(execution_mode::training);
      } else if (c.phase() == "validation") {
        debug_cb = new lbann_callback_debug(execution_mode::validation);
      } else if (c.phase() == "test") {
        debug_cb = new lbann_callback_debug(execution_mode::testing);
      } else {
        debug_cb = new lbann_callback_debug();
      }
      model->add_callback(debug_cb);
    }
  }

}


sequential_model *init_model(lbann_comm *comm, optimizer_factory *optimizer_fac, const lbann_data::LbannPB& p)
{
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
  } else if (name == "greedy_layerwise_autoencoder") {
    model = new greedy_layerwise_autoencoder(mini_batch_size, comm, obj, optimizer_fac);
  } else {
    err << __FILE__ << " " << __LINE__
        << " :: init_model() - unknown model name: " << name << endl
        << "; should be one of: dnn, greedy_layerwise_autoencoder";
    throw lbann_exception(err.str());
  }

  //add the metrics
  data_layout dl = get_data_layout(m.data_layout(), __FILE__, __LINE__);
  int size = m.metric_size();

  for (int j=0; j<size; j++) {
    const lbann_data::Metric &metric = m.metric(j);
    if (metric.has_categorical_accuracy()) {
      if (dl == data_layout::MODEL_PARALLEL) {
        model->add_metric(new metrics::categorical_accuracy<data_layout::MODEL_PARALLEL>(comm));
      } else {
        model->add_metric(new metrics::categorical_accuracy<data_layout::DATA_PARALLEL>(comm));
      }
    } else if (metric.has_mean_squared_error()) {
      if (dl == data_layout::MODEL_PARALLEL) {
        model->add_metric(new metrics::mean_squared_error<data_layout::MODEL_PARALLEL>(comm));
      } else {
        model->add_metric(new metrics::mean_squared_error<data_layout::DATA_PARALLEL>(comm));
      }
    } else if (metric.has_top_k_categorical_accuracy()) {
      const lbann_data::TopKCategoricalAccuracy &a = metric.top_k_categorical_accuracy();
      if (dl == data_layout::MODEL_PARALLEL) {
        model->add_metric(new metrics::top_k_categorical_accuracy<data_layout::MODEL_PARALLEL>(a.top_k(), comm));
      } else {
        model->add_metric(new metrics::top_k_categorical_accuracy<data_layout::DATA_PARALLEL>(a.top_k(), comm));
      }
    }
  }

  //set checkpoint values
  model->set_checkpoint_dir(m.checkpoint_dir());
  model->set_checkpoint_epochs(m.checkpoint_epochs());
  model->set_checkpoint_steps(m.checkpoint_steps());
  model->set_checkpoint_secs(m.checkpoint_secs());

  return model;
}

optimizer_factory *init_optimizer_factory(lbann_comm *comm, const lbann_data::LbannPB& p)
{
  bool master = comm->am_world_master();
  optimizer_factory *factory = 0;
  const lbann_data::Optimizer &opt = p.optimizer();
  if (opt.has_adagrad()) {
    const lbann_data::Adagrad &a = opt.adagrad();
    factory = new adagrad_factory(comm, a.learn_rate(), a.eps());
  } else if (opt.has_rmsprop()) {
    const lbann_data::Rmsprop &a = opt.rmsprop();
    factory = new rmsprop_factory(comm, a.learn_rate(), a.decay_rate(), a.eps());
  } else if (opt.has_adam()) {
    const lbann_data::Adam &a = opt.adam();
    factory = new adam_factory(comm, a.learn_rate(), a.beta1(), a.beta2(), a.eps());
  } else if (opt.has_hypergradient_adam()) {
    const lbann_data::HypergradientAdam &a = opt.hypergradient_adam();
    factory = new hypergradient_adam_factory(comm, a.init_learning_rate(), a.hyper_learning_rate(), a.beta1(), a.beta2(), a.eps());
  } else if (opt.has_sgd()) {
    const lbann_data::Sgd &a = opt.sgd();
    factory = new sgd_factory(comm, a.learn_rate(), a.momentum(), a.decay_rate(), a.nesterov());
  } else {
    if (master) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__
          << " :: init_optimizer_factory: prototext does not appear to contain an optimizer!";
      throw lbann_exception(err.str());
    }
  }

  return factory;
}

void init_data_readers(bool master, const lbann_data::LbannPB& p, std::map<execution_mode, generic_data_reader *>& data_readers, int mini_batch_size)
{
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
    } else if (name == "cnpy") {
      reader = new cnpy_reader(mini_batch_size, shuffle);
    } else if (name == "cifar10") {
      reader = new cifar10_reader(mini_batch_size, shuffle);
    } else if (name == "synthetic") {
      reader = new data_reader_synthetic(mini_batch_size, readme.num_samples(), readme.num_features(), shuffle);
    } else {
      err << __FILE__ << " " << __LINE__ << " :: unknown name for data reader: "
          << name;
      throw lbann_exception(err.str());
    }

    reader->set_data_filename( readme.data_filename() );
    if (readme.label_filename() != "") {
      reader->set_label_filename( readme.label_filename() );
    }
    if (readme.data_filedir() != "") {
      reader->set_file_dir( readme.data_filedir() );
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
        reader_validation = new imagenet_reader(mini_batch_size, shuffle);
        (*(imagenet_reader *)reader_validation) = (*(imagenet_reader *)reader);
      } else if (name == "nci") {
        reader_validation = new data_reader_nci(mini_batch_size, shuffle);
        (*(data_reader_nci *)reader_validation) = (*(data_reader_nci *)reader);
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
        double validate_percent = ((double) num_validate / (double) (num_train+num_validate))*100.0;
        double train_percent = ((double) num_train / (double) (num_train+num_validate))*100.0;
        cout << "Training using " << train_percent << "% of the training data set, which is " << reader->getNumData() << " samples." << endl
             << "Validating training using " << validate_percent << "% of the training data set, which is " << reader_validation->getNumData() << " samples." << endl;
      }

      data_readers[execution_mode::validation] = reader_validation;
    }
  }
}

void read_prototext_file(string fn, lbann_data::LbannPB& pb)
{
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

bool write_prototext_file(const char *fn, lbann_data::LbannPB& pb)
{
  int fd = open(fn, O_APPEND | O_CREAT | O_TRUNC, 0644);
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

void set_num_parallel_readers(lbann::lbann_comm *comm, lbann_data::LbannPB& p)
{
  bool master = comm->am_world_master();

  lbann_data::Model *model = p.mutable_model();

  int parallel_io = model->num_parallel_readers();
  if (parallel_io == 0) {
    if (master) {
      cout << "\tMax Parallel I/O Fetch: " << comm->get_procs_per_model() <<
           " (Limited to # Processes)" << endl;
    }
    parallel_io = comm->get_procs_per_model();
    model->set_num_parallel_readers(parallel_io); //adjust the prototext
  } else {
    if (master) {
      cout << "\tMax Parallel I/O Fetch: " << parallel_io << endl;
    }
  }
}

void set_data_readers_filenames(std::string which, lbann_data::LbannPB& p) {
  options *opts = options::get();
  lbann_data::DataReader *readers = p.mutable_data_reader();
  int size = readers->reader_size();
  for (int j=0; j<size; j++) {
    lbann_data::Reader *r = readers->mutable_reader(j);
    if (r->role() == which) {
      std::stringstream s;
      s << "data_filedir_" << which;
      if (opts->has_string(s.str().c_str())) {
         r->set_data_filedir(opts->get_string(s.str().c_str()));
      }
      s.clear();
      s.str("");
      s << "data_filename_" << which;
      if (opts->has_string(s.str().c_str())) {
         r->set_data_filename(opts->get_string(s.str().c_str()));
      }
      s.clear();
      s.str("");
      s << "label_filename_" << which;
      if (opts->has_string(s.str().c_str())) {
         r->set_label_filename(opts->get_string(s.str().c_str()));
      }
    }
  }
}

void get_cmdline_overrides(lbann::lbann_comm *comm, lbann_data::LbannPB& p)
{
  bool master = comm->am_world_master();
  if (not master) {
    return;
  }

  options *opts = options::get();
  lbann_data::Model *model = p.mutable_model();

  if (opts->has_string("data_filedir_train") or opts->has_string("data_filename_train")
      or opts->has_string("label_filename_train")) {
    set_data_readers_filenames("train", p);
  }
  if (opts->has_string("data_filedir_test") or opts->has_string("data_filename_test")
      or opts->has_string("label_filename_test")) {
    set_data_readers_filenames("test", p);
  }

  if (opts->has_string("image_dir")) {
    int sz = model->callback_size();
    for (int j=0; j<sz; j++) {
      lbann_data::Callback *c = model->mutable_callback(j);
      if (c->has_save_images()) {
        lbann_data::CallbackSaveImages *i = c->mutable_save_images();
        i->set_image_dir(opts->get_string("image_dir"));
      }
    }
  }
  if (opts->has_bool("no_im_comm") and opts->get_bool("no_im_comm")) {
    int sz = model->callback_size();
    for (int j=0; j<sz; j++) {
      lbann_data::Callback *c = model->mutable_callback(j);
      if (c->has_imcomm()) {
        c->clear_imcomm();
      }
    }
  }

  if (opts->has_int("mini_batch_size")) {
    model->set_mini_batch_size(opts->get_int("mini_batch_size"));
  }
  if (opts->has_int("num_epochs")) {
    model->set_num_epochs(opts->get_int("num_epochs"));
  }
  if (opts->has_int("block_size")) {
    model->set_block_size(opts->get_int("block_size"));
  }
  if (opts->has_int("procs_per_model")) {
    model->set_procs_per_model(opts->get_int("procs_per_model"));
  }
  if (opts->has_int("num_gpus")) {
    model->set_num_gpus(opts->get_int("num_gpus"));
  }
  if (opts->has_int("num_parallel_readers")) {
    model->set_num_parallel_readers(opts->get_int("num_parallel_readers"));
  }
  if (opts->has_bool("use_cudnn")) {
    model->set_use_cudnn(opts->get_int("use_cudnn"));
  }


  if (opts->has_string("opt")) {
    //defaults
    double learn_rate = opts->has_float("learn_rate") ? opts->get_float("learn_rate") : 0.01;
    double eps = opts->has_float("eps") ? opts->get_float("eps") : 1e-8;
    double beta1 = opts->has_float("beta1") ? opts->get_float("beta1") : 0.9;
    double beta2 = opts->has_float("beta2") ? opts->get_float("beta2") : 0.99;
    double init_learning_rate = opts->has_float("init_learning_rate") ? opts->get_float("init_learning_rate") : 0.01;
    double hyper_learning_rate = opts->has_float("hyper_learning_rate") ? opts->get_float("hyper_learning_rate") : 1e-7;
    double momentum = opts->has_float("momentum") ? opts->get_float("momentum") : 0.9;
    double decay_rate = opts->has_float("decay_rate") ? opts->get_float("decay_rate") : 0.5;
    bool nesterov = opts->has_bool("nesterov") ? opts->get_float("nesterov") : false;

    lbann_data::Optimizer *opt = new lbann_data::Optimizer;

    //construct the new optimizer
    std::string opt_string = opts->get_string("opt");
    if (opt_string == "adagrad") {
      lbann_data::Adagrad *a = new lbann_data::Adagrad;
      a->set_learn_rate(learn_rate);
      a->set_eps(eps);
      opt->set_allocated_adagrad(a);
    } else if (opt_string == "adam") {
      lbann_data::Adam *a = new lbann_data::Adam;
      a->set_learn_rate(learn_rate);
      a->set_eps(eps);
      a->set_beta1(beta1);
      a->set_beta2(beta2);
      opt->set_allocated_adam(a);
    } else if (opt_string == "hypergradient_adam") {
      lbann_data::HypergradientAdam *a = new lbann_data::HypergradientAdam;
      a->set_init_learning_rate(init_learning_rate);
      a->set_hyper_learning_rate(hyper_learning_rate);
      a->set_beta1(beta1);
      a->set_beta2(beta2);
      a->set_eps(eps);
      opt->set_allocated_hypergradient_adam(a);
    } else if (opt_string == "rmsprop") {
      lbann_data::Rmsprop *a = new lbann_data::Rmsprop;
      a->set_learn_rate(learn_rate);
      a->set_decay_rate(decay_rate);
      a->set_eps(eps);
      opt->set_allocated_rmsprop(a);
    } else if (opt_string == "sgd") {
      if (master) std::cerr << "\n\nsetting: sgd\n\n";
      lbann_data::Sgd *a = new lbann_data::Sgd;
      a->set_learn_rate(learn_rate);
      a->set_momentum(momentum);
      a->set_decay_rate(decay_rate);
      a->set_nesterov(nesterov);
      opt->set_allocated_sgd(a);
    } else {
        std::stringstream err;
        err << __FILE__ << " " << __LINE__
            << " :: unknown string for --optimizer: " << opt_string
            << " should be on of: adagrad, adam, hypergradient_adam, rmsprop, sgd";
        throw lbann_exception(err.str());
    }
    p.set_allocated_optimizer(opt);
  }
}

void print_parameters(lbann::lbann_comm *comm, lbann_data::LbannPB& p)
{
  if (not comm->am_world_master()) {
    return;
  }

  const lbann_data::Model &m = p.model();

  cout << endl
       << "Running with these parameters:\n"
       << " General:\n"
       << "  mini_batch_size:      " << m.mini_batch_size() << endl
       << "  num_epochs:           " << m.num_epochs()  << endl
       << "  block_size:           " << m.block_size()  << endl
       << "  procs_per_model:      " << m.procs_per_model()  << endl
       << "  num_gpus:             " << m.num_gpus()  << endl
       << "  num_parallel_readers: " << m.num_parallel_readers()  << endl
       << "  use_cudnn:            " << m.use_cudnn()  << endl
       << "  objective_function:   " << m.objective_function()  << endl
       << "  data_layout:          " << m.data_layout()  << endl
       << "     (only used for metrics)\n"
       << "\n"
       << " Optimizer:  ";

  const lbann_data::Optimizer &o = p.optimizer();
  if (o.has_adagrad()) {
    const lbann_data::Adagrad &a = o.adagrad();
    cout << "  Adagrad\n"
         << "  learn_rate: " << a.learn_rate()  << endl
         << "  eps:        " << a.eps()  << endl;
  } else if (o.has_rmsprop()) {
    const lbann_data::Rmsprop &a = o.rmsprop();
    cout <<  "  Rmsprop\n"
         << "  learn_rate: " << a.learn_rate()  << endl
         << "  decay_rate: " << a.decay_rate()  << endl
         << "  eps:        " << a.eps()  << endl;
  } else if (o.has_adam()) {
    const lbann_data::Adam &a = o.adam();
    cout << "  Adam\n"
         << "  learn_rate: " << a.learn_rate()  << endl
         << "  beta1:      " << a.beta1()  << endl
         << "  beta2:      " << a.beta2()  << endl
         << "  eps:        " << a.eps()  << endl;
  } else if (o.has_hypergradient_adam()) {
    const lbann_data::HypergradientAdam &a = o.hypergradient_adam();
    cout << "  HypergradientAdam\n"
         << "  init_learning_rate:  " << a.init_learning_rate()  << endl
         << "  hyper_learning_rate: " << a.hyper_learning_rate()  << endl
         << "  beta1:               " << a.beta1()  << endl
         << "  beta2:               " << a.beta2()  << endl
         << "  eps:                 " << a.eps()  << endl;
  } else if (o.has_sgd()) {
    const lbann_data::Sgd &a = o.sgd();
    cout << "  Sgd\n"
         << "  learn_rate: " << a.learn_rate()  << endl
         << "  momentum:   " << a.momentum()  << endl
         << "  decay_rate: " << a.decay_rate()  << endl
         << "  nesterov:   " << a.nesterov()  << endl;
  }
}

void print_help(lbann::lbann_comm *comm)
{
  if (not comm->am_world_master()) {
    return;
  }

  cerr <<
       "General usage: you need to specify three prototext files, e.g:\n"
       "  srun -n# proto --model=<string> --optimizer=<string> --reader=<string>\n"
       "\n"
       "  However, if you are re-running an experiment from a previously saved\n"
       "  file, you only need to specify --model=<string>\n"
       "  When proto is run, an output file containing the concatenated prototext\n"
       "  files, along with other data is written. The default name for this file\n"
       "  is 'data.prototext'  You can specify an alternative name via the option:\n"
       "  --saveme=<string>  You can suppress writing the file via the option:\n"
       "  --saveme=0\n"
       "\n"
       "Some prototext values can be over-riden on the command line;\n"
       "(notes: use '1' or '0' for bool; if no value is given for a flag,\n"
       "        e.g: --use_cudnn, then a value of '1' is assigned)\n"
       "\n"
       "General:\n"
       "  --mini_batch_size=<int>\n"
       "  --num_epochs=<int>\n"
       "  --block_size=<int>\n"
       "  --procs_per_model=<int>\n"
       "  --num_gpus=<int>\n"
       "  --use_cudnn=<bool>\n"
       "     has no effect unless lbann was compiled with: __LIB_CUDNN\n"
       "  --objective_function<string>\n"
       "      <string> must be: categorical_cross_entropy or mean_squared_error\n"
       "  --data_layout<string>\n"
       "      <string> must be: data_parallel or model_parallel\n"
       "      note: this will be applied to all layers, metrics (and others)\n"
       "            that take DATA_PARALLEL or MODEL_PARALLEL as a template parameter\n"
       "\n"
       "DataReaders:\n"
       "  --data_filedir_train=<string>   --data_filedir_test=<string>\n"
       "  --data_filename_train=<string>  --data_filename_test=<string>\n"
       "  --label_filename_train=<string> --label_filename_test=<string>\n"
       "\n"
       "Callbacks:\n"
       "  --image_dir=<string>\n"
       "      if the model has callback_save_images, this determines where the\n"
       "      images are saved\n"
       "  --no_im_comm=<bool>\n"
       "      removes ImComm callback, if present; this is intended for\n"
       "      running alexnet with a single model, but may be useful elsewhere\n"
       "\n"
       "Optimizers; all values except for nesterov are floats;\n"
       "            the values shown in <...> are the default values, that will be\n"
       "            used if the option is not specified on the cmd line.\n"
       "            If you specify an option that is not applicable to your choice\n"
       "            of optimizer, the option is ignored\n"
       "\n"
       "  --opt=<string>\n"
       "     <string> must be one of:\n"
       "         adagrad, adam, hypergradient_adam, rmsprop, sgd\n"
       "\n"
       "  --learn_rate=< 0.01 >          (all except hypergradient_adam)\n"
       "  --eps=< 1e-8 >                 (all except sgd)\n"
       "  --beta1=< 0.9 >                (adam, hypergradient_adam)\n"
       "  --beta2=< 0.99 >               (adam, hypergradient_adam)\n"
       "  --init_learning_rate=< 0.01 >  (hypergradient_adam)\n"
       "  --hyper_learning_rate=< 1e-7 > (hypergradient_adam)\n"
       "  --momentum=< 0.9 >             (sgd)\n"
       "  --decay_rate=< 0.5 >           (sgd, rmsprop)\n"
       "  --nesterov=< false >           (sgd)\n";
}

void copy_file(std::string fn, std::ofstream &out)
{
  std::ifstream in(fn.c_str());
  if (not in.is_open()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: failed to open file for reading: " << fn;
    throw std::runtime_error(err.str());
  }
  std::stringstream s;
  s << in.rdbuf();
  out << s.str();
}

void save_session(lbann::lbann_comm *comm, int argc, char **argv, lbann_data::LbannPB& p)
{
  if (not comm->am_world_master()) {
    return;
  }

  options *opts = options::get();

  //do not write output file for a repeated experiment;
  //may want to revisit this decision later ...
  if (opts->has_string("loadme")) {
    return;
  }

  //get output filename
  std::string base = ".";
  if (not opts->has_string("saveme")) {
    std::cerr << "\nNOT WRITING SAVE_SESSION FILE since option --saveme=<string> is absent\n\n";
    return;
  }
  std::string name = opts->get_string("saveme");
  if (name == "0") {
    std::cerr << "\nNOT WRITING SAVE_SESSION FILE due to option: --saveme=0\n\n";
    return;
  }

  //check if "name" exists; if yes, append "_1"
  bool exists = false;
  ifstream in(name.c_str());
  if (in) {
    exists = true;
    in.close();
  }
  if (exists) {
    name += "_1";
    //opts["saveme"] = name;
  }

  //open output file
  std::ofstream out(name.c_str());
  if (not out.is_open()) {
    std::stringstream err;
    err << __FILE__ << " " << __LINE__
        << " :: failed to open file for writing: " << name;
    throw std::runtime_error(err.str());
  }
  std::cout << std::endl << "writing options and prototext to file: " << name << "\n\n";

  //output all data
  out << "# cmd line for original experiment:\n#  $ ";
  for (int h=0; h<argc; h++) {
    out << argv[h] << " ";
  }
  std::string lbann_version("unknown: LBANN_VERSION is not defined");

#ifdef LBANN_VERSION
  lbann_version = LBANN_VERSION;
#endif

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::time_t r = std::time(nullptr);
  char *tm = std::ctime(&r);
  size_t fixme = strlen(tm);
  tm[fixme-1] = 0;
  out << "\n#\n# Experiment conducted at: "
      <<  tm
      << "\n#\n#\n# Experiment was run with lbann version: "
      << lbann_version << "\n#\n#\n# To rerun the experiment: \n"
      << "#  $ srun -n" << size << " " << argv[0]
      << " --loadme=" << opts->get_string("saveme") << "\n#\n#\n";

  out << "# Selected SLURM Environment Variables:\n";
  std::vector<std::string> v = {"HOST", "SLURM_NODELIST", "SLURM_NNODES", "SLURM_NTASKS", "SLURM_TASKS_PER_NODE"};
  for (size_t i=0; i<v.size(); i++) {
    char *c = std::getenv(v[i].c_str());
    if (c != 0) {
      out << "# " << v[i] << "=" << c << std::endl;
    }
  }
  out << "\n#\n#\n";

  std::string s;
  google::protobuf::TextFormat::PrintToString(p, &s);
  out << s;
  out.close();
}
