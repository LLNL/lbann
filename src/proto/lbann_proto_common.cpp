#include "lbann/proto/lbann_proto_common.hpp"

#include "lbann/lbann.hpp"
#include "lbann/lbann_base.hpp"
#include "lbann/lbann_comm.hpp"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <unordered_map>

using namespace lbann;

lbann_callback_imcomm::comm_type get_comm_type(const string &s) {
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

const data_layout get_data_layout(const string& s, const char *file, int line)
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
  bool master = comm->am_world_master();

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

    //////////////////////////////////////////////////////////////////
    // LAYER: Relu
    //////////////////////////////////////////////////////////////////
    if (layer.has_relu()) {
      const lbann_data::Relu &ell = layer.relu();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new relu_layer<data_layout::MODEL_PARALLEL>(layer_id, comm, mb_size, cudnn);
      } else {
        d = new relu_layer<data_layout::DATA_PARALLEL>(layer_id, comm, mb_size, cudnn);
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: sigmoid
    //////////////////////////////////////////////////////////////////
    if (layer.has_sigmoid()) {
      const lbann_data::Sigmoid &ell = layer.sigmoid();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new sigmoid_layer<data_layout::MODEL_PARALLEL>(layer_id, comm, mb_size);
      } else {
        d = new sigmoid_layer<data_layout::DATA_PARALLEL>(layer_id, comm, mb_size);
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
          model->create_optimizer(),
          mb_size,
          all_layers[original_layer]
        );  
      } else {
        d = new reconstruction_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          model->create_optimizer(),
          mb_size,
          all_layers[original_layer]
        );  
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: input_distributed_minibatch_parallel_io
    //////////////////////////////////////////////////////////////////
    if (layer.has_input_distributed_minibatch_parallel_io()) {
      const lbann_data::InputDistributedMiniBatchParallelIO& ell = layer.input_distributed_minibatch_parallel_io();
      //please do not delete this! it's here to remind me that something needs
      //fixing. Thanks, Dave H.
      if (master) cout << "XX numreaders: " << m.num_parallel_readers() << endl;
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new input_layer_distributed_minibatch_parallel_io<data_layout::MODEL_PARALLEL>(
          comm,
          mb_size,
          m.num_parallel_readers(),
          data_readers);
      } else {
        d = new input_layer_distributed_minibatch_parallel_io<data_layout::DATA_PARALLEL>(
          comm,
          mb_size,
          m.num_parallel_readers(),
          data_readers);
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: input_partitioned_minibatch_parallel_io
    //////////////////////////////////////////////////////////////////
    if (layer.has_input_partitioned_minibatch_parallel_io()) {
      const lbann_data::InputPartitionedMiniBatchParallelIO& ell = layer.input_partitioned_minibatch_parallel_io();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new input_layer_partitioned_minibatch_parallel_io<data_layout::MODEL_PARALLEL>(
          comm,
          mb_size,
          m.num_parallel_readers(),
          data_readers);
      } else {
        d = new input_layer_partitioned_minibatch_parallel_io<data_layout::DATA_PARALLEL>(
          comm,
          mb_size,
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
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new fully_connected_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm,
          mb_size,
          ell.num_neurons(),
          get_weight_initialization(ell.weight_initialization()),
          model->create_optimizer(),
          ell.has_bias());
      } else {
        d = new fully_connected_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size,
          ell.num_neurons(),
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
        d = new pooling_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm,
          mb_size,
          ell.num_dims(),
          &pool_dims[0],
          &pool_pads[0],
          &pool_strides[0],
          get_pool_mode(ell.pool_mode()),
          cudnn
        );
      } else {
        d = new pooling_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size,
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

      vector<int> input_dims;
      std::stringstream ss(ell.input_dims());
      int i;
      while (ss >> i) {
        input_dims.push_back(i);
      }

      vector<int> filter_dims;
      ss.clear();
      ss.str(ell.filter_dims());
      while (ss >> i) {
        filter_dims.push_back(i);
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
      int num_input_channels = ell.num_input_channels();
      int num_output_channels = ell.num_output_channels();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new convolution_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm,
          mb_size,
          num_dims,
          num_output_channels,
          &filter_dims[0],
          &conv_pads[0],
          &conv_strides[0],
          get_weight_initialization(ell.weight_initialization()),
          model->create_optimizer(),
          cudnn
        );
      } else {
        d = new convolution_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size,
          num_dims,
          num_output_channels,
          &filter_dims[0],
          &conv_pads[0],
          &conv_strides[0],
          get_weight_initialization(ell.weight_initialization()),
          model->create_optimizer(),
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

      vector<int> dims;
      std::stringstream ss(ell.dims());
      int i;
      while (ss >> i) {
        dims.push_back(i);
      }

      int num_dims = ell.num_dims();
      int num_channels = ell.num_channels();
      DataType lrn_alpha = ell.lrn_alpha();
      DataType lrn_beta = ell.lrn_beta();
      DataType lrn_k = ell.lrn_k();
      int window_width = ell.window_width();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new local_response_normalization_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm,
          mb_size,
          window_width,
          lrn_alpha,
          lrn_beta,
          lrn_k,
          cudnn);
      } else {
        d = new local_response_normalization_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size,
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
          mb_size,
          ell.keep_prob(),
          ell.alpha(),
          ell.scale()
        );
      } else {
        d = new selu_dropout<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size,
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
        d = new batch_normalization<data_layout::MODEL_PARALLEL>(
          layer_id,
          comm,
          mb_size,
          ell.decay(),
          ell.gamma(),
          ell.beta());
      } else {
        d = new batch_normalization<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size,
          ell.decay(),
          ell.gamma(),
          ell.beta());
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
          mb_size,
          ell.alpha(),
          ell.scale()
        );
      } else {
        d = new selu_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size,
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
          comm,
          mb_size
        );  
      } else {
        d = new tanh_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size
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
          comm,
          mb_size
        );  
      } else {
        d = new softplus_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size
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
          comm,
          mb_size
        );  
      } else {
        d = new smooth_relu_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size
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
          mb_size,
          ell.leak()
        );
      } else {
        d = new leaky_relu_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size,
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
          comm,
          mb_size
        );  
      } else {
        d = new id_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size
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
          mb_size,
          ell.alpha()
        );
      } else {
        d = new elu_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size,
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
          mb_size,
          ell.keep_prob());
      } else {
        d = new dropout<data_layout::DATA_PARALLEL>(
          layer_id,
          comm,
          mb_size,
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
      const lbann_data::Softmax& ell = layer.softmax();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new softmax_layer<data_layout::MODEL_PARALLEL>(
          layer_id,
          mb_size,
          comm,
          model->create_optimizer()
        );
      } else {
        d = new softmax_layer<data_layout::DATA_PARALLEL>(
          layer_id,
          mb_size,
          comm,
          model->create_optimizer()
        );
      }
      all_layers[layer.index()] = d;
      layer_mapping[layer.index()] = model->get_layers().size();
      model->add(d);
    }

    //////////////////////////////////////////////////////////////////
    // LAYER: target_partitioned_minibatch_parallel_io
    //////////////////////////////////////////////////////////////////
    if (layer.has_target_partitioned_minibatch_parallel_io()) {
      const lbann_data::TargetPartitionedMinibatchParallelIO& ell = layer.target_partitioned_minibatch_parallel_io();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new  target_layer_partitioned_minibatch_parallel_io<data_layout::MODEL_PARALLEL>(
          comm,
          mb_size,
          m.num_parallel_readers(),
          data_readers,
          ell.shared_data_reader(),
          ell.for_regression());
      } else {
        d = new  target_layer_partitioned_minibatch_parallel_io<data_layout::DATA_PARALLEL>(
          comm,
          mb_size,
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
    // LAYER: target_distributed_minibatch_parallel_io
    //////////////////////////////////////////////////////////////////
    if (layer.has_target_distributed_minibatch_parallel_io()) {
      const lbann_data::TargetDistributedMinibatchParallelIO& ell = layer.target_distributed_minibatch_parallel_io();
      if (dl == data_layout::MODEL_PARALLEL) {
        d = new  target_layer_distributed_minibatch_parallel_io<data_layout::MODEL_PARALLEL>(
          comm,
          mb_size,
          m.num_parallel_readers(),
          data_readers,
          ell.shared_data_reader(),
          ell.for_regression());
      } else {
        d = new  target_layer_distributed_minibatch_parallel_io<data_layout::DATA_PARALLEL>(
          comm,
          mb_size,
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
      lbann_summary *summarizer = nullptr;
      if (c.dir() != "none") {
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
      lbann_summary *summarizer = nullptr;
      if (c.dir() != "none") {
        summarizer = new lbann_summary(c.dir(), comm);
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
  int size = m.metric_size();
  for (int j=0; j<size; j++) {
    string metric = m.metric(j);
    data_layout dl = get_data_layout(m.data_layout(), __FILE__, __LINE__);
    if (metric == "categorical_accuracy") {
      if (dl == data_layout::MODEL_PARALLEL) {
        model->add_metric(new metrics::categorical_accuracy<data_layout::MODEL_PARALLEL>(comm));
      } else {
        model->add_metric(new metrics::categorical_accuracy<data_layout::DATA_PARALLEL>(comm));
      }
    } else if (metric == "mean_squared_error") {
      if (dl == data_layout::MODEL_PARALLEL) {
        model->add_metric(new metrics::mean_squared_error<data_layout::MODEL_PARALLEL>(comm));
      } else {
        model->add_metric(new metrics::mean_squared_error<data_layout::DATA_PARALLEL>(comm));
      }
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

optimizer_factory *init_optimizer_factory(lbann_comm *comm, const lbann_data::LbannPB& p)
{
  const lbann_data::Optimizer& optimizer = p.optimizer();

  const string name = optimizer.name();
  double learn_rate = optimizer.learn_rate();
  double momentum = optimizer.momentum();
  double decay_rate = optimizer.decay();
  double beta1 = optimizer.beta1();
  double beta2 = optimizer.beta2();
  double eps = optimizer.eps();
  bool nesterov = optimizer.nesterov();

  //note: learn_rate, momentum, decay are DataType in LBANN, which is
  //      probably float. They'll be properly cast in the following

  optimizer_factory *factory;

  if (name == "adagrad") {
    factory = new adagrad_factory(comm, learn_rate, eps);
  } else if (name == "rmsprop") {
    factory = new rmsprop_factory(comm, learn_rate, decay_rate, eps);
  } else if (name == "adam") {
    factory = new adam_factory(comm, learn_rate, beta1, beta2, eps);
  } else if (name == "hypergradient_adam") {
    factory = new hypergradient_adam_factory(comm, learn_rate, beta1, beta2, eps);
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
        double validate_percent = ((double) num_validate / (double) (num_train+num_validate))*100.0;
        double train_percent = ((double) num_train / (double) (num_train+num_validate))*100.0;
        cout << "Training using " << train_percent << "% of the training data set, which is " << reader->getNumData() << " samples." << endl
             << "Validating training using " << validate_percent << "% of the training data set, which is " << reader_validation->getNumData() << " samples." << endl;
      }

      data_readers[execution_mode::validation] = reader_validation;
    }
  }
}

void readPrototextFile(string fn, lbann_data::LbannPB& pb)
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

bool writePrototextFile(const char *fn, lbann_data::LbannPB& pb)
{
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
