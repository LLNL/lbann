import os, time

def get_command(dir_name, executable, model_name, output_file_name):
    optimizer = '--optimizer=%s/model_zoo/optimizers/opt_adagrad.prototext' % dir_name
    # N => number of nodes.
    # p => partition.
    # t => timeout period, in minutes.
    # n => number of processes to run. MPI Rank.
    # n / procs_per_model = how many models should be made. (It must be true that n >= procs_per_model).
    # num_epochs => number of epochs.
    # data_reader_percent => how much of the data to use.
    if model_name == 'conv_autoencoder_mnist':
        model = '--model=%s/model_zoo/models/autoencoder_mnist/model_%s.prototext' % (dir_name, model_name)
        reader = '--reader=%s/model_zoo/data_readers/data_reader_mnist.prototext' % dir_name
        command = 'salloc -N 16 -p pbatch -t 600 srun -n 32 %s %s %s %s --num_epochs=20 --data_reader_percent=0.10 > %s' % (executable, model, reader, optimizer, output_file_name)
        return command
    elif model_name == 'conv_autoencoder_imagenet':
        model = '--model=%s/model_zoo/models/autoencoder_imagenet/model_%s.prototext' % (dir_name, model_name)
        reader = '--reader=%s/model_zoo/data_readers/data_reader_imagenet.prototext' % dir_name
        command = 'salloc -N 1 -p pbatch -t 60 srun -n 2 %s %s %s %s --num_epochs=5 > %s' % (executable, model, reader, optimizer, output_file_name)
        return command
    else:
        raise Exception('Invalid model name: %s' % model_name)

def run_lbann(command, model_name, output_file_name, should_log=False):
  print('About to run: %s' % command)
  print('%s began waiting in the queue at ' % model_name + time.strftime('%H:%M:%S', time.localtime()))
  value = os.system(command)
  print('%s finished at ' % model_name + time.strftime('%H:%M:%S', time.localtime()))
  if should_log or (value != 0):
    output_file = open(output_file_name, 'r')
    for line in output_file:
      print('%s: %s' % (output_file_name, line))
    raise Exception('Model %s crashed' % model_name)

def skeleton(dir_name, executable, model_name, should_log=False):
    output_file_name = '%s_output.txt' % model_name
    command = get_command(dir_name, executable, model_name, output_file_name)
    run_lbann(command, model_name, output_file_name, should_log)
