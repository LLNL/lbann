import csv, os, pprint, re, time

# Set up the command ##########################################################

def get_model_option(dir_name, model_folder, model_name):
    return '--model=%s/model_zoo/models/%s/model_%s.prototext' % (dir_name, model_folder, model_name)

def get_reader_option(dir_name, reader_name):
    return '--reader=%s/model_zoo/data_readers/data_reader_%s.prototext' % (dir_name, reader_name)

def get_optimizer_option(dir_name, optimizer_name):
    return '--optimizer=%s/model_zoo/optimizers/opt_%s.prototext' % (dir_name, optimizer_name)

def get_command(dir_name, model_folder, model_name, executable, output_file_name):
    # N => number of nodes.
    # p => partition.
    # t => timeout period, in minutes.
    # n => number of processes to run. MPI Rank.
    # n / procs_per_model = how many models should be made. (It must be true that n >= procs_per_model).
    # num_epochs => number of epochs.
    # data_reader_percent => how much of the data to use.
    model_option = get_model_option(dir_name, model_folder, model_name)
    optimizer_option = get_optimizer_option(dir_name, 'adagrad')
    if model_name == 'alexnet':
        reader_option = get_reader_option(dir_name, 'imagenet')
        command = 'salloc -N 16 -p pbatch -t 600 srun -n 32 %s %s %s %s --num_epochs=20 --data_reader_percent=0.10 > %s' % (executable, model_option, reader_option, optimizer_option, output_file_name)
    elif model_name == 'conv_autoencoder_mnist':
        reader_option = get_reader_option(dir_name, 'mnist')
        command = 'salloc -N 1 -p pdebug -t 10 srun -n 2 %s %s %s %s --num_epochs=5 > %s' % (executable, model_option, reader_option, optimizer_option, output_file_name)
    elif model_name == 'conv_autoencoder_imagenet':
        reader_option = get_reader_option(dir_name, 'imagenet')
        command = 'salloc -N 16 -p pbatch -t 600 srun -n 32 %s %s %s %s --num_epochs=20 --data_reader_percent=0.10 > %s' % (executable, model_option, reader_option, optimizer_option, output_file_name)
    elif model_name == 'lenet_mnist':
        reader_option = get_reader_option(dir_name, 'mnist')
        command = 'salloc -N 1 -p pdebug -t 10 srun -n 2 %s %s %s %s --num_epochs=5 > %s' % (executable, model_option, reader_option, optimizer_option, output_file_name)
    else:
        raise Exception('Invalid model: %s' % model_name)
    return command

# Run LBANN ###################################################################

def run_lbann(command, model_name, output_file_name, should_log):
    print('About to run: %s' % command)
    print('%s began waiting in the queue at ' % model_name + time.strftime('%H:%M:%S', time.localtime()))
    value = os.system(command)
    print('%s finished at ' % model_name + time.strftime('%H:%M:%S', time.localtime()))
    if should_log or (value != 0):
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
    if value != 0:
        raise Exception('Model %s crashed' % model_name)

# Extract data from output ####################################################

def populate_data_dict_epoch(regex, line, data_field, data_fields, data_dict, model_id):
    is_match = re.search(regex, line)
    if is_match and (data_field in data_fields):
        if model_id not in data_dict[data_field].keys():
            data_dict[data_field][model_id] = {}
        epoch_id = is_match.group(1)
        value = float(is_match.group(2))
        data_dict[data_field][model_id][epoch_id] = value

def populate_data_dict_overall(regex, line, data_field, data_fields, data_dict, model_id):
    is_match = re.search(regex, line)
    if is_match and (data_field in data_fields):
        if model_id not in data_dict[data_field].keys():
            data_dict[data_field][model_id] = {}
        value = float(is_match.group(1))
        data_dict[data_field][model_id]['overall'] = value

# data_dict[data_field][model_id][epoch_id] = float
# data_fields is the list or set of data we're interested in.
def extract_data(output_file_name, data_fields, should_log):
    output_file = open(output_file_name, 'r')
    data_dict = {}
    for data_field in data_fields:
        data_dict[data_field] = {}

    for line in output_file:
        if should_log:
            print('%s: %s' % (output_file_name, line))
            
        # Check if line is reporting model results
        is_model = re.search('^Model ([0-9]+)', line)
        if is_model:
            model_id = is_model.group(1)

            regex = 'training epoch ([0-9]+) objective function : ([0-9.]+)'
            data_field = 'training_objective_function'
            populate_data_dict_epoch(regex, line, data_field, data_fields, data_dict, model_id)

            regex = 'training epoch ([0-9]+) run time : ([0-9.]+)'
            data_field = 'training_run_time'
            populate_data_dict_epoch(regex, line, data_field, data_fields, data_dict, model_id)

            regex = 'training epoch ([0-9]+) mini-batch time statistics : ([0-9.]+)s mean, ([0-9.]+)s max, ([0-9.]+)s min, ([0-9.]+)s stdev'
            is_match = re.search(regex, line)
            if is_match:
                epoch_id = is_match.group(1)
                mean_value = float(is_match.group(2))
                max_value = float(is_match.group(3))
                min_value = float(is_match.group(4))
                stdev_value = float(is_match.group(5))
                data_field = 'training_mean'
                if data_field in data_fields:
                    if model_id not in data_dict[data_field].keys():
                        data_dict[data_field][model_id] = {}
                    data_dict[data_field][model_id][epoch_id] = mean_value
                data_field = 'training_max'
                if data_field in data_fields:
                    if model_id not in data_dict[data_field].keys():
                        data_dict[data_field][model_id] = {}
                    data_dict[data_field][model_id][epoch_id] = max_value
                data_field = 'training_min'
                if data_field in data_fields:
                    if model_id not in data_dict[data_field].keys():
                        data_dict[data_field][model_id] = {}
                    data_dict[data_field][model_id][epoch_id] = min_value
                data_field = 'training_stdev'
                if data_field in data_fields:
                    if model_id not in data_dict[data_field].keys():
                        data_dict[data_field][model_id] = {}
                    data_dict[data_field][model_id][epoch_id] = stdev_value

            regex = 'test categorical accuracy : ([0-9.]+)'
            data_field = 'test_accuracy'
            populate_data_dict_overall(regex, line, data_field, data_fields, data_dict, model_id)
    output_file.close()
    if should_log:
        pprint.pprint(data_dict)
    return data_dict

# Skeleton ####################################################################

def skeleton(dir_name, executable, model_folder, model_name, data_fields, should_log):
    output_file_name = '%s/bamboo/integration_tests/%s_output.txt' % (dir_name, model_name)
    command = get_command(dir_name, model_folder, model_name, executable, output_file_name)
    run_lbann(command, model_name, output_file_name, should_log)
    return extract_data(output_file_name, data_fields, should_log)

# Misc. functions  ############################################################

# csv_dict[row_header][column_header] = float
def csv_to_dict(csv_path):
  with open(csv_path, 'r') as csv_file:
    reader = csv.reader(csv_file, skipinitialspace=True)
    column_headers = reader.next()
    values = {}
    for row in reader:
      row_header = row[0]
      values[row_header] = dict(zip(column_headers[1:], map(float, row[1:])))
  return values
