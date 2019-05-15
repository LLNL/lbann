import sys
sys.path.insert(0, '../common_python')
import tools
import collections, csv, os, pprint, re, time


# Set up the command ##########################################################
def get_command(cluster, dir_name, model_folder, model_name, executable,
                output_file_name, error_file_name, compiler_name, weekly=False):
    if model_name in ['alexnet', 'conv_autoencoder_imagenet']:
        data_reader_percent = 0.01
        if weekly:
            data_reader_percent = 0.10
        command = tools.get_command(
            cluster=cluster, executable=executable, num_nodes=16,
            partition='pbatch', time_limit=600, num_processes=32,
            dir_name=dir_name,
            data_filedir_train_default='/p/lscratchh/brainusr/datasets/ILSVRC2012/original/train/',
            data_filename_train_default='/p/lscratchh/brainusr/datasets/ILSVRC2012/labels/train.txt',
            data_filedir_test_default='/p/lscratchh/brainusr/datasets/ILSVRC2012/original/val/',
            data_filename_test_default='/p/lscratchh/brainusr/datasets/ILSVRC2012/labels/val.txt',
            data_reader_name='imagenet', data_reader_percent=data_reader_percent,
            model_folder=model_folder, model_name=model_name, num_epochs=20,
            optimizer_name='adagrad', output_file_name=output_file_name,
            error_file_name=error_file_name)
    elif model_name in ['conv_autoencoder_mnist', 'lenet_mnist']:
        if (model_name == 'lenet_mnist') and \
                (compiler_name in ['clang4', 'intel18']):
            partition = 'pbatch'
            time_limit = 600
        else:
            partition = 'pdebug'
            time_limit = 30
        if (cluster == 'ray') and (model_name == 'conv_autoencoder_mnist'):
            num_processes = 20
        else:
            num_processes = 2
        command = tools.get_command(
            cluster=cluster, executable=executable, num_nodes=1,
            partition=partition, time_limit=time_limit,
            num_processes=num_processes, dir_name=dir_name,
            data_filedir_default='/p/lscratchh/brainusr/datasets/MNIST',
            data_reader_name='mnist', model_folder=model_folder,
            model_name=model_name, num_epochs=5, optimizer_name='adagrad',
            output_file_name=output_file_name, error_file_name=error_file_name)
    else:
        raise Exception('Invalid model: %s' % model_name)
    return command

# Run LBANN ###################################################################


def run_lbann(command, model_name, output_file_name, error_file_name,
              should_log=False):
    print('About to run: %s' % command)
    print('%s began waiting in the queue at ' % model_name +
          time.strftime('%H:%M:%S', time.localtime()))
    output_value = os.system(command)
    print('%s finished at ' % model_name +
          time.strftime('%H:%M:%S', time.localtime()))
    lbann_exceptions = []
    timed_out = False
    if should_log or (output_value != 0):
        output_file = open(output_file_name, 'r')
        for line in output_file:
            print('%s: %s' % (output_file_name, line))
            is_match = re.search(
                'This lbann_exception is about to be thrown:(.*)', line)
            if is_match:
                lbann_exceptions.append(is_match.group(1))
            is_match = re.search('CANCELLED AT (.*) DUE TO TIME LIMIT', line)
            if is_match:
                timed_out = True
        error_file = open(error_file_name, 'r')
        for line in error_file:
            print('%s: %s' % (error_file_name, line))
            is_match = re.search('LBANN error on (.*)', line)
            if is_match:
                lbann_exceptions.append(is_match.group(1))
    if output_value != 0:
        error_string = ('Model %s crashed with output_value=%d, timed_out=%s,'
                        ' and lbann exceptions=%s. Command was: %s') % (
            model_name, output_value, str(timed_out),
            str(collections.Counter(lbann_exceptions)), command)
        raise Exception(error_string)
    return output_value

# Extract data from output ####################################################


def populate_data_dict_epoch(regex, line, data_field, data_fields, data_dict,
                             model_id):
    is_match = re.search(regex, line)
    if is_match and (data_field in data_fields):
        if model_id not in data_dict[data_field].keys():
            data_dict[data_field][model_id] = {}
        epoch_id = is_match.group(1)
        value = float(is_match.group(2))
        data_dict[data_field][model_id][epoch_id] = value


def populate_data_dict_overall(regex, line, data_field, data_fields, data_dict,
                               model_id):
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
            print('extract_data: %s: %s' % (output_file_name, line))

        # Check if line is reporting model results
        is_model = re.search('^Model ([0-9]+)', line)
        if not is_model:
            is_model = re.search('^model([0-9]+)', line)
        if is_model:
            print('extract_data: is_model={is_model}'.format(is_model=is_model))
            model_id = is_model.group(1)

            regex = 'training epoch ([0-9]+) objective function : ([0-9.]+)'
            data_field = 'training_objective_function'
            populate_data_dict_epoch(regex, line, data_field, data_fields,
                                     data_dict, model_id)

            regex = 'training epoch ([0-9]+) run time : ([0-9.]+)'
            data_field = 'training_run_time'
            populate_data_dict_epoch(regex, line, data_field, data_fields,
                                     data_dict, model_id)

            regex = 'training epoch ([0-9]+) mini-batch time statistics : ([0-9.]+)s mean, ([0-9.]+)s max, ([0-9.]+)s min, ([0-9.]+)s stdev'
            is_match = re.search(regex, line)
            if is_match:
                print('extract_data: is_mini-batch time statistics={is_match}'.format(
                    is_match=is_match))
                epoch_id = is_match.group(1)
                mean_value = float(is_match.group(2))
                max_value = float(is_match.group(3))
                min_value = float(is_match.group(4))
                stdev_value = float(is_match.group(5))
                data_field = 'training_mean'
                if data_field in data_fields:
                    if model_id not in data_dict[data_field].keys():
                        data_dict[data_field][model_id] = {}
                    print('extract_data: mean_value={mv}'.format(mv=mean_value))
                    data_dict[data_field][model_id][epoch_id] = mean_value
                data_field = 'training_max'
                if data_field in data_fields:
                    if model_id not in data_dict[data_field].keys():
                        data_dict[data_field][model_id] = {}
                    print('extract_data: max_value={mv}'.format(mv=max_value))
                    data_dict[data_field][model_id][epoch_id] = max_value
                data_field = 'training_min'
                if data_field in data_fields:
                    if model_id not in data_dict[data_field].keys():
                        data_dict[data_field][model_id] = {}
                    print('extract_data: min_value={mv}'.format(mv=min_value))
                    data_dict[data_field][model_id][epoch_id] = min_value
                data_field = 'training_stdev'
                if data_field in data_fields:
                    if model_id not in data_dict[data_field].keys():
                        data_dict[data_field][model_id] = {}
                    print('extract_data: stdev={sv}'.format(sv=stdev_value))
                    data_dict[data_field][model_id][epoch_id] = stdev_value

            regex = 'test categorical accuracy : ([0-9.]+)'
            data_field = 'test_accuracy'
            populate_data_dict_overall(regex, line, data_field, data_fields,
                                       data_dict, model_id)
    output_file.close()
    if should_log:
        print('extract_data: Extracted Data below:')
        pprint.pprint(data_dict)
    return data_dict

# Skeleton ####################################################################


def skeleton(cluster, dir_name, executable, model_folder, model_name,
             data_fields, should_log, compiler_name=None, weekly=False):
    if compiler_name is None:
        output_file_name = '%s/bamboo/integration_tests/output/%s_output.txt' % (dir_name, model_name)
        error_file_name = '%s/bamboo/integration_tests/error/%s_error.txt' % (dir_name, model_name)
    else:
        output_file_name = '%s/bamboo/integration_tests/output/%s_%s_output.txt' % (dir_name, model_name, compiler_name)
        error_file_name = '%s/bamboo/integration_tests/error/%s_%s_error.txt' % (dir_name, model_name, compiler_name)
    command = get_command(
        cluster, dir_name, model_folder, model_name, executable,
        output_file_name, error_file_name, compiler_name, weekly=weekly)
    run_lbann(command, model_name, output_file_name,
              error_file_name, should_log)  # Don't need return value
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
            values[row_header] = dict(
                zip(column_headers[1:], map(float, row[1:])))
    return values
