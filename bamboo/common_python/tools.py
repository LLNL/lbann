import math, os

def check_list(substrings, strings):
    errors = []
    for string in strings:
        for substring in substrings:
            if (string != None) and (substring in string):
               errors.append('%s contains %s' % (string, substring))
    return errors

def get_command(cluster, executable, num_nodes=None, partition=None,
                time_limit=None, num_processes=None, dir_name=None,
                data_filedir_ray=None, data_filedir_train_ray=None,
                data_filename_train_ray=None, data_filedir_test_ray=None,
                data_filename_test_ray=None, data_reader_name=None,
                data_reader_path=None, data_reader_percent=None,
                exit_after_setup=False, mini_batch_size=None,
                model_folder=None, model_name=None, model_path=None,
                num_epochs=None, optimizer_name=None, optimizer_path=None,
                processes_per_model=None, output_file_name=None,
                return_tuple=False):
    # Check parameters for black-listed characters like semi-colons that
    # would terminate the command and allow for an extra command
    blacklist = [';', '--']
    strings = [partition, dir_name, data_filedir_ray, data_filedir_train_ray,
               data_filename_train_ray, data_filedir_test_ray,
               data_filename_test_ray, data_reader_name, data_reader_path,
               model_folder, model_name, model_path, optimizer_name,
               optimizer_path, output_file_name]
    invalid_character_errors = check_list(blacklist, strings)
    if invalid_character_errors != []:
        raise Exception('Invalid character(s): %s' % ' , '.join(invalid_character_errors))

    # Determine scheduler
    if cluster in ['catalyst', 'surface']:
        scheduler = 'slurm'
    elif cluster == 'ray':
        scheduler = 'lsf'
    else:
        raise Exception('Unsupported Cluster: %s' % cluster)

    # Description of command line options are from the appropriate command's
    # man pages
    if scheduler == 'slurm':
        # Create allocate command
        command_allocate = ''
        # Allocate a node if we don't have one already
        # Running the tests manually allows for already having a node allocated
        if os.getenv('SLURM_NNODES') == None:
            command_allocate = 'salloc'
            option_num_nodes = ''
            option_partition = ''
            option_time_limit = ''
            if num_nodes != None:
                # --nodes=<minnodes[-maxnodes]> =>
                # Request that a minimum of minnodes nodes be allocated to this
                # job. A maximum node count may also be specified with
                # maxnodes.
                option_num_nodes = ' --nodes=%d' % num_nodes
            if partition != None:
                # Surface does not have pdebug, so switch to pbatch
                if (cluster == 'surface') and (partition == 'pdebug'):
                    partition = 'pbatch'
                # --partition => Request a specific partition for the resource
                # allocation.
                option_partition = ' --partition=%s' % partition
            if time_limit != None:
                # --time => Set a limit on the total run time of the job
                # allocation.
                # Time limit in minutes
                option_time_limit = ' --time=%d' % time_limit
            command_allocate = '%s%s%s%s' % (
                command_allocate, option_num_nodes, option_partition,
                option_time_limit)
            
        # Create run command
        if command_allocate == '':
            command_run = 'srun'
        else:
            command_run = ' srun'
        option_num_processes = ''
        if num_processes != None:
            # --ntasks => Specify  the  number of tasks to run.
            # Number of processes to run => MPI Rank
            option_num_processes = ' --ntasks=%d' % num_processes
        command_run = '%s%s' % (command_run, option_num_processes)
        
    elif scheduler == 'lsf':
        # Create allocate command
        command_allocate = ''
        # Allocate a node if we don't have one already
        # Running the tests manually allows for already having a node allocated
        if os.getenv('LSB_HOSTS') == None:
            command_allocate = 'bsub'
            # x => Puts the host running your job into exclusive execution
            # mode.
            option_exclusive = ' -x'
            # G=> For fairshare scheduling. Associates the job with the
            # specified group.
            option_group = ' -G guests'
            # Is => Submits an interactive job and creates a pseudo-terminal
            # with shell mode when the job starts.
            option_interactive = ' -Is'
            option_num_processes = ''
            option_partition = ''
            option_processes_per_node = ''
            option_time_limit = ''
            if num_processes != None:
                # n => Submits a parallel job and specifies the number of
                # tasks in the job.
                option_num_processes = ' -n %d' % num_processes
                if (num_nodes != None) and (num_nodes != 0):
                    # R => Runs the job on a host that meets the specified
                    # resource requirements.
                    option_processes_per_node = ' -R "span[ptile=%d]"' % int(
                        math.ceil(float(num_processes)/num_nodes))
            if partition != None:
                # q => Submits the job to one of the specified queues.
                option_partition = ' -q %s' % partition
            if time_limit != None:
                # W => Sets the runtime limit of the job.
                option_time_limit = ' -W %d' % time_limit
            command_allocate = '%s%s%s%s%s%s%s%s' % (
                command_allocate, option_exclusive, option_group,
                option_interactive, option_num_processes, option_partition,
                option_processes_per_node, option_time_limit)

        # Create run command
        if command_allocate == '':
            command_run = 'mpirun'
        else:
            command_run = ' mpirun'
        option_num_processes = ''
        option_processes_per_node = ''
        if num_processes != None:
            # -np => Run this many copies of the program on the given nodes.
            option_num_processes = ' -np %d' % num_processes
            if (num_nodes != None) and (num_nodes != 0):
                option_processes_per_node = ' -N %d' % int(
                    math.ceil(float(num_processes)/num_nodes))
        command_run = '%s%s%s' % (
            command_run, option_num_processes, option_processes_per_node)
        
    else:
        raise Exception('Unsupported Scheduler %s' % scheduler)

    # Create LBANN command
    option_data_filedir = ''
    option_data_filedir_train = ''
    option_data_filename_train = ''
    option_data_filedir_test = ''
    option_data_filename_test = ''
    option_data_reader = ''
    option_data_reader_percent = ''
    option_exit_after_setup = ''
    option_mini_batch_size = ''
    option_model = ''
    option_num_epochs = ''
    option_optimizer = ''
    option_processes_per_model = ''
    lbann_errors = []
    if model_path != None:
        # If model_folder and/or model_name are set, an exception will be
        # raised later.
        option_model = ' --model=%s' % model_path
    if data_reader_path != None:
        # If data_reader_name is set, an exception will be raised later.
        option_data_reader_name = ' --reader=%s' % data_reader_path
    if optimizer_path != None:
        # If optimizer_name is set, an exception will be raised later.
        option_optimizer_name = ' --optimizer=%s' % optimizer_path
    if dir_name != None:
        if model_path != None:
            if (model_folder != None) or (model_name != None):
                lbann_errors.append(
                    'model_path is set but so is at least one of model folder and model_name')
        else:
            if (model_folder != None) and (model_name != None):
                option_model = ' --model=%s/model_zoo/%s/model_%s.prototext' % (dir_name, model_folder, model_name)
            elif model_folder != None:
                lbann_errors.append('model_folder set but not model_name.')
            elif model_name != None:
                lbann_errors.append('model_name set but not model_folder.')
        if data_reader_name != None:
            if data_reader_path != None:
                lbann_errors.append('data_reader_path is set but so is data_reader_name')
            else:
                option_data_reader = ' --reader=%s/model_zoo/data_readers/data_reader_%s.prototext' % (dir_name, data_reader_name)
        if optimizer_name != None:
            if optimizer_path != None:
                lbann_errors.append('optimizer_path is set but so is optimizer_name')
            else:
                option_optimizer = ' --optimizer=%s/model_zoo/optimizers/opt_%s.prototext' % (dir_name, optimizer_name)
        if (model_folder == None) and (model_name == None) and (data_reader_name == None) and (optimizer_name == None):
            lbann_errors.append('dir_name set but none of model_folder, model_name, data_reader_name, optimizer_name are.')
    elif (model_folder != None) or (model_name != None) or (data_reader_name != None) or (optimizer_name != None):
        lbann_errors.append(
            'dir_name is not set but at least one of model_folder, model_name, data_reader_name, optimizer_name is.')
    ray_parameters = [data_filedir_train_ray,
                      data_filename_train_ray,
                      data_filedir_test_ray,
                      data_filename_test_ray]
    if (data_reader_name != None) or (data_reader_path != None):
        if (cluster == 'ray'):
            if data_filedir_ray != None:
                if ray_parameters == [None, None, None, None]:
                    option_data_filedir = ' --data_filedir=%s' % data_filedir_ray
                else:
                    lbann_errors.append('data_fildir_ray set but so is at least one of [data_filedir_train_ray, data_filename_train_ray, data_filedir_test_ray, data_filename_test_ray]')
            elif None not in ray_parameters:
                option_data_filedir_train = ' --data_filedir_train=%s' % data_filedir_train_ray
                option_data_filename_train = ' --data_filename_train=%s' % data_filename_train_ray
                option_data_filedir_test = ' --data_filedir_test=%s' % data_filedir_test_ray
                option_data_filename_test = ' --data_filename_test=%s' % data_filename_test_ray
            else:
                lbann_errors.append('data_reader_name or data_reader_path is set but not data_filedir_ray. If a data reader is provided, an alternative filedir must be available for Ray. Alternatively, all of [data_filedir_train_ray, data_filename_train_ray, data_filedir_test_ray, data_filename_test_ray] can be set.')
    elif data_filedir_ray != None:
        lbann_errors.append(
            'data_filedir_ray set but neither data_reader_name or data_reader_path are.')
    elif filter(lambda x: x != None, ray_parameters) != []:
        lbann_errors.append('At least one of [data_filedir_train_ray, data_filename_train_ray, data_filedir_test_ray, data_filename_test_ray] is set, but neither data_reader_name or data_reader_path are.')
    if data_reader_percent != None:
        option_data_reader_percent = ' --data_reader_percent=%f' % data_reader_percent
    if exit_after_setup:
        option_exit_after_setup = ' --exit_after_setup'
    if mini_batch_size != None:
        option_mini_batch_size = ' --mini_batch_size=%d' % mini_batch_size
    if num_epochs != None:
        option_num_epochs = ' --num_epochs=%d' % num_epochs
    if processes_per_model != None:
        option_processes_per_model = ' --procs_per_model=%d' % processes_per_model
    if lbann_errors != []:
        raise Exception('Invalid Usage: ' + ' , '.join(lbann_errors))
    command_lbann = '%s%s%s%s%s%s%s%s%s%s%s%s%s%s' % (
        executable, option_data_filedir, option_data_filedir_train,
        option_data_filename_train, option_data_filedir_test,
        option_data_filename_test, option_data_reader,
        option_data_reader_percent, option_exit_after_setup,
        option_mini_batch_size, option_model, option_num_epochs,
        option_optimizer, option_processes_per_model)

    # Create output command
    command_output = ''
    if output_file_name != None:
        command_output = ' > %s' % output_file_name

    t = (command_allocate, command_run, command_lbann, command_output)

    if return_tuple:
        return t
    else:
        return '%s%s %s%s' % t
