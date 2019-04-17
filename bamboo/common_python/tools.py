import pytest
import math, os, re


def check_list(substrings, strings):
    errors = []
    for string in strings:
        for substring in substrings:
            if (string != None) and (substring in string):
                errors.append('%s contains %s' % (string, substring))
    return errors


def get_command(cluster,
                executable,
                num_nodes=None,
                partition=None,
                time_limit=None,
                num_processes=None,
                dir_name=None,
                data_filedir_default=None,
                data_filedir_train_default=None,
                data_filename_train_default=None,
                data_filedir_test_default=None,
                data_filename_test_default=None,
                data_reader_name=None,
                data_reader_path=None,
                data_reader_percent=None,
                exit_after_setup=False,
                mini_batch_size=None,
                model_folder=None,
                model_name=None,
                model_path=None,
                num_epochs=None,
                optimizer_name=None,
                optimizer_path=None,
                processes_per_model=None,
                ckpt_dir=None,
                output_file_name=None,
                error_file_name=None,
                return_tuple=False,
                check_executable_existence=True,
                skip_no_exe=True):
    # Check parameters for black-listed characters like semi-colons that
    # would terminate the command and allow for an extra command
    blacklist = [';', '--']
    strings = [partition, dir_name, data_filedir_default,
               data_filedir_train_default,
               data_filename_train_default, data_filedir_test_default,
               data_filename_test_default, data_reader_name, data_reader_path,
               model_folder, model_name, model_path, optimizer_name,
               optimizer_path, output_file_name, error_file_name]
    invalid_character_errors = check_list(blacklist, strings)
    if invalid_character_errors != []:
        raise Exception('Invalid character(s): %s' % ' , '.join(
            invalid_character_errors))

    # Never give lbannusr an allocation for over 12 hours though.
    strict_time_limit = 60*6  # 6 hours.
    if time_limit > strict_time_limit:
        time_limit = strict_time_limit

    # Check executable existence
    if check_executable_existence:
        process_executable_existence(executable, skip_no_exe)

    # Determine scheduler
    if cluster in ['catalyst', 'pascal', 'quartz', 'surface']:
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
        if os.getenv('SLURM_JOB_NUM_NODES') == None:
            command_allocate = 'salloc'
            option_num_nodes = ''
            option_partition = ''
            option_time_limit = ''
            if num_nodes is not None:
                # --nodes=<minnodes[-maxnodes]> =>
                # Request that a minimum of minnodes nodes be allocated to this
                # job. A maximum node count may also be specified with
                # maxnodes.
                option_num_nodes = ' --nodes=%d' % num_nodes
            if partition is not None:
                # Surface does not have pdebug, so switch to pbatch
                if (cluster in ['surface', 'pascal']) and \
                        (partition == 'pdebug'):
                    partition = 'pbatch'
                # --partition => Request a specific partition for the resource
                # allocation.
                option_partition = ' --partition=%s' % partition
            if time_limit is not None:
                # --time => Set a limit on the total run time of the job
                # allocation.
                # Time limit in minutes
                option_time_limit = ' --time=%d' % time_limit
            command_allocate = '%s%s%s%s' % (
                command_allocate, option_num_nodes, option_partition,
                option_time_limit)

        # Create run command
        if command_allocate == '':
            command_run = 'srun --mpibind=off'
        else:
            command_run = ' srun --mpibind=off'
        option_num_processes = ''
        if num_processes is not None:
            # --ntasks => Specify  the  number of tasks to run.
            # Number of processes to run => MPI Rank
            option_num_processes = ' --ntasks=%d' % num_processes
        command_run = '%s%s' % (command_run, option_num_processes)

    elif scheduler == 'lsf':
        # Create allocate command
        command_allocate = ''
        # Allocate a node if we don't have one already
        # Running the tests manually allows for already having a node allocated
        if os.getenv('LSB_HOSTS') is None:
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
            if num_processes is not None:
                # n => Submits a parallel job and specifies the number of
                # tasks in the job.
                option_num_processes = ' -n %d' % num_processes
                if (num_nodes is not None) and (num_nodes != 0):
                    # R => Runs the job on a host that meets the specified
                    # resource requirements.
                    option_processes_per_node = ' -R "span[ptile=%d]"' % int(
                        math.ceil(float(num_processes)/num_nodes))
            if partition is not None:
                # q => Submits the job to one of the specified queues.
                option_partition = ' -q %s' % partition
            if time_limit is not None:
                if cluster == 'ray':
                    max_ray_time = 480
                    if time_limit > max_ray_time:
                        time_limit = max_ray_time
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
        if num_processes is not None:
            # -np => Run this many copies of the program on the given nodes.
            option_num_processes = ' -np %d' % num_processes
            if (num_nodes is not None) and (num_nodes != 0):
                option_processes_per_node = ' -N %d' % int(
                    math.ceil(float(num_processes)/num_nodes))
        command_run = '%s%s%s' % (
            command_run, option_num_processes, option_processes_per_node)

    else:
        raise Exception('Unsupported Scheduler %s' % scheduler)

    # Create LBANN command
    option_ckpt_dir = ''
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
    if model_path is not None:
        # If model_folder and/or model_name are set, an exception will be
        # raised later.
        option_model = ' --model=%s' % model_path
    if data_reader_path is not None:
        # If data_reader_name is set, an exception will be raised later.
        option_data_reader = ' --reader=%s' % data_reader_path
    if optimizer_path is not None:
        # If optimizer_name is set, an exception will be raised later.
        option_optimizer_name = ' --optimizer=%s' % optimizer_path
    if dir_name is not None:
        if model_path is not None:
            if (model_folder is not None) or (model_name is not None):
                lbann_errors.append(
                    ('model_path is set but so is at least one of model'
                     ' folder and model_name'))
        else:
            if (model_folder is not None) and (model_name is not None):
                option_model = ' --model=%s/model_zoo/%s/model_%s.prototext' % (
                    dir_name, model_folder, model_name)
            elif model_folder is not None:
                lbann_errors.append('model_folder set but not model_name.')
            elif model_name is not None:
                lbann_errors.append('model_name set but not model_folder.')
        if data_reader_name is not None:
            if data_reader_path is not None:
                lbann_errors.append(('data_reader_path is set but so is'
                                     ' data_reader_name'))
            else:
                option_data_reader = ' --reader=%s/model_zoo/data_readers/data_reader_%s.prototext' % (dir_name, data_reader_name)
        if optimizer_name is not None:
            if optimizer_path is not None:
                lbann_errors.append(('optimizer_path is set but so is'
                                     ' optimizer_name'))
            else:
                option_optimizer = ' --optimizer=%s/model_zoo/optimizers/opt_%s.prototext' % (dir_name, optimizer_name)
        if (model_folder is None) and (model_name is None) and \
                (data_reader_name is None) and (optimizer_name is None):
            lbann_errors.append(
                ('dir_name set but none of model_folder, model_name,'
                 ' data_reader_name, optimizer_name are.'))
    elif (model_folder is not None) or (model_name is not None) or \
            (data_reader_name is not None) or (optimizer_name is not None):
        lbann_errors.append(
            ('dir_name is not set but at least one of model_folder,'
             ' model_name, data_reader_name, optimizer_name is.'))
    data_file_parameters = [data_filedir_train_default,
                            data_filename_train_default,
                            data_filedir_test_default,
                            data_filename_test_default]
    # Determine data file paths
    # If there is no regex match, then re.sub keeps the original string
    if data_filedir_default is not None:
        if cluster in ['catalyst', 'pascal', 'surface']:
            # option_data_filedir = data_filedir_default # lscratchh, presumably
            pass  # No need to pass in a parameter
        elif cluster == 'quartz':
            option_data_filedir = ' --data_filedir=%s' % re.sub(
                '[a-z]scratch[a-z]', 'lscratchh', data_filedir_default)
        elif cluster == 'ray':
            option_data_filedir = ' --data_filedir=%s' % re.sub(
                '[a-z]scratch[a-z]', 'gscratchr', data_filedir_default)
    elif None not in data_file_parameters:
        if cluster in ['catalyst', 'pascal', 'surface']:
            # option_data_filedir_train = data_filedir_train_default
            # option_data_filename_train = data_filename_train_default
            # option_data_filedir_test = data_filedir_test_default
            # option_data_filename_train = data_filename_test_default
            pass # No need to pass in a parameter
        elif cluster == 'quartz':
            option_data_filedir_train  = ' --data_filedir_train=%s'  % re.sub('[a-z]scratch[a-z]', 'lscratchh', data_filedir_train_default)
            option_data_filename_train = ' --data_filename_train=%s' % re.sub('[a-z]scratch[a-z]', 'lscratchh', data_filename_train_default)
            option_data_filedir_test   = ' --data_filedir_test=%s'   % re.sub('[a-z]scratch[a-z]', 'lscratchh', data_filedir_test_default)
            option_data_filename_train = ' --data_filename_test=%s'  % re.sub('[a-z]scratch[a-z]', 'lscratchh', data_filename_test_default)
        elif cluster == 'ray':
            option_data_filedir_train  = ' --data_filedir_train=%s'  % re.sub('[a-z]scratch[a-z]', 'gscratchr', data_filedir_train_default)
            option_data_filename_train = ' --data_filename_train=%s' % re.sub('[a-z]scratch[a-z]', 'gscratchr', data_filename_train_default)
            option_data_filedir_test   = ' --data_filedir_test=%s'   % re.sub('[a-z]scratch[a-z]', 'gscratchr', data_filedir_test_default)
            option_data_filename_test = ' --data_filename_test=%s'  % re.sub('[a-z]scratch[a-z]', 'gscratchr', data_filename_test_default)
    if (data_reader_name is not None) or (data_reader_path is not None):
        if data_filedir_default is not None:
            # If any are not None
            if data_file_parameters != [None, None, None, None]:
                lbann_errors.append(
                    ('data_fildir_default set but so is at least one of'
                     ' [data_filedir_train_default, data_filename_train'
                     '_default, data_filedir_test_default,'
                     ' data_filename_test_default]'))
            # else: only data_filedir_default is set
        else:
            # if None in data_file_parameters: # If any are None
            if data_file_parameters == [None, None, None, None]: # If all are None
                lbann_errors.append(
                    ('data_reader_name or data_reader_path is set but not'
                     ' data_filedir_default. If a data reader is provided,'
                     ' the default filedir must be set. This allows for'
                     ' determining what the filedir should be on each'
                     ' cluster. Alternatively, some or all of'
                     ' [data_filedir_train_default, data_filename_train'
                     '_default, data_filedir_test_default, data_filename'
                     '_test_default] can be set.'))
            # else: no data_file parameters are set
    else:
        if data_filedir_default is not None:
            lbann_errors.append(
                ('data_filedir_default set but neither data_reader_name'
                 ' or data_reader_path are.'))
        elif filter(lambda x: x is not None, data_file_parameters) != []:
            # If the list of non-None data_file parameters is not empty
            lbann_errors.append(
                ('At least one of [data_filedir_train_default, data_filename'
                 '_train_default, data_filedir_test_default, data_filename'
                 '_test_default] is set, but neither data_reader_name or'
                 ' data_reader_path are.'))
        # else: no conflicts
    if data_reader_percent is not None:
        option_data_reader_percent = ' --data_reader_percent=%f' % data_reader_percent
    if exit_after_setup:
        option_exit_after_setup = ' --exit_after_setup'
    if mini_batch_size is not None:
        option_mini_batch_size = ' --mini_batch_size=%d' % mini_batch_size
    if num_epochs is not None:
        option_num_epochs = ' --num_epochs=%d' % num_epochs
    if processes_per_model is not None:
        option_processes_per_model = ' --procs_per_model=%d' % processes_per_model
    if ckpt_dir is not None:
        option_ckpt_dir = ' --ckpt_dir=%s' % ckpt_dir
    if lbann_errors != []:
        print('lbann_errors={lbann_errors}.'.format(lbann_errors=lbann_errors))
        raise Exception('Invalid Usage: ' + ' , '.join(lbann_errors))
    command_lbann = '%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s' % (
        executable, option_ckpt_dir, option_data_filedir,
        option_data_filedir_train, option_data_filename_train,
        option_data_filedir_test, option_data_filename_test,
        option_data_reader, option_data_reader_percent,
        option_exit_after_setup, option_mini_batch_size,
        option_model, option_num_epochs, option_optimizer,
        option_processes_per_model)

    # Create redirect command
    command_output = ''
    command_error = ''
    if output_file_name is not None:
        command_output = ' > %s' % output_file_name
    if error_file_name is not None:
        command_error = ' 2> %s' % error_file_name
    command_redirect = '%s%s' % (command_output, command_error)

    t = (command_allocate, command_run, command_lbann, command_redirect)

    if return_tuple:
        print('command_tuple=' + str(t))
        return t
    else:
        command_string = '%s%s %s%s' % t
        print('command_string=' + command_string)
        return command_string


def process_executable_existence(executable, skip_no_exe=True):
    executable_exists = os.path.exists(executable)
    if not executable_exists:
        error_string = 'Executable does not exist: %s' % executable
        if skip_no_exe:
            pytest.skip(error_string)
        else:
            raise Exception(error_string)


def get_spack_exes(default_dirname, cluster):
    exes = {}

    exes['clang4'] = '%s/bamboo/compiler_tests/builds/%s_clang-4.0.0_rel/build/model_zoo/lbann' % (default_dirname, cluster)
    exes['gcc4'] = '%s/bamboo/compiler_tests/builds/%s_gcc-4.9.3_rel/build/model_zoo/lbann' % (default_dirname, cluster)
    exes['gcc7'] = '%s/bamboo/compiler_tests/builds/%s_gcc-7.1.0_rel/build/model_zoo/lbann' % (default_dirname, cluster)
    exes['intel18'] = '%s/bamboo/compiler_tests/builds/%s_intel-18.0.0_rel/build/model_zoo/lbann' % (default_dirname, cluster)

    exes['clang4_debug'] = '%s/bamboo/compiler_tests/builds/%s_clang-4.0.0_debug/build/model_zoo/lbann' % (default_dirname, cluster)
    exes['gcc4_debug'] = '%s/bamboo/compiler_tests/builds/%s_gcc-4.9.3_debug/build/model_zoo/lbann' % (default_dirname, cluster)
    exes['gcc7_debug'] = '%s/bamboo/compiler_tests/builds/%s_gcc-7.1.0_debug/build/model_zoo/lbann' % (default_dirname, cluster)
    exes['intel18_debug'] = '%s/bamboo/compiler_tests/builds/%s_intel-18.0.0_debug/build/model_zoo/lbann' % (default_dirname, cluster)

    return exes


def get_default_exes(default_dirname, cluster):
    exes = get_spack_exes(default_dirname, cluster)
    # Use build script as a backup if the Spack build doesn't work.
    if not os.path.exists(exes['clang4']):
        exes['clang4'] = '%s/build/clang.Release.%s.llnl.gov/install/bin/lbann' % (default_dirname, cluster)
    if not os.path.exists(exes['gcc7']):
        exes['gcc7'] = '%s/build/gnu.Release.%s.llnl.gov/install/bin/lbann' % (default_dirname, cluster)
    if not os.path.exists(exes['intel18']):
        exes['intel18'] = '%s/build/intel.Release.%s.llnl.gov/install/bin/lbann' % (default_dirname, cluster)

    if not os.path.exists(exes['clang4_debug']):
        exes['clang4_debug'] = '%s/build/clang.Debug.%s.llnl.gov/install/bin/lbann' % (default_dirname, cluster)
    if not os.path.exists(exes['gcc7_debug']):
        exes['gcc7_debug'] = '%s/build/gnu.Debug.%s.llnl.gov/install/bin/lbann' % (default_dirname, cluster)
    if not os.path.exists(exes['intel18_debug']):
        exes['intel18_debug'] = '%s/build/intel.Debug.%s.llnl.gov/install/bin/lbann' % (default_dirname, cluster)

    default_exes = {}
    default_exes['default'] = '%s/build/gnu.Release.%s.llnl.gov/install/bin/lbann' % (default_dirname, cluster)
    if cluster in ['catalyst', 'quartz', 'pascal']:
        # x86_cpu - catalyst, quartz
        # x86_gpu_pascal - pascal
        default_exes['clang4'] = exes['clang4']
        default_exes['gcc4'] = exes['gcc4']
        default_exes['gcc7'] = exes['gcc7']
        default_exes['intel18'] = exes['intel18']

        default_exes['clang4_debug'] = exes['clang4_debug']
        default_exes['gcc4_debug'] = exes['gcc4_debug']
        default_exes['gcc7_debug'] = exes['gcc7_debug']
        default_exes['intel18_debug'] = exes['intel18_debug']
    elif cluster in ['surface']:
        # x86_gpu - surface
        default_exes['gcc4'] = exes['gcc4']
        default_exes['gcc4_debug'] = exes['gcc4_debug']

    print('default_exes={d}'.format(d=default_exes))
    return default_exes
