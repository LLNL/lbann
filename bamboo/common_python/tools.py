import collections.abc
import copy
import math
import os
import re
import socket
import sys
import numpy as np
import pytest
import shutil
import subprocess
from filecmp import cmp

def check_list(substrings, strings):
    errors = []
    for string in strings:
        for substring in substrings:
            if (string is not None) and (isinstance(string, str)) and (substring in string):
                errors.append('%s contains %s' % (string, substring))
    return errors


def get_command(cluster,
                # Allocation/Run Parameters
                num_nodes=None,
                num_processes=None,
                partition=None,
                time_limit=None,
                # LBANN Parameters
                ckpt_dir=None,
                disable_cuda=None,
                dir_name=None,
                sample_list_train_default=None,
                sample_list_test_default=None,
                data_filedir_default=None,
                data_filedir_train_default=None,
                data_filename_train_default=None,
                data_filedir_test_default=None,
                data_filename_test_default=None,
                data_reader_name=None,
                data_reader_path=None,
                data_reader_percent=None,
                exit_after_setup=False,
                metadata=None,
                mini_batch_size=None,
                model_folder=None,
                model_name=None,
                model_path=None,
                num_epochs=None,
                optimizer_name=None,
                optimizer_path=None,
                processes_per_model=None,
                restart_dir=None,
                extra_lbann_flags=None,
                # Error/Output Redirect
                error_file_name=None,
                output_file_name=None,
                # Misc. Parameters
                check_executable_existence=True,
                return_tuple=False,
                skip_no_exe=True,
                weekly=False):
    # Check parameters for black-listed characters like semi-colons that
    # would terminate the command and allow for an extra command
    blacklist = [';', '--']
    strings = [
        cluster,
        # Allocation/Run Parameters
        num_nodes, num_processes, partition, time_limit,
        # LBANN Parameters
        ckpt_dir, dir_name,
        sample_list_train_default, sample_list_test_default,
        data_filedir_default, data_filedir_train_default,
        data_filename_train_default, data_filedir_test_default,
        data_filename_test_default, data_reader_name, data_reader_path,
        data_reader_percent, exit_after_setup, metadata, mini_batch_size,
        model_folder, model_name, model_path, num_epochs, optimizer_name,
        optimizer_path, processes_per_model, restart_dir,
        # Error/Output Redirect
        error_file_name, output_file_name,
        # Misc. Parameters
        check_executable_existence, return_tuple,  skip_no_exe, weekly
    ]
    executable = shutil.which('lbann')
    lbann_errors = []
    if extra_lbann_flags is not None:
        if not isinstance(extra_lbann_flags, dict):
            lbann_errors.append(
                ('extra_lbann_flags must be a dict e.g. `{flag :'
                 ' None, flag: 4}`. Use `None` if a flag has no value attached '
                 'to it.'))
        else:
            strings += list(extra_lbann_flags.keys())
            strings += list(extra_lbann_flags.values())
    invalid_character_errors = check_list(blacklist, strings)
    if invalid_character_errors != []:
        raise Exception('Invalid character(s): %s' % ' , '.join(
            invalid_character_errors))

    DEFAULT_TIME = 35
    MAX_TIME = 360  # 6 hours.
    if time_limit is None:
        if weekly:
            time_limit = MAX_TIME
        else:
            time_limit = DEFAULT_TIME
    if time_limit > MAX_TIME:
        time_limit = MAX_TIME

    # Determine scheduler
    if cluster in ['catalyst', 'corona', 'pascal']:
        scheduler = 'slurm'
    elif cluster in ['lassen', 'ray']:
        scheduler = 'lsf'
    else:
        raise Exception('Unsupported Cluster: %s' % cluster)

    # Description of command line options are from the appropriate command's
    # man pages
    if scheduler == 'slurm':
        # Create allocate command
        command_allocate = ''
        # Allocate nodes only if we don't already have an allocation.
        if os.getenv('SLURM_JOB_NUM_NODES') is None:
            print('Allocating slurm nodes.')
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
                # If cluster doesn't have pdebug switch to pbatch.
                if (cluster in ['pascal']) and \
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
        else:
            print('slurm nodes already allocated.')

        # Create run command
        if command_allocate == '':
            space = ''
        else:
            space = ' '
        command_run = '{s}srun --mpibind=off --time={t}'.format(
            s=space, t=time_limit)
        option_num_processes = ''
        if num_processes is not None:
            # --ntasks => Specify  the  number of tasks to run.
            # Number of processes to run => MPI Rank
            option_num_processes = ' --ntasks=%d' % num_processes
        command_run = '%s%s' % (command_run, option_num_processes)

    elif scheduler == 'lsf':
        # Create allocate command
        command_allocate = ''
        # Allocate nodes only if we don't already have an allocation.
        if (os.getenv('LSB_HOSTS') is None) and (os.getenv('LSB_JOBID') is None):
            print('Allocating lsf nodes.')
            command_allocate = 'bsub'
            option_exclusive = ''
            if cluster != 'lassen':
                # x => Puts the host running your job into exclusive execution
                # mode.
                option_exclusive = ' -x'
            # G=> For fairshare scheduling. Associates the job with the
            # specified group.
            option_group = ' -G guests'
            # Is => Submits an interactive job and creates a pseudo-terminal
            # with shell mode when the job starts.
            option_interactive = ' -Is'
            option_num_nodes = ''
            option_num_processes = ''
            option_partition = ''
            option_processes_per_node = ''
            option_time_limit = ''
            if cluster == 'lassen':
                option_num_nodes = ' -nnodes {n}'.format(n=num_nodes)
            elif num_processes is not None:
                # n => Submits a parallel job and specifies the number of
                # tasks in the job.
                option_num_processes = ' -n %d' % num_processes
                if (num_nodes is not None) and (num_nodes != 0):
                    # R => Runs the job on a host that meets the specified
                    # resource requirements.
                    option_processes_per_node = ' -R "span[ptile=%d]"' % int(
                        math.ceil(float(num_processes) / num_nodes))
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
            command_allocate = '%s%s%s%s%s%s%s%s%s' % (
                command_allocate, option_exclusive, option_group,
                option_interactive, option_num_processes, option_partition,
                option_num_nodes, option_processes_per_node, option_time_limit)
        else:
            print('lsf nodes already allocated.')

        # Create run command
        if command_allocate == '':
            space = ''
        else:
            space = ' '
        if cluster == 'lassen':
            # Cannot specify time limit for jsrun.
            command_run = '{s}jsrun'.format(s=space)
        else:
            command_run = '{s}mpirun --timeout {t}'.format(s=space, t=time_limit*60)
        option_bind = ''
        option_cpu_per_resource = ''
        option_gpu_per_resource = ''
        option_launch_distribution = ''
        option_num_processes = ''
        option_processes_per_node = ''
        option_resources_per_host = ''
        option_tasks_per_resource = ''
        if num_processes is not None:
            if cluster == 'lassen':
                option_bind = ' -b "packed:8"'
                option_cpu_per_resource = ' --cpu_per_rs ALL_CPUS'
                option_gpu_per_resource = ' --gpu_per_rs ALL_GPUS'
                option_launch_distribution = ' --launch_distribution packed'
                # By default there should be 4 prcesses per node (especially when using GPUs)
                resources_per_node = 4
                if disable_cuda:
                    # When CUDA is disabled, allow the number of resources per node to be overridden
                    resources_per_node = math.ceil(float(num_processes)/num_nodes)
                # Avoid `nrs (32) should not be greater than rs_per_host (1) * number of servers available (16).`
                if num_nodes is None:
                    num_nodes = math.ceil(float(num_processes)/resources_per_node)
                # The "option_num_processes" is a misnomer for the LSF case. Rather than
                # changing the rest of the code, set it to be the number of nodes. Within
                # JSRUN, the correct number of processes will be obtained when combined
                # with "option_tasks_per_resource".
                option_num_processes = ' --nrs {n}'.format(n=num_nodes)
                option_resources_per_host = ' --rs_per_host 1'
                option_tasks_per_resource = ' --tasks_per_rs {n}'.format(n=resources_per_node)
                if (num_processes%num_nodes) is not 0:
                    raise Exception('num_processes %s, is not divisible by num_nodes %d'
                                    % (num_processes, num_nodes))

            else:
                # -np => Run this many copies of the program on the given nodes.
                option_num_processes = ' -np %d' % num_processes
                if (num_nodes is not None) and (num_nodes != 0):
                    processes_per_node = int(
                        math.ceil(float(num_processes)/num_nodes))
                    option_processes_per_node = ' -N %d' % processes_per_node
        command_run = '%s%s%s%s%s%s%s%s%s' % (
            command_run, option_bind, option_cpu_per_resource,
            option_gpu_per_resource, option_launch_distribution,
            option_num_processes, option_processes_per_node,
            option_resources_per_host, option_tasks_per_resource)

    else:
        raise Exception('Unsupported Scheduler %s' % scheduler)

    # Create LBANN command
    option_ckpt_dir = ''
    option_disable_cuda = ''
    option_data_filedir = ''
    option_sample_list_train = ''
    option_sample_list_test = ''
    option_data_filedir_train = ''
    option_data_filename_train = ''
    option_data_filedir_test = ''
    option_data_filename_test = ''
    option_data_reader = ''
    option_data_reader_percent = ''
    option_exit_after_setup = ''
    option_metadata = ''
    option_mini_batch_size = ''
    option_model = ''
    option_num_epochs = ''
    option_optimizer = ''
    option_processes_per_model = ''
    option_restart_dir = ''
    if model_path is not None:
        # If model_folder and/or model_name are set, an exception will be
        # raised later.
        option_model = ' --model=%s' % model_path
    if data_reader_path is not None:
        # If data_reader_name is set, an exception will be raised later.
        option_data_reader = ' --reader=%s' % data_reader_path
    if optimizer_path is not None:
        # If optimizer_name is also set, an exception will be raised later.
        option_optimizer = ' --optimizer=%s' % optimizer_path
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
    sample_list_parameters = [sample_list_train_default,
                              sample_list_test_default]
    # Determine data file paths
    # If there is no regex match, then re.sub keeps the original string
    if data_filedir_default is not None:
        if cluster in ['catalyst', 'corona', 'pascal',]:
            # option_data_filedir = data_filedir_default # lscratchh, presumably
            pass  # No need to pass in a parameter
        elif cluster == 'lassen':
            option_data_filedir = ' --data_filedir=%s' % re.sub(
                '[a-z]scratch[a-z]', 'gpfs1', data_filedir_default)
        elif cluster == 'ray':
            option_data_filedir = ' --data_filedir=%s' % re.sub(
                '[a-z]scratch[a-z]', 'gscratchr', data_filedir_default)
    elif not sample_list_parameters == [None, None]:
        if cluster in ['catalyst', 'corona', 'pascal',]:
            # option_data_filedir = data_filedir_default # lscratchh, presumably
            pass  # No need to pass in a parameter
        elif cluster == 'lassen':
            option_sample_list_train = ' --sample_list_train=%s' % re.sub(
                'lustre[0-9]', 'gpfs1', sample_list_train_default)
            option_sample_list_test = ' --sample_list_test=%s' % re.sub(
                'lustre[0-9]', 'gpfs1', sample_list_test_default)
        elif cluster == 'ray':
            option_sample_list_train = ' --sample_list_train=%s' % re.sub(
                'lustre[0-9]', 'gscratchr', sample_list_train_default)
            option_sample_list_test = ' --sample_list_test=%s' % re.sub(
                'lustre[0-9]', 'gscratchr', sample_list_test_default)
    elif not data_file_parameters == [None, None, None, None]:
        # Any of the data_file_parameters has a non-None value.
        if cluster in ['catalyst', 'corona', 'pascal']:
            # option_data_filedir_train = data_filedir_train_default
            # option_data_filename_train = data_filename_train_default
            # option_data_filedir_test = data_filedir_test_default
            # option_data_filename_train = data_filename_test_default
            pass  # No need to pass in a parameter
        elif cluster == 'lassen':
            if data_filedir_train_default is not None:
                option_data_filedir_train  = ' --data_filedir_train=%s'  % re.sub('[a-z]scratch[a-z]', 'gpfs1', data_filedir_train_default)
            if data_filename_train_default is not None:
                filename_train = re.sub(
                    '[a-z]scratch[a-z]', 'gpfs1', data_filename_train_default)
                filename_train = re.sub(
                    'labels', 'original/labels', filename_train)
                option_data_filename_train = ' --data_filename_train=%s' % filename_train
            if data_filedir_test_default is not None:
                option_data_filedir_test   = ' --data_filedir_test=%s'   % re.sub('[a-z]scratch[a-z]', 'gpfs1', data_filedir_test_default)
            if data_filename_test_default is not None:
                filename_test = re.sub(
                    '[a-z]scratch[a-z]', 'gpfs1', data_filename_test_default)
                filename_test = re.sub(
                    'labels', 'original/labels', filename_test)
                option_data_filename_test  = ' --data_filename_test=%s'  % filename_test
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
            if data_file_parameters == [None, None, None, None] and sample_list_parameters == [None, None]: # If all are None
                if data_reader_name != 'synthetic':
                    lbann_errors.append(
                        ('data_reader_name or data_reader_path is set but not'
                         ' sample_list_[train|test]_default or'
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
        elif list(filter(lambda x: x is not None, data_file_parameters)) != []:
            # If the list of non-None data_file parameters is not empty
            lbann_errors.append(
                ('At least one of [data_filedir_train_default, data_filename'
                 '_train_default, data_filedir_test_default, data_filename'
                 '_test_default] is set, but neither data_reader_name or'
                 ' data_reader_path are.'))
        # else: no conflicts
    if data_reader_percent != "prototext":
        if data_reader_percent is not None:

            # If data_reader_percent is not None, then it will override `weekly`.
            # If it is None however, we choose its value based on `weekly`.
            try:
                data_reader_percent = float(data_reader_percent)

            except ValueError:
                lbann_errors.append(
                    'data_reader_percent={d} is not a float.'.format(
                        d=data_reader_percent))
        elif weekly:
            data_reader_percent = 1.00
        else:
            # Nightly
            data_reader_percent = 0.10
        option_data_reader_percent = ' --data_reader_percent={d}'.format(
            d=data_reader_percent)
    # else: use the data reader's value
    if exit_after_setup:
        option_exit_after_setup = ' --exit_after_setup'
    if metadata is not None:
        option_metadata = ' --metadata={d}/{m}'.format(d=dir_name, m=metadata)
    if mini_batch_size is not None:
        option_mini_batch_size = ' --mini_batch_size=%d' % mini_batch_size
    if num_epochs is not None:
        option_num_epochs = ' --num_epochs=%d' % num_epochs
    if processes_per_model is not None:
        option_processes_per_model = ' --procs_per_model=%d' % processes_per_model
    if ckpt_dir is not None:
        option_ckpt_dir = ' --ckpt_dir=%s' % ckpt_dir
    if restart_dir is not None:
        option_restart_dir = ' --restart_dir=%s' % restart_dir
    if disable_cuda is not None:
        option_disable_cuda = ' --disable_cuda=%d' % int(bool(disable_cuda))
    extra_options = ''
    if extra_lbann_flags is not None:
        # If extra_lbann_flags is not a dict, then we have already appended
        # this error to lbann_errors.
        if isinstance(extra_lbann_flags, dict):
            # See `lbann --help` or src/proto/proto_common.cpp
            # Commented out flags already have their own parameters
            # in this function.
            allowed_flags = [
                # 'model',
                # 'optimizer',
                # 'reader',
                # 'metadata',

                # General:
                # 'mini_batch_size',
                # 'num_epochs',
                'hydrogen_block_size',
                'procs_per_trainer',
                'num_parallel_readers',
                'num_io_threads',
                'serialize_io',
                'disable_background_io_activity',
                #'disable_cuda',
                'random_seed',
                'objective_function',
                'data_layout',
                'print_affinity',
                'use_data_store',
                'preload_data_store',
                'super_node',
                'write_sample_list',
                'ltfb_verbose',
                'ckpt_dir',
                #'restart_dir',
                'restart_dir_is_fullpath',

                # DataReaders:
                # 'data_filedir',
                # 'data_filedir_train',
                # 'data_filedir_test',
                # 'data_filename_train',
                # 'data_filename_test',
                'sample_list_train',
                'sample_list_test',
                'label_filename_train',
                'label_filename_test',
                # 'data_reader_percent',
                'share_testing_data_readers',

                # Callbacks:
                'image_dir',
                'no_im_comm',

                # Not listed by `lbann --help`:
                # 'exit_after_setup',
                # 'procs_per_model'
            ]
            for flag, value in sorted(extra_lbann_flags.items()):
                if flag in allowed_flags:
                    if value is not None:
                        extra_options += ' --{f}={v}'.format(f=flag, v=value)
                    else:
                        extra_options += ' --{f}'.format(f=flag)
                else:
                    s = ('extra_lbann_flags includes invalid flag={f}.'
                         ' Flags must be in {flags}.').format(
                        f=flag, flags=allowed_flags)
                    lbann_errors.append(s)
    if lbann_errors != []:
        print('lbann_errors={lbann_errors}.'.format(lbann_errors=lbann_errors))
        raise Exception('Invalid Usage: ' + ' , '.join(lbann_errors))
    command_lbann = '%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s' % (
        executable, option_ckpt_dir, option_disable_cuda,
        option_sample_list_train, option_sample_list_test,
        option_data_filedir,
        option_data_filedir_train, option_data_filename_train,
        option_data_filedir_test, option_data_filename_test,
        option_data_reader, option_data_reader_percent,
        option_exit_after_setup, option_metadata, option_mini_batch_size,
        option_model, option_num_epochs, option_optimizer,
        option_processes_per_model, option_restart_dir, extra_options)

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


def get_error_line(error_file_name):
    with open(error_file_name, 'r') as error_file:
        error_line = ''
        previous_line = ''
        for line in error_file:
            if ('ERROR' in line) or ('LBANN error' in line) or \
                    ('Error:' in line) or \
                    ('Expired or invalid job' in line) or \
                    ('Segmentation fault (core dumped)' in line) or \
                    ('Relinquishing job allocation' in line):
                error_line = line
                break
            elif ('Stack trace:' in line) or \
                    ('Error is not recoverable: exiting now' in line):
                error_line = previous_line
                break
            else:
                previous_line = line
    return error_line


def assert_success(return_code, error_file_name):
    if return_code != 0:
        error_line = get_error_line(error_file_name)
        raise AssertionError(
            'return_code={rc}\n{el}\nSee {efn}'.format(
                rc=return_code, el=error_line, efn=error_file_name))


def assert_failure(return_code, expected_error, error_file_name):
    if return_code == 0:
        raise AssertionError(
            'return_code={rc}\nSuccess when expecting failure.\nSee {efn}'.format(
                rc=return_code, efn=error_file_name))
    with open(error_file_name, 'r') as error_file:
        for line in error_file:
            if expected_error in line:
                return True
    # If we're at this point, then we know the test did not succeed,
    # but we didn't get the expected error.
    actual_error = get_error_line(error_file_name)
    raise AssertionError(
        'return_code={rc}\nFailed with error different than expected.\nactual_error={ae}\nexpected_error={ee}\nSee {efn}'.format(
            rc=return_code, ae=actual_error, ee=expected_error,
            efn=error_file_name))


def create_tests(setup_func,
                 test_file,
                 test_name_base=None,
                 **kwargs):
    """Create functions that can interact with PyTest

    This function creates tests that involve running an LBANN
    experiment with the Python frontend. `setup_func` should be a
    function that takes in the LBANN Python module and outputs objects
    for an LBANN experiment. A test succeeds if LBANN runs and exits
    with an exit code of 0, and fails otherwise.

    PyTest detects tests by loading in a Python script and looking for
    functions prefixed with 'test_'. After you call this function
    within a script to generate test functions, make sure to add the
    test functions to the script's scope. For example:

        _test_funcs = tools.create_tests(setup_func, __file__)
        for t in _test_funcs:
            globals()[t.__name__] = t

    Args:
        setup_func (function): Sets up an LBANN experiment using the
            Python frontend. It takes in the LBANN Python module as
            input and returns a `(lbann.Trainer, lbann.Model,
            lbann.reader_pb2.DataReader, lbann.Optimizer)`.
        test_file (str): Python script being run by PyTest. In most
            cases, use `__file__`.
        test_name (str, optional): Descriptive name (default: test
            file name with '.py' removed).
        **kwargs: Keyword arguments to pass into
            `lbann.contrib.launcher.run`.

    Returns:
        Iterable of function: Tests that can interact with PyTest.
            Each function returns a dict containing log files and
            other output data.

    """

    # Make sure test name is valid
    test_file = os.path.realpath(test_file)
    if not test_name_base:
        # Create test name by removing '.py' from file name
        test_name_base = os.path.splitext(os.path.basename(test_file))[0]
    if not re.match('^test_.', test_name_base):
        # Make sure test name is prefixed with 'test_'
        test_name_base = 'test_' + test_name_base

    def test_func(cluster, dirname):
        """Function that can interact with PyTest.

        Returns a dict containing log files and other output data.

        """
        test_name = '{}'.format(test_name_base)

        # Load LBANN Python frontend
        import lbann
        import lbann.contrib.launcher

        # Setup LBANN experiment
        trainer, model, data_reader, optimizer = setup_func(lbann)

        # Configure kwargs to LBANN launcher
        _kwargs = copy.deepcopy(kwargs)
        if 'work_dir' not in _kwargs:
            _kwargs['work_dir'] = os.path.join(os.path.dirname(test_file),
                                               'experiments',
                                               test_name)

        if 'skip_clusters' in _kwargs:
            if cluster in _kwargs['skip_clusters']:
                e = "test \"%s\" not supported on cluster \"%s\"" % (test_name, cluster)
                print('Skip - ' + e)
                pytest.skip(e)
            _kwargs.remove("skip_clusters")

        # If the user provided a suffix for the work directory, append it
        if 'work_subdir' in _kwargs:
            _kwargs['work_dir'] = os.path.join(_kwargs['work_dir'], _kwargs['work_subdir'])
            del _kwargs['work_subdir']

        # Delete the work directory
        if os.path.isdir(_kwargs['work_dir']):
            shutil.rmtree(_kwargs['work_dir'])

        if 'job_name' not in _kwargs:
            _kwargs['job_name'] = f'lbann_{test_name}'
        if 'overwrite_script' not in _kwargs:
            _kwargs['overwrite_script'] = True

        # Run LBANN
        work_dir = _kwargs['work_dir']
        stdout_log_file = os.path.join(work_dir, 'out.log')
        stderr_log_file = os.path.join(work_dir, 'err.log')
        return_code = lbann.contrib.launcher.run(
            trainer=trainer,
            model=model,
            data_reader=data_reader,
            optimizer=optimizer,
            **_kwargs,
        )
        assert_success(return_code, stderr_log_file)
        return {
            'return_code': return_code,
            'work_dir': work_dir,
            'stdout_log_file': stdout_log_file,
            'stderr_log_file': stderr_log_file,
        }

    # Specific test functions name
    test_func.__name__ = test_name_base

    return (
        test_func,
    )


def create_python_data_reader(lbann,
                              file_name,
                              sample_function_name,
                              num_samples_function_name,
                              sample_dims_function_name,
                              execution_mode):
    """Create protobuf message for Python data reader

    A Python data reader gets data by importing a Python module and
    calling functions in its scope.

    Args:
        lbann (module): Module for LBANN Python frontend.
        file_name (str): Python file.
        sample_function_name (str): Function to get a data sample. It
            takes one integer argument for the sample index and
            returns an `Iterator` of `float`s.
        sample_dims_function_name (str): Function to get dimensions of
            a data sample. It takes no arguments and returns a
            `(int,)`.
        num_samples_function_name (str): Function to get number of
            data samples in data set. It takes no arguments and
            returns an `int`.
        execution_mode (str): 'train', 'validation', or 'test'

    """

    # Extract paths
    file_name = os.path.realpath(file_name)
    dir_name = os.path.dirname(file_name)
    module_name = os.path.splitext(os.path.basename(file_name))[0]

    # Construct protobuf message for data reader
    reader = lbann.reader_pb2.Reader()
    reader.name = 'python'
    reader.role = execution_mode
    reader.shuffle = False
    reader.percent_of_data_to_use = 1.0
    reader.python.module = module_name
    reader.python.module_dir = dir_name
    reader.python.sample_function = sample_function_name
    reader.python.num_samples_function = num_samples_function_name
    reader.python.sample_dims_function = sample_dims_function_name

    return reader


def numpy_l2norm2(x):
    """Square of L2 norm, computed with NumPy

    The computation is performed with 64-bit floats.

    """
    if x.dtype is not np.float64:
        x = x.astype(np.float64)
    x = x.reshape(-1)
    return np.inner(x, x)


def make_iterable(obj):
    """Convert to an iterable object

    Simply returns `obj` if it is alredy iterable. Otherwise returns a
    1-tuple containing `obj`.

    """
    if isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str):
        return obj
    else:
        return (obj,)


def str_list(it):
    """Convert an iterable object to a space-separated string"""
    return ' '.join([str(i) for i in make_iterable(it)])

# Define evaluation function
def collect_metrics_from_log_func(log_file, key):
    metrics = []
    with open(log_file) as f:
        for line in f:
            match = re.search(key + ' : ([0-9.]+)', line)
            if match:
                metrics.append(float(match.group(1)))
    return metrics

def compare_metrics(baseline_metrics, test_metrics):
    assert len(baseline_metrics) == len(test_metrics), \
        'baseline and test experiments did not run for same number of epochs'
    for i in range(len(baseline_metrics)):
        x = baseline_metrics[i]
        xhat = test_metrics[i]
        assert x == xhat, \
            'found discrepancy in metrics for baseline {b} and test {t}'.format(b=x, t=xhat)


# Perform a diff across a directoy where not all of the subdirectories will exist in
# the test directory.  Return a list of unchecked subdirectories, the running error code
# and the list of failed directories
def multidir_diff(baseline, test, fileList):
    tmpList = []
    err_msg = ""
    err = 0
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        d = os.path.basename(filePath)
        t = os.path.basename(os.path.dirname(filePath))
        c = os.path.join(test, t, d)
        if os.path.exists(c):
            ret = subprocess.run('diff -rq {baseline} {test}'.format(
                baseline=filePath, test=c), capture_output=True, shell=True, text=True)
            if ret.returncode != 0:
                err_msg += 'diff -rq {baseline} {test} failed {dt}\n'.format(
                    dt=ret.returncode, baseline=filePath, test=c)
                err_msg += ret.stdout
            err += ret.returncode
        else:
            tmpList.append(filePath)

    return tmpList, err, err_msg

# Perform a line by line difference of an xml file and look for any floating point values
# For each floating point value, check to see if it is close-enough and log a warning if it
# is within a threshhold.
def approx_diff_xml_files(file1, file2, rel_tol):
    f1 = open(file1, 'r')
    f2 = open(file2, 'r')
    files_differ = False
    diff_list = []
    near_diff_list = []
    for l1 in f1:
        l2 = next(f2)
        if l1 != l2:
            try:
                v1 = float(re.sub(r'\s*<\w*>(\S*)<\/\w*>\s*', r'\1', l1))
                v2 = float(re.sub(r'\s*<\w*>(\S*)<\/\w*>\s*', r'\1', l2))
                close = math.isclose(v1, v2, rel_tol=rel_tol, abs_tol=0.0)
                if not close:
                    err = ('lines: %s and %s differ: %.13f != %.13f (+/- %.1e)' % (l1.rstrip(), l2.rstrip(), v1, v2, rel_tol))
                    diff_list.append(err)
                    files_differ = True
                else:
                    warn = ('lines: %s and %s are close: %.13f ~= %.13f (+/- %.1e)' % (l1.rstrip(), l2.rstrip(), v1, v2, rel_tol))
                    near_diff_list.append(warn)
            except ValueError:
                # Non-numerical diff.
                err = ('lines: %s and %s differ' % (l1.rstrip(), l2.rstrip()))
                diff_list.append(err)
                files_differ = True
    return files_differ, diff_list, near_diff_list

# Given a recursive python diff from dircmp, perform a recursive exploration of any files
# with differences.  For files with differences, if check any XML files for approximate equivalence
# which can be seen in some of the floating point recorded values
def print_diff_files(dcmp):
    any_files_differ = False
    all_diffs = []
    all_warns = []
    for name in dcmp.diff_files:
        from pprint import pprint
        err = f'Files {os.path.join(dcmp.left, name)} and {os.path.join(dcmp.right, name)} differ'
        if re.search('.xml', name):
            files_differ, diff_list, warn_list = approx_diff_xml_files(
                os.path.join(dcmp.left, name), os.path.join(dcmp.right, name), 1e-6)
            if files_differ:
                any_files_differ = True
                all_diffs.append(err)
                for d in diff_list:
                    all_diffs.append(d)
            if len(warn_list) > 0:
                warn = f'Files {os.path.join(dcmp.left, name)} and {os.path.join(dcmp.right, name)} have a near difference'
                all_warns.append(warn)
                for w in warn_list:
                    all_warns.append(w)
        else:
            any_files_differ = True
            all_diffs.append(err)

    for sub_dcmp in dcmp.subdirs.values():
        files_differ, diff_list, warn_list = print_diff_files(sub_dcmp)
        if files_differ:
            any_files_differ = True
            for d in diff_list:
                all_diffs.append(d)
        for d in warn_list:
            all_warns.append(d)

    return any_files_differ, all_diffs, all_warns


def system(lbann):
    """Name of compute system."""
    compute_center = lbann.contrib.launcher.compute_center()
    if compute_center != "unknown":
        return getattr(lbann.contrib, compute_center).systems.system()
    else:
        return re.sub(r'\d+', '', socket.gethostname())


# Get the number of GPUs per compute node.
# Return 0 if the system is unknown.
def gpus_per_node(lbann):
    compute_center = lbann.contrib.launcher.compute_center()
    if compute_center != "unknown":
        return getattr(lbann.contrib, compute_center).systems.gpus_per_node()
    else:
        return 0


# Get the environment variables for Distconv.
def get_distconv_environment():
    # TODO: Use the default halo exchange and shuffle method. See https://github.com/LLNL/lbann/issues/1659
    return {"LBANN_DISTCONV_HALO_EXCHANGE": "AL",
            "LBANN_DISTCONV_TENSOR_SHUFFLER": "AL"}
