from __future__ import annotations
import itertools
import os
import shutil

import lbann
import lbann.launcher
import lbann.proto

def cartesian_hyperparameter_sweep(
        script,
        make_experiment,
        procs_per_trainer=1,
        **kwargs,
):
    """Train LBANN models with Cartesian product of hyperparameter values

    Models are evaluated independently by LBANN trainers. If the
    number of models is greater than the number of trainers, LBANN is
    run multiple times.

    Args:
        script (lbann.launcher.batch_script.BatchScript): Batch script
            manager with MPI job configuration info.
        make_experiment (function): Function that returns a tuple with
            an lbann.Model, lbann.Optimizer,
            lbann.reader_pb2.DataReader, and lbann.Trainer.
        procs_per_trainer (int): Number of MPI ranks in an LBANN
            trainer.
        **kwargs (list): Hyperparameter values. Each kwarg should be a
            list of values to pass to the corresponding kwarg in
            make_experiment.

    """

    # MPI launch configuration
    num_nodes = script.nodes
    procs_per_node = script.procs_per_node
    if procs_per_node % procs_per_trainer and procs_per_trainer % procs_per_node:
        raise RuntimeError(
            f'Trainer size ({procs_per_trainer}) and '
            f'number of MPI ranks per node ({procs_per_node}) '
            'are not multiples of each other')
    num_trainers = (num_nodes * procs_per_node) // procs_per_trainer
    num_nodes = (procs_per_trainer * num_trainers) // procs_per_node
    work_dir = script.work_dir

    # Iterate through Cartesian product of hyperparameter values
    # Note: Export Protobuf message for each configuration and add
    # LBANN invocation as needed.
    arg_keys = list(kwargs.keys())
    arg_values = list(kwargs.values())
    configs_file = open(os.path.join(work_dir, 'hyperparameters.csv'), 'w')
    configs_file.write('config_id,run_id,trainer_id,')
    configs_file.write(','.join(arg_keys))
    configs_file.write('\n')
    for config_id, values in enumerate(itertools.product(*arg_values)):

        # Construct experiment
        args = { key : values[i] for i, key in enumerate(arg_keys) }
        model, optimizer, reader, trainer = make_experiment(**args)

        # Export Protobuf file
        run_id = config_id // num_trainers
        trainer_id = config_id % num_trainers
        lbann.proto.save_prototext(
            os.path.join(work_dir, f'run{run_id}.trainer{trainer_id}'),
            model=model,
            optimizer=optimizer,
            data_reader=reader,
            trainer=trainer)

        # Write hyperparameters to file
        configs_file.write(f'{config_id},{run_id},{trainer_id},')
        configs_file.write(','.join(str(val) for val in values))
        configs_file.write('\n')

        # Invocation to launch LBANN
        if trainer_id == num_trainers-1:
            command = [
                lbann.lbann_exe(),
                f'--procs_per_trainer={procs_per_trainer}',
                '--generate_multi_proto',
                f'--prototext={work_dir}/run{run_id}.trainer0']
            script.add_parallel_command(
                command, nodes=num_nodes, procs_per_node=procs_per_node)

    # Special handling for last LBANN run
    # Note: If number of experiments doesn't neatly divide number of
    # trainers, repeat the last configuration until we are using a
    # whole number of nodes.
    if trainer_id+1 != num_trainers:
        last_prototext = os.path.join(work_dir, f'run{run_id}.trainer{trainer_id}')
        while ((trainer_id+1)*procs_per_trainer) % procs_per_node != 0:
            trainer_id += 1
            shutil.copyfile(
                last_prototext,
                os.path.join(work_dir, f'run{run_id}.trainer{trainer_id}'))
        command = [
            lbann.lbann_exe(),
            f'--procs_per_trainer={procs_per_trainer}',
            '--generate_multi_proto',
            f'--prototext={work_dir}/run{run_id}.trainer0']
        script.add_parallel_command(
            command,
            nodes=((trainer_id+1)*procs_per_trainer) // procs_per_node,
            procs_per_node=procs_per_node)

    # Clean up file with hyperparameters
    configs_file.close()

    # Run script
    script.run(True)
