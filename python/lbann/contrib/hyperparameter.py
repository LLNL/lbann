import itertools
import os
import shutil

import lbann
import lbann.launcher
import lbann.proto

def grid_search(
        script,
        make_experiment,
        procs_per_trainer=1,
        hyperparameters_file=None,
        use_data_store=False,
        **kwargs,
):
    """Run LBANN with exhaustive grid search over hyperparameter values

    Models are evaluated independently by LBANN trainers. If the
    number of models is greater than the number of trainers, LBANN is
    run multiple times.

    Model configurations are saved to a CSV file.

    Args:
        script (lbann.launcher.batch_script.BatchScript): Batch script
            manager with LBANN launch options.
        make_experiment (function): Function that returns a tuple with
            an lbann.Model, lbann.Optimizer,
            lbann.reader_pb2.DataReader, and lbann.Trainer.
        procs_per_trainer (int): Number of parallel processes per
            LBANN trainer.
        hyperparameters_file (str): CSV file to write model
            configurations (default: hyperparameters.csv in work
            directory).
        **kwargs (list): Hyperparameter values. Each kwarg should be a
            list of values to pass to the corresponding kwarg in
            make_experiment.

    """

    # Launch configuration
    work_dir = script.work_dir
    num_nodes = script.nodes
    procs_per_node = script.procs_per_node
    if (procs_per_node % procs_per_trainer != 0
        and procs_per_trainer % procs_per_node != 0):
        raise RuntimeError(
            f'Trainer size ({procs_per_trainer}) and '
            f'number of MPI ranks per node ({procs_per_node}) '
            'are not multiples of each other')
    num_trainers = (num_nodes * procs_per_node) // procs_per_trainer
    num_nodes = (procs_per_trainer * num_trainers) // procs_per_node

    # Iterate through Cartesian product of hyperparameter values
    # Note: Export Protobuf message for each configuration and add
    # LBANN invocation as needed.
    arg_keys = list(kwargs.keys())
    arg_values = list(kwargs.values())
    hyperparameters = [['run_id', 'trainer_id', 'model_name'] + arg_keys]
    for model_id, values in enumerate(itertools.product(*arg_values)):
        run_id = model_id // num_trainers
        trainer_id = model_id % num_trainers

        # Construct experiment
        args = dict(zip(arg_keys, values))
        model, optimizer, reader, trainer = make_experiment(**args)
        if not model.name:
            model.name = f'model{model_id}'

        # Record hyperparameters
        hyperparameters.append([run_id, trainer_id, model.name] + list(values))

        # Export Protobuf file
        lbann.proto.save_prototext(
            os.path.join(work_dir, f'run{run_id}.trainer{trainer_id}'),
            model=model,
            optimizer=optimizer,
            data_reader=reader,
            trainer=trainer)

        # Invocation to launch LBANN
        if trainer_id == num_trainers-1:
            command = [
                lbann.lbann_exe(),
                f'--procs_per_trainer={procs_per_trainer}',
                '--generate_multi_proto',
                f'--prototext={work_dir}/run{run_id}.trainer0']
            if use_data_store:
                command = command + [
                f'--use_data_store',
                f'--preload_data_store']
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
            hyperparameters.append(list(hyperparameters[-1]))
            hyperparameters[-1][1] = trainer_id
            shutil.copyfile(
                last_prototext,
                os.path.join(work_dir, f'run{run_id}.trainer{trainer_id}'))
        command = [
            lbann.lbann_exe(),
            f'--procs_per_trainer={procs_per_trainer}',
            '--generate_multi_proto',
            f'--prototext={work_dir}/run{run_id}.trainer0']
        if use_data_store:
            command = command + [
            f'--use_data_store',
            f'--preload_data_store']
        script.add_parallel_command(
            command,
            nodes=((trainer_id+1)*procs_per_trainer) // procs_per_node,
            procs_per_node=procs_per_node)

    # Write run configurations to file
    if not hyperparameters_file:
        hyperparameters_file = os.path.join(work_dir, 'hyperparameters.csv')
    with open(hyperparameters_file, 'w') as f:
        for line in hyperparameters:
            f.write(','.join(str(val) for val in line))
            f.write('\n')

    # Run script
    script.run(True)
