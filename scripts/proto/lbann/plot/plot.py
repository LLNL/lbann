# Global imports
import os
import sys
import json
import matplotlib.pyplot as plt
import texttable as tt
import pandas as pd

# Local imports
from . import parser

PRETTY_YLIM_LOSS = 5

def _get_time_axis(time_list, units='hours'):
    """Convert time to sequential format and convert time units."""
    time_axis = []
    for i in range(len(time_list)):
        time_sum = sum(time_list[:i])
        if units == 'seconds':
            pass
        elif units == 'minutes':
            time_sum /= 60.0
        elif units == 'hours':
            time_sum /= 3600.0
        time_axis.append(time_sum)
    return time_axis

def plot(stat_path_list, stat_name_list, ind_var='time', time_units='hours',
         plot_accuracy=True, merge_train_val=False, pretty_ylim=True, save_fig=None, save_csv=None, ylim=None,
         test_loss=False):
    """Tabulate and plot stats from LBANN or PyTorch training in common format."""

    if pretty_ylim and ylim is not None:
        print('ERROR: pretty_ylim and ylim must not be set at the same time.')
        sys.exit(1)

    ### Load stat dicts and print stat summary
    stat_dict_list = []
    # Get run names
    if stat_name_list is None:
        print('WARNING: No trial names provided, using file names by default.')
        run_name_list = [os.path.splitext(os.path.basename(stat_path))[0] for stat_path in stat_path_list]
    elif len(stat_name_list) != len(stat_path_list):
        print('WARNING: # stat paths ({}) does not match # stat names ({}). Using file names by default.'.format(
            len(stat_path_list), len(stat_name_list)))
        run_name_list = [os.path.splitext(os.path.basename(stat_path))[0] for stat_path in stat_path_list]
    else:
        run_name_list = stat_name_list
    # Create table for comparing trials
    stat_table = tt.Texttable()
    headings = ['Trial', 'Num Procs', 'Num Nodes', 'Num Epochs', 'Avg. Train Time (s)', 'Avg. Val Time (s)']
    if plot_accuracy:
        headings += ['Peak Train Acc', 'Peak Val Acc']

    headings += ['Min. Train Loss', 'Min. Val Loss']
    if test_loss:
        headings += ['Min. Test Loss']

    stat_table.header(headings)
    # Loop through each trial
    rows = []
    row_names = []
    for run_name, stat_path in zip(run_name_list, stat_path_list):
        # Load stat file
        stat_ext = os.path.splitext(stat_path)[1]
        if stat_ext == '.json':
            with open(stat_path, 'r') as fp:
                d = json.load(fp)
        elif stat_ext == '.out' or stat_ext == '.txt':
            d = parser.parse(stat_path)
            if d is None:
                print('WARNING: Failed to parse outputs from {}'.format(stat_path))
                continue
        else:
            print('ERROR: Invalid file extension: {} from {}\nPlease provide either an LBANN output file with .out or .txt extension or a PyTorch output file with .json extension.'.format(stat_ext, stat_path))
            sys.exit(1)

        # Total number of processes
        def parse_num(d, key):
            if key in d.keys() and len(set(d[key])) == 1:
                return d[key][0]
            else:
                return None

        num_procs = parse_num(d, 'num_procs')
        num_procs_on_node = parse_num(d, 'num_procs_on_node')
        if num_procs is not None and num_procs_on_node is not None:
            assert (num_procs % num_procs_on_node) == 0
            num_nodes = int(num_procs / num_procs_on_node)
        else :
            num_nodes = None
            print('WARNING: No process counts are provided from {}'.format(stat_path))

        # Total epochs of training
        total_epochs = len(d['val_time'])

        # Compute accuracy stats
        if plot_accuracy:
            if len(d['train_acc']) == 0:
                print('WARNING: No accuracy information is provided from {}'.format(stat_path))
                continue

            peak_train_acc = max(d['train_acc'])
            peak_train_epoch = d['train_acc'].index(peak_train_acc)
            peak_val_acc = max(d['val_acc'])
            peak_val_epoch = d['val_acc'].index(peak_val_acc)

        if len(d['train_loss']) == 0:
            print('WARNING: No loss information is provided from {}'.format(stat_path))
            continue

        # Compute loss stats
        min_train_loss = min(d['train_loss'])
        min_train_epoch = d['train_loss'].index(min_train_loss)
        min_val_loss = min(d['val_loss'])
        min_val_epoch = d['val_loss'].index(min_val_loss)
        min_test_loss = d['test_loss'][0] if test_loss else None

        # Compute time stats
        avg_train_time = int(sum(d['train_time'])/len(d['train_time']))
        avg_val_time = int(sum(d['val_time'])/len(d['val_time']))

        # Create independent variable axis
        if ind_var == 'epoch':
            d['train_axis'] = range(len(d['train_time']))
            d['val_axis'] = range(len(d['val_time']))
            xlabel = 'Epoch'
        elif ind_var == 'time':
            d['train_axis'] = _get_time_axis(d['train_time'], units=time_units)
            d['val_axis'] = _get_time_axis(d['val_time'], units=time_units)
            xlabel = 'Time ({})'.format(time_units)
        else:
            raise Exception('Invalid indepedent variable: {}'.format(ind_var))

        # Store the stat dict for plotting
        stat_dict_list.append((run_name, d))

        # Add row to stats table for current trial
        row = [run_name, num_procs, num_nodes, total_epochs, avg_train_time, avg_val_time] \
            + ([peak_train_acc, peak_val_acc] if plot_accuracy else []) \
            + [min_train_loss, min_val_loss] \
            + ([min_test_loss] if test_loss else [])
        rows.append(row)
        row_names.append(run_name)

    for row in rows:
        stat_table.add_row(row)

    # Print the stats table
    print()
    table_str = stat_table.draw()
    print(table_str)
    print()

    ### Plot stats
    plt.figure(figsize=(12, 10 if plot_accuracy else 5))
    plt.suptitle('Trial Stats vs. {}'.format(ind_var.title(), fontsize=20))
    subplot_nrow = 2 if plot_accuracy else 1
    subplot_ncol = 2 if not merge_train_val else 1
    for run_name, stat_dict in stat_dict_list:
        run_name_train = run_name if not merge_train_val else run_name+' (train)'
        run_name_val   = run_name if not merge_train_val else run_name+' (val)'

        # Train acc
        if plot_accuracy:
            plt.subplot(subplot_nrow, subplot_ncol, 1)
            plt.title('Train Accuracy vs. {}'.format(ind_var.title()))
            plt.xlabel(xlabel)
            plt.ylabel('Train Accuracy')
            if pretty_ylim:
                plt.ylim(0, 1)

            plt.plot(stat_dict['train_axis'], stat_dict['train_acc'], label=run_name_train)

        # Val acc
        if plot_accuracy:
            plt.subplot(subplot_nrow, subplot_ncol, 2 if not merge_train_val else 1)
            plt.title('Val Accuracy vs. {}'.format(ind_var.title()))
            plt.xlabel(xlabel)
            plt.ylabel('Val Accuracy')
            if pretty_ylim:
                plt.ylim(0, 1)

            plt.plot(stat_dict['val_axis'], stat_dict['val_acc'], label=run_name_val)

        loss_subplot = 1 + ((2 if not merge_train_val else 1) if plot_accuracy else 0)

        # Train loss
        plt.subplot(subplot_nrow, subplot_ncol, loss_subplot)
        plt.title('Train Loss vs. {}'.format(ind_var.title()))
        plt.xlabel(xlabel)
        plt.ylabel('Train Loss')
        if pretty_ylim:
            plt.ylim(0, PRETTY_YLIM_LOSS)
        elif ylim is not None:
            plt.ylim(*ylim)

        p, = plt.plot(stat_dict['train_axis'], stat_dict['train_loss'], label=run_name_train)

        # Val loss
        plt.subplot(subplot_nrow, subplot_ncol, loss_subplot + (1 if not merge_train_val else 0))
        plt.title('Val Loss vs. {}'.format(ind_var.title()))
        plt.xlabel(xlabel)
        plt.ylabel('Val Loss')
        if pretty_ylim:
            plt.ylim(0, PRETTY_YLIM_LOSS)
        elif ylim is not None:
            plt.ylim(*ylim)

        kwargs = {} if not merge_train_val else {"color": p.get_color(), "linestyle": "dashed"}
        plt.plot(stat_dict['val_axis'], stat_dict['val_loss'], label=run_name_val, **kwargs)

    # Legend position will likely only be good for the test example
    # plt.legend(loc=(0.25, 1.22))
    plt.legend()

    if save_fig is None:
        # Show the plot
        plt.show()
    else:
        plt.savefig(save_fig)

    if save_csv is not None:
        df = pd.DataFrame([dict(zip(headings, row)) for row in rows],
                          index=row_names)
        df.to_csv(save_csv)
