"""Helper functions to add common command-line arguments."""
import argparse

def add_scheduler_arguments(parser):
    """Add command-line arguments for common scheduler settings.

    Adds the following options: nodes, procs_per_node, partition,
    account, time_limit. The caller is responsible for using them.

    Args:
        parser (argparse.ArgumentParser): command-line argument
            parser.

    """
    if not isinstance(parser, argparse.ArgumentParser):
        raise TypeError('expected an argparse.ArgumentParser')
    parser.add_argument(
        '--nodes', action='store', type=int,
        help='number of compute nodes', metavar='NUM')
    parser.add_argument(
        '--procs-per-node', action='store', type=int,
        help='number of processes per compute node', metavar='NUM')
    parser.add_argument(
        '--partition', action='store', type=str,
        help='scheduler partition', metavar='NAME')
    parser.add_argument(
        '--account', action='store', type=str,
        help='scheduler account', metavar='NAME')
    parser.add_argument(
        '--time-limit', action='store', type=int,
        help='time limit (in minutes)', metavar='MIN')
