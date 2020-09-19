#!/bin/bash
# ./launch_merlin.sh  "num_nodes:4 8"  "mb_size:512 2048"  "ppt:16 32"  "ipt:1024"

# Turn off core files to work aroung flux exec issue.
ulimit -c 0

YAML=train_atom.yaml

merlin stop-workers --spec ${YAML}
merlin purge -f ${YAML}
# Send all tasks to the broker
# Use a custom generator to give good names to each test
CMD="merlin run train_atom.yaml --pgen merlin_pgen_make_test_permutations.py --parg \"$1\" --parg \"$2\" --parg \"$3\" --parg \"$4\""
echo $CMD
merlin run train_atom.yaml --pgen merlin_pgen_make_test_permutations.py --parg "$1" --parg "$2" --parg "$3" --parg "$4"

# Show the workers command
merlin run-workers ${YAML} --echo

# Start workers to run the tasks in the broker
merlin run-workers ${YAML}

# Keep the allocation alive until all workers stop
merlin monitor ${YAML}
