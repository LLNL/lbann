#!/bin/bash
# ./launch_merlin.sh  "num_nodes:4 8"  "mb_size:512 2048"  "ppt:16 32"  "ipt:1024"
# ./launch_merlin.sh -y train_atom_wae.yaml -n "4 16" -b 4096 -p "4 16" -i 100

echo "Running job: $0 $@"

NUM_NODES="num_nodes:1"
MB_SIZE="mb_size:1"
PPT="ppt:1"
IPT="ipt:1"
YAML=train_atom.yaml
LR_SCALING="lr_scaling:False"

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }

[ $# -eq 0 ] && usage
while getopts ":hn:b:i:p:y:l:" arg; do
  case $arg in
    y) # Yaml file
      YAML=${OPTARG}
      echo $YAML
      ;;
    n) # Number of nodes
      NUM_NODES="num_nodes:${OPTARG}"
      echo $NUM_NODES
      ;;
    b) # mini-batch size
      MB_SIZE="mb_size:${OPTARG}"
      echo $MB_SIZE
      ;;
    p) # processors per trainer
      PPT="ppt:${OPTARG}"
      echo $PPT
      ;;
    i) # iterations per tournament
      IPT="ipt:${OPTARG}"
      echo $IPT
      ;;
    l) # lr_scaling (True or False)
      LR_SCALING="lr_scaling:${OPTARG}"
      echo $LR_SCALING
      ;;
    h | *) # Display help.
      usage
      exit 0
      ;;
  esac
done
shift $((OPTIND-1))

#if [ -z "${n}" ] || [ -z "${b}" ]  || [ -z "${p}" ]  || [ -z "${i}" ]; then
# echo "${n}"
# echo "${b}"
# echo "${p}"
# echo "${i}"
# if [ -z "${n}" ] || [ -z "${b}" ]  || [ -z "${p}" ]  || [ -z "${i}" ]; then
#     echo "Missing required arguments"
#     usage
# fi

# Turn off core files to work aroung flux exec issue.
ulimit -c 0

merlin stop-workers --spec ${YAML}
merlin purge -f ${YAML}
# Send all tasks to the broker
# Use a custom generator to give good names to each test
CMD="merlin run ${YAML} --pgen merlin_pgen_make_test_permutations.py --parg \"$NUM_NODES\" --parg \"$MB_SIZE\" --parg \"$PPT\" --parg \"$IPT\" --parg \"$LR_SCALING\""
echo $CMD
merlin run ${YAML} --pgen merlin_pgen_make_test_permutations.py --parg "$NUM_NODES" --parg "$MB_SIZE" --parg "$PPT" --parg "$IPT" --parg "$LR_SCALING"

# Show the workers command
merlin run-workers ${YAML} --echo

# Start workers to run the tasks in the broker
merlin run-workers ${YAML}

# Keep the allocation alive until all workers stop
merlin monitor ${YAML}
