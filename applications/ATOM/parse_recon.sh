#!/bin/bash

# Parse command-line arguments
if [ -z "$1" ]; then
    echo "Usage: $(basename $0) LOG_FILE"
    exit 0
fi
log_file=$(realpath "$1")

# Parse log file
cat ${log_file} \
    | grep "^.*(instance [0-9]*) training epoch [0-9]* recon" \
    | sed "s/^.*(instance \([0-9]*\)) training epoch \([0-9]*\).*: \(.*\)/\1\t\2\t\3/" \
    | awk '
function max(x,y) {
  if (x == "") { return y; }
  if (y == "") { return x; }
  return x > y ? x : y;
}
function min(x,y) {
  if (x == "") { return y; }
  if (y == "") { return x; }
  return x < y ? x : y;
}
{
  sums[$2] += $3;
  sqsums[$2] += $3*$3;
  maxs[$2] = max($3,maxs[$2]);
  mins[$2] = min($3,mins[$2]);
  trainers = max(trainers,$1+1);
  epochs = max(epochs,$2+1);
}
END {
  for (i=0;i<epochs;++i) {
    mean = sums[i]/trainers;
    sqmean = sqsums[i]/trainers;
    print("Epoch ",i,", recon : mean=",mean,", stdev=",sqmean-mean*mean,", min=",mins[i],", max=",maxs[i]);  }
}' \
