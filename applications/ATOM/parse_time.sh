#!/bin/bash

# Parse command-line arguments
if [ -z "$1" ]; then
    echo "Usage: $(basename $0) LOG_FILE"
    exit 0
fi
log_file=$(realpath "$1")

# Parse log file
cat ${log_file} \
    | grep "^.*(instance [0-9]*) training epoch [0-9]* run time" \
    | sed "s/^.*(instance \([0-9]*\)) training epoch \([0-9]*\).*: \(.*\)s/\1\t\2\t\3/" \
    | awk '

# Helper functions
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

# Compute sums, square of sums, mins, and maxes
{
  sums[$2] += $3;
  sqsums[$2] += $3*$3;
  maxs[$2] = max($3,maxs[$2]);
  mins[$2] = min($3,mins[$2]);
  trainers = max(trainers,$1+1);
  epochs = max(epochs,$2+1);
}

# Print stats
END {

  # Print per-epoch stats
  for (i=0;i<epochs;++i) {
    mean = sums[i]/trainers;
    sqmean = sqsums[i]/trainers;
    print("Epoch ",i,", epoch time : mean=",mean,", stdev=",sqmean-mean*mean,", min=",mins[i],", max=",maxs[i]);   }

  # Print all-epoch stats
  epoch_sum = 0;
  epoch_sqsum = 0;
  epoch_min = "";
  epoch_max = "";
  for (i=0;i<epochs;++i) {
    epoch_sum += sums[i];
    epoch_sqsum += sqsums[i];
    epoch_min = min(epoch_min, mins[i]);
    epoch_max = max(epoch_max, maxs[i]);
  }
  mean = epoch_sum / (trainers * epochs);
  sqmean = epoch_sqsum / (trainers * epochs);
  print("All epochs (including 0) epoch time : mean=",mean,", stdev=",sqmean-mean*mean,", min=",epoch_min,", max=",epoch_max);

  # Print all-but-first-epoch stats
  epoch_sum = 0;
  epoch_sqsum = 0;
  epoch_min = "";
  epoch_max = "";
  for (i=1;i<epochs;++i) {
    epoch_sum += sums[i];
    epoch_sqsum += sqsums[i];
    epoch_min = min(epoch_min, mins[i]);
    epoch_max = max(epoch_max, maxs[i]);
  }
  mean = epoch_sum / (trainers * epochs);
  sqmean = epoch_sqsum / (trainers * epochs);
  print("All epochs (except 0) epoch time : mean=",mean,", stdev=",sqmean-mean*mean,", min=",epoch_min,", max=",epoch_max);

}' \
