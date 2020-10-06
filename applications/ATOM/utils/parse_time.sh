#!/bin/bash

# Parse command-line arguments
if [ -z "$1" ]; then
    echo "Usage: $(basename $0) LOG_FILE"
    exit 0
fi
log_file=$(realpath "$1")

# Parse log file for epoch times
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
  trainer = $1
  epoch = $2
  val = $3
  if (val * 1 > 0) {
    sums[epoch] += val;
    sqsums[epoch] += val*val;
    maxs[epoch] = max(val,maxs[epoch]);
    mins[epoch] = min(val,mins[epoch]);
    counts[epoch]++;
    num_epochs = max(num_epochs,epoch+1);
  }
}

# Print stats
END {

  # Print per-epoch stats
  for (epoch=0; epoch<num_epochs; ++epoch) {
    mean = sums[epoch] / counts[epoch];
    sqmean = sqsums[epoch] / counts[epoch];
    print("Epoch ",epoch,", epoch time : ",
          "mean=",mean,", stdev=",sqmean-mean*mean,", ",
          "min=",mins[epoch],", max=",maxs[epoch]);
  }

  # Print all-epoch stats
  sum = 0;
  count = 0;
  for (epoch=0; epoch<num_epochs; ++epoch) {
    sum += sums[epoch];
    count += counts[epoch];
  }
  print("All epochs (including 0) epoch time : mean=",sum/count);

  # Print all-but-first-epoch stats
  sum = 0;
  count = 0;
  for (epoch=1; epoch<num_epochs; ++epoch) {
    sum += sums[epoch];
    count += counts[epoch];
  }
  print("All epochs (except 0) epoch time : mean=",sum/count);

}'

# Parse log file for mini-batch times
cat ${log_file} \
    | grep "^.*(instance [0-9]*) training epoch [0-9]* mini-batch" \
    | sed "s/^.*(instance \([0-9]*\)) training epoch \([0-9]*\).*: \(.*\)s/\1\t\2\t\3/" \
    | awk '
{
  trainer = $1;
  epoch = $2;
  val = $3;
  if (epoch > 0) {
    sum += val;
    count += 1;
  }
}

END {
  print("All epochs (except 0) mini-batch time : mean=",sum/count);
}
'
