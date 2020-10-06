#!/bin/bash

# Parse command-line arguments
if [ -z "$1" ]; then
    echo "Usage: $(basename $0) LOG_FILE"
    exit 0
fi
log_file=$(realpath "$1")

# Parse log file for training recon loss
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
END {
  for (epoch=0; epoch<num_epochs; ++epoch) {
    mean = sums[epoch] / counts[epoch];
    sqmean = sqsums[epoch] / counts[epoch];
    print("Epoch ",epoch,", train recon : ",
          "mean=",mean,", stdev=",sqmean-mean*mean,", ",
          "min=",mins[epoch],", max=",maxs[epoch]);
  }
}'

# Parse log file for test recon loss
cat ${log_file} \
    | grep "^.*(instance [0-9]*) test recon" \
    | sed "s/^.*(instance \([0-9]*\)) test recon : \(.*\)/\1\t\2/" \
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
  trainer = $1
  epoch = int(epochs[trainer]);
  epochs[trainer]++;
  val = $2
  if (val * 1 > 0) {
    sums[epoch] += val;
    sqsums[epoch] += val*val;
    maxs[epoch] = max(val,maxs[epoch]);
    mins[epoch] = min(val,mins[epoch]);
    counts[epoch]++;
    num_epochs = max(num_epochs,epoch+1);
  }
}
END {
  for (epoch=0; epoch<num_epochs; ++epoch) {
    mean = sums[epoch] / counts[epoch];
    sqmean = sqsums[epoch] / counts[epoch];
    print("Epoch ",epoch,", train recon : ",
          "mean=",mean,", stdev=",sqmean-mean*mean,", ",
          "min=",mins[epoch],", max=",maxs[epoch]);
  }
}'
