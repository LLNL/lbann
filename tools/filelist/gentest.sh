#!/bin/sh

for i in `seq 1 1000`
do
  str=file$i
  for j in `seq 0 999`
  do
     str=$str' sample_'$i'_'$j
  done
  echo $str
done
