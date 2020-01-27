The tools in this directory are embarrassingly 
parallel. They don't use GPUs, so you are advised to compile
lbann without CUDA, in order to use all avalailable CPUs
on your nodes. 

Typical invocation on lassen:
  $ jsrun  -n 8 -a 40  -d packed -b "packed:10" -r 1 -c 40 <executable> <args>
