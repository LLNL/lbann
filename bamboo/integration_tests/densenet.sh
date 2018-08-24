# Temporary Test for DenseNet
# See https://ryanstutorials.net/bash-scripting-tutorial/bash-loops.php
# Estimated three days to train to convergence => 3*24*60 = 4,320 minutes
# 4,320 minutes / 600 minutes = 7.2 iterations
index=1
# Is it true that if we put more iterations than necessary,
# then the extra ones will start/stop instantly
# (e.g. nothing left to do, so return the nodes)?
# Of course, we still have to worry about queue time to get the nodes.
while [ $index -le 19 ]
do
   # Checkpoint will allow for picking up where the last iteration left off.
   # salloc --nodes=1 --partition=pbatch --time=1 ./densenet_on_allocated_node.sh
   salloc --nodes=32 --partition=pbatch --time=600 ./densenet_on_allocated_node.sh
   ((index++))
done
