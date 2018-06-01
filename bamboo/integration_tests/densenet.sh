# Temporary Test for DenseNet
# See https://ryanstutorials.net/bash-scripting-tutorial/bash-loops.php
# Three days to train to convergence => 3*24*60 = 4,320 minutes
# 4,320 minutes / 600 minutes = 7.2 iterations
index=1
#while [ $index -le 7 ]
while [ $index -le 1 ] 
do
   # Checkpoint will allow for picking up where the last iteration left off.
   # salloc --nodes=1 --partition=pbatch --time=1 ./densenet_on_allocated_node.sh
   salloc --nodes=32 --partition=pbatch --time=600 ./densenet_on_allocated_node.sh
   ((index++))
done
