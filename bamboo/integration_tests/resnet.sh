# Temporary Test for ResNet
# See https://ryanstutorials.net/bash-scripting-tutorial/bash-loops.php
index=1
while [ $index -le 7 ]
do
   # Checkpoint will allow for picking up where the last iteration left off.
   salloc --nodes=16 --partition=pbatch --time=600 ./resnet_on_allocated_node.sh
   ((index++))
done
