salloc -N16 --enable-hyperthreads -t 1440
salloc -N16 --enable-hyperthreads --clear-ssd -t 1440
run_lbann_dnn_imagenet.sh 1 1 500
