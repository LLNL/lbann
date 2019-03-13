#!/bin/bash

# Download .tar.gz ONNX models from ${urls} to ${dir} and decompress them.
# Usage: ./download_onnx_model_zoo.sh

dir=onnx_model_zoo
urls=( \
       https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz \
           https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_alexnet.tar.gz \
           https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_caffenet.tar.gz \
           https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_googlenet.tar.gz \
           https://s3.amazonaws.com/download.onnx/models/opset_9/vgg19.tar.gz \
           https://s3.amazonaws.com/download.onnx/models/opset_9/resnet50.tar.gz \
    )

mkdir -p ${dir}

for url in ${urls[@]}; do
    filename=`basename ${url}`
    modeldirname=`echo ${filename} | sed "s/\.tar\.gz$//g"`
    if [[ ! -e ${dir}/${filename} ]]; then
        wget ${url} --directory-prefix=${dir}
    fi
    if [[ ! -e ${dir}/${modeldirname} ]]; then
        tar -zxvf ${dir}/${filename} -C ${dir}
    fi
done
