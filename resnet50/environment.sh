#!/bin/bash

# You can find the URl in the README file
model_url="https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz"

wget $model_url

tar xzf $(basename $model_url)