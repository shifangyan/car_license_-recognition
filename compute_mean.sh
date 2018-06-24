#!/bin/bash
/home/sfy/caffe/build/tools/convert_imageset --resize_width=32 --resize_height=32 --backend=lmdb /home/sfy/car_license_-recognition/car_license_dataset/preproccess_car_license_dataset/ ./dataset_txt/train.txt  ./dataset_txt/dataset_train_lmdb
/home/sfy/caffe/build/tools/convert_imageset --resize_width=32 --resize_height=32 --backend=lmdb /home/sfy/car_license_-recognition/car_license_dataset/preproccess_car_license_dataset/ ./dataset_txt/test.txt  ./dataset_txt/dataset_test_lmdb
/home/sfy/caffe/build/tools/compute_image_mean ./dataset_txt/dataset_train_lmdb  ./mean/dataset_mean.binaryproto
