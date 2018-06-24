#!/usr/bin/env python
# coding=utf-8

import argparse
import caffe
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("--binaryproto_file",required=True,help="the file store the mean value of binary")
parser.add_argument("--npy_file",required=True,help="the file store the mean value of npy")
args = parser.parse_args()

blob = caffe.proto.caffe_pb2.BlobProto()
binary_mean = open(args.binaryproto_file,"rb").read()
blob.ParseFromString(binary_mean)
arr = np.array(caffe.io.blobproto_to_array(blob))
npy_mean = arr[0]
np.save(args.npy_file,npy_mean)

