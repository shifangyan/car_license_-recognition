#!/usr/bin/env python
# coding=utf-8

import caffe
import skimage
from skimage import filters
from skimage import morphology
from skimage import transform
import matplotlib.pyplot as plt
import argparse
import numpy as np

def car_license_segmentation(img_dir):
#    skimage.io.use_plugin("matplotlib","imshow")
    img = skimage.img_as_float(skimage.io.imread(img_dir)).astype(np.float32)  #读入图像
    resize_img = transform.resize(img,(450,300))
    skimage.io.imsave("sfy1.jpg",resize_img)
    gray_img = skimage.color.rgb2gray(img)
    thresh = filters.threshold_otsu(gray_img)
    binary_img = (gray_img > thresh)
    open_img = morphology.binary_opening(binary_img)
    close_img = morphology.binary_closing(open_img)
    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.subplot(2,3,2)
    plt.imshow(gray_img,cmap="gray")
    plt.subplot(2,3,3)
    plt.imshow(binary_img,cmap="gray")
    plt.subplot(2,3,4)
    plt.imshow(open_img,cmap="gray")
    plt.subplot(2,3,5)
    plt.imshow(close_img,cmap="gray")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--deploy",help="the prototxt file of model",default="/home/sfy/car_license_-recognition/prototxt/lenet_deploy.prototxt")
    parser.add_argument("--caffe_model",required=True,help="the weight of model")
    parser.add_argument("--mean_file",help="the mean file of dataset",default="/home/sfy/car_license_-recognition/mean/dataset_mean.npy")
    parser.add_argument("--image_dir",required=True,help="the image  of car license")

    args = parser.parse_args()

    help(skimage)
    car_license_segmentation(args.image_dir)

