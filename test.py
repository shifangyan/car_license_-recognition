#!/usr/bin/env python
# coding=utf-8

import caffe
import sys
import numpy as np
import matplotlib as plt
import skimage
import skimage.io
import skimage.transform

def ImgPrepropcess(img_dir):
    mean_file_dir = "/home/sfy/resnet50_project/ucf101_mean.npy"
    resize_height = 256
    resize_width = 256
    crop_size = 224
    scale =  1.0/256.0

    img = skimage.img_as_float(skimage.io.imread(img_dir)).astype(np.float32)  #图像被读入
    #print img.shape

    resized_img = skimage.transform.resize(img,(resize_height,resize_width))  #缩放图像到指定大小

    mean_npy = np.load(mean_file_dir)

    if(crop_size > resize_height or crop_size > resize_width):
        print "error:裁剪大小不合适 无法裁剪"

    h_off = (resize_height - crop_size) / 2
    w_off = (resize_width - crop_size) /2

    croped_img = resized_img[h_off:h_off+crop_size,w_off:w_off+crop_size] #裁剪图片
    height,width,channels = croped_img.shape
    
    #print "before:",croped_img.shape
    croped_img = croped_img.swapaxes(1,2).swapaxes(0,1)      #交换维度顺序  HWC->CHW
    #print "after:",croped_img.shape

    croped_img = croped_img[(2,1,0),:,:]  #RGB->BGR

    for c in range(channels):
        for h in range(height):
            for w in range(width):
                croped_img[c,h,w] = ((croped_img[c,h,w]*255) - mean_npy[c,h+h_off,w+w_off])  * scale
    return croped_img

def TestImg(deploy_dir,caffemodel_dir,mean_file_dir,img):
    caffe.set_mode_gpu()

    net = caffe.Net(deploy_dir,caffemodel_dir,caffe.TEST)  #加载Net和caffemodel


    transformed_image = ImgPrepropcess(img)
    #print transformed_image
    #plt.imshow(transformed_image)
    #plt.show()

    net.blobs['data'].data[...] = transformed_image   #加载图片到blob

    output = net.forward()
    prob = output['prob'][0]     #取出最后输出的所有类别的概率值 prob为最后一层的名称
    #print prob
    order = prob.argmax()
    print "predicted class is:",order
    return order
def TestLabelTxt(deploy_dir,caffemodel_dir,mean_file_dir,label_txt):
    dir = "/data/users/trandu/datasets/ucf101/frm/"
    caffe.set_mode_gpu()

    net = caffe.Net(deploy_dir,caffemodel_dir,caffe.TEST)



    img_num = 0
    correct_num = 0
    #打开label文件 对每一个图片进行预测
    f = open(label_txt)
    for line in f:  #获取每一行
        img_num = img_num + 1
        strings = line.split(" ")   #分割每一行  获取图片路径和对应标签
        img = strings[0]
        label = int(strings[1])
        img_dir = dir + img

        transformer_img = ImgPrepropcess(img_dir)  #执行图片预处理
        #print transformer_img
        net.blobs['data'].data[...] = transformer_img   #加载图片到blob

        output = net.forward()
        prob = output['prob'][0]   #取出最后输出的所有类别的概率值  prob为最后一层的名称
        order = prob.argmax()

        if(order == label):
            correct_num = correct_num + 1
        #print order,label
        if(img_num % 10000 == 0):
            print "Accuracy:",float(correct_num)/float(img_num)
        #if(img_num > 1000):
        #    break

if __name__ == "__main__":
    if(len(sys.argv) != 4):
        print "error:参数数量错误 usag: python test.py caffemodel  0/1   img/label_txt"
        sys.exit()
    deploy_dir = "/home/sfy/resnet50_project/ResNet50_deploy.prototxt"
    mean_file_dir = "/home/sfy/resnet50_project/ucf101_mean.npy"
    flag = -1
    caffemodel_dir = sys.argv[1]
    flag = int(sys.argv[2])
    print flag
    if(flag == 0):
        print "111"
        img = sys.argv[3]
        TestImg(deploy_dir,caffemodel_dir,mean_file_dir,img)
    elif(flag == 1):
        label_txt = sys.argv[3]
        TestLabelTxt(deploy_dir,caffemodel_dir,mean_file_dir,label_txt)
