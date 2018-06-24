#!/usr/bin/env python
# coding=utf-8
#creat resnet50_ucf101.lst
import os
import random
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_txt",required=True,help="the txt that store label")
    parser.add_argument("--dataset_dir",help="the dataset's path.",default="/home/sfy/car_license_-recognition/car_license_dataset/preproccess_car_license_dataset/")
#    parser.add_argument("--train_txt",required=True,help="the txt that store train dataset.")
#    parser.add_argument("--test_txt",required=True,help="the txt that store test dataset.")
    parser.add_argument("--scale_train_test",help="scale of train set and test set",type=float,default=7.0/3.0)
    args = parser.parse_args()

    print args.scale_train_test
    train_lines = []
    test_lines = []
    label_txt_dic = {}
    chinese_characters = ["京","闽","粤","苏","沪","浙"]

    train_fw = open("./dataset_txt/train.txt","w")
    test_fw = open("./dataset_txt/test.txt","w")

    #建立标签字典
    label_txt_fr = open(args.label_txt,"r")

    for line in label_txt_fr:
        strings = line.split(" ")
        label_txt_dic[strings[0]] = int(strings[1])
    
    dirlist1 = os.listdir(args.dataset_dir)
    for dir1 in dirlist1:
#        print dir1
        full_dir = args.dataset_dir + "/" + dir1
        dirlist2 = os.listdir(full_dir)
        for dir2 in dirlist2:
           # print dir2
            if(dir1 == "chinese-characters"):
                chinese_character = chinese_characters[int(dir2)]
                print chinese_character
                if label_txt_dic.has_key(chinese_character):
                    #print "111"
                    full_dir = args.dataset_dir + "/" + dir1 + "/" + dir2
                    dirlist3 = os.listdir(full_dir)
                    for dir3 in dirlist3:
                        img_dir = dir1 + "/"+ dir2 + "/" + dir3 
                        line = img_dir + " " + str(label_txt_dic[chinese_character]) + "\n" 
                        if(random.randint(0,100) < (args.scale_train_test /(args.scale_train_test+1.0))*100.0): 
                            train_lines.append(line)
                        else:
                            test_lines.append(line)
            else:
                if label_txt_dic.has_key(dir2):
                    full_dir = args.dataset_dir + "/" + dir1 + "/" + dir2
                    dirlist3 = os.listdir(full_dir)
                    for dir3 in dirlist3:
                        img_dir = dir1 + "/" + dir2 + "/" + dir3
                        line = img_dir + " " + str(label_txt_dic[dir2]) + "\n"
                        if(random.randint(0,100) < (args.scale_train_test/(args.scale_train_test+1.0))*100.0):
                            train_lines.append(line)
                        else:
                            test_lines.append(line)
    
    train_fw.writelines(train_lines)
    train_fw.close()
    test_fw.writelines(test_lines)
    test_fw.close()

    print "success to write file"
                
