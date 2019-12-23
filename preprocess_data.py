'''
@Author: your name
@Date: 2019-12-20 13:46:23
@LastEditTime : 2019-12-23 15:03:16
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \catVSdog\python\preprocess_data.py
'''
# -*- coding: utf-8 -*-
import os
import shutil # 用来移动图片的库，直接移走

""" 图片初步处理 """

def preprocess_data():
    data_file = os.listdir('data/Dataset') # 读取所有图片的名字
    # print(len(data_file)) # 查看数据大小
    # 将图片名为cat和dog的图片分别取出来，存为两个list
    cat_file = list(filter(lambda x:x[:3]=='cat',data_file))
    dog_file = list(filter(lambda x:x[:3]=='dog',data_file))
    
    data_root = 'data/'
    train_root = 'data/train'
    val_root = 'data/val'
    
    for i in range(len(cat_file)):
        print(i)
        pic_path = data_root + 'Dataset/' + cat_file[i]
        if i < len(cat_file)*0.9:
            obj_path = train_root + '/cat/' + cat_file[i]
        else:
            obj_path = val_root + '/cat/' + cat_file[i]
        shutil.move(pic_path,obj_path)
        
    for j in range(len(dog_file)):
        print(j) # 查看进度
        pic_path = data_root + 'Dataset/' + dog_file[j]
        if j < len(dog_file)*0.9:
            obj_path = train_root + '/dog/' + dog_file[j]
        else:
            obj_path = val_root + '/dog/' + dog_file[j]
        shutil.move(pic_path,obj_path)
        

# 程序运行接口，调用函数
if __name__ == '__main__':
    preprocess_data()
