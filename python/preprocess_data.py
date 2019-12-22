'''
@Author: your name
@Date: 2019-12-20 13:46:23
@LastEditTime : 2019-12-20 13:58:29
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: \catVSdog\python\preprocess_data.py
'''
# -*- coding: utf-8 -*-
import os
import shutil # lib to move pic to new folder

""" init the pic """

def preprocess_data():
    data_file = os.listdir('../../data/Dataset') # read all pics'name
    # print(len(data_file)) #  data size
    # move the pics to two new list where name==cat or dog
    cat_file = list(filter(lambda x:x[:3]=='cat',data_file))
    dog_file = list(filter(lambda x:x[:3]=='dog',data_file))
    
    data_root = '../../data/'
    train_root = '../../data/train'
    val_root = '../../data/val'
    
    for i in range(len(cat_file)):
        print(i)
        pic_path = data_root + 'Dataset/' + cat_file[i]
        if i < len(cat_file)*0.9:
            obj_path = train_root + '/cat/' + cat_file[i]
        else:
            obj_path = val_root + '/cat/' + cat_file[i]
        shutil.move(pic_path,obj_path)
        
    for j in range(len(dog_file)):
        print(j) # show progress
        pic_path = data_root + 'Dataset/' + dog_file[j]
        if j < len(dog_file)*0.9:
            obj_path = train_root + '/dog/' + dog_file[j]
        else:
            obj_path = val_root + '/dog/' + dog_file[j]
        shutil.move(pic_path,obj_path)
        

# the interface of program, all the function
if __name__ == '__main__':
    preprocess_data()
