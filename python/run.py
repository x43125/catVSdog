# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 22:46:21 2019

@author: ZQQ

cankao：https://blog.csdn.net/qq_36556893/article/details/88943162
"""

import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets,transforms,models

# save matplotlib pics
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import time

import argparse # minglinghang canshu jiexi mokuai

from tensorboardX import SummaryWriter

from network import feature_net

# canshu shezhi
parser = argparse.ArgumentParser(description='cifar10')
parser.add_argument('--pre_epoch',default=0,help='begin epoch')
parser.add_argument('--total_epoch',default=1,help='time for ergodic') 
parser.add_argument('--model',default='vgg19',help='model for training')
parser.add_argument('--outf',default='./model',help='folder to output images and checkpoints') # shuchu jieguo baocun lujing
parser.add_argument('--pre_model',default=False,help='use pre_model') # huifu xunlianshi moxing de lujing
args = parser.parse_args()

# diongyi shiyong moxing
model = args.model

# ruguo you gpu ziyuan shiyong gpu,fouze shiyong cpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load pics
path = '../../data'

# pics yuchuli caozuo zuhe zaiyiqi 
transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

data_image = {x:datasets.ImageFolder(root = os.path.join(path,x),transform = transform) for x in ["train","val"]}


data_loader_image = {x:torch.utils.data.DataLoader(dataset=data_image[x],
                                                   batch_size = 4,
                                                   shuffle = True) for x in ["train","val"]}

classes = data_image["train"].classes # an wenjian ming fenlei
class_index = data_image["train"].class_to_idx # wenjianjia leiming suo duiying de lianzhi
print(classes) # print datatype
print(class_index)

# print train_list,confirm_list size
print("train data set:",len(data_image["train"]))
print("val data set:",len(data_image["val"]))

image_train,label_train = next(iter(data_loader_image["train"]))
mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]
img = torchvision.utils.make_grid(image_train) # batch_size zhang pics pincheng yizhang
print(img.shape) # torch.Size([3,228,906])
img = img.numpy().transpose((1,2,0)) # benlai shi (0,1,2)xiangdangyu ba diyiwei bianwei disanwei,qita liangwei qianyi
print(img.shape)
img = img*std + mean # (228,906,3)fanwei you(-1,1)baincheng(0,1)

print([classes[i] for i in label_train]) # print image_train tuxiang zhong duiying de label_train yejiu shi tuxiang de leixing
plt.savefig('pics/guiyihua.jpg')
plt.imshow(img) # xianshi shuju guiyihua dao 0-1 de tuxiang
#plt.show()

# create network
use_model = feature_net(model,dim=512,n_classes=2)
for parma in use_model.feature.parameters():
    parma.requires_grad = False
    
for index,parma in enumerate(use_model.classifier.parameters()):
    if index == 6:
        parma.requires_grad = True
        
if use_cuda:
    use_model = use_model.to(device)
    
# def loss function
loss = torch.nn.CrossEntropyLoss()

# def youhuaqi
optimizer = torch.optim.Adam(use_model.classifier.parameters())

print(use_model)

# use yuxunlian model
if args.pre_model:
    print("Resume from checkpoint...")
    assert os.path.isdir('checkpoint','Error:no checkpoint directory found')
    state = torch.load('./checkpoint/ckpt.t7')
    use_model.load_state_dict(state['state_dict'])
    best_test_acc = state['acc']
    pre_epoch = state['epoch']

else:
    # def zuiyouhua de ceshi zhunquelv
    best_test_acc = 0
    pre_epoch = args.pre_epoch
    
if __name__ == '__main__':
    total_epoch = args.total_epoch
    writer = SummaryWriter(logdir='./log')
    print("Start Training,{}...".format(model))
    with open("acc.txt","w") as acc_f:
        with open("log.txt","w") as log_f:
            start_time = time.time()
            
            for epoch in range(pre_epoch,total_epoch):
                
                print("epoch{}/{}".format(epoch,total_epoch))
                print("_"*10)
                # start train
                sum_loss = 0.0
                accuracy = 0.0
                total = 0
                for i, data in enumerate(data_loader_image["train"]):
                    image,label = data
                    if use_cuda:
                        image,label = Variable(image.to(device)),Variable(label.to(device))
                    else:
                        image,label = Variable(image),Variable(label)
                        
                    # qianxiang chuanbo
                    label_prediction = use_model(image)
                    
                    _,prediction = torch.max(label_prediction.data,1)
                    total += label.size(0)
                    current_loss = loss(label_prediction,label)
                    # houxiang chuanbo
                    optimizer.zero_grad()
                    current_loss.backward()
                    optimizer.step()
                    
                    sum_loss += current_loss.item()
                    accuracy += torch.sum(prediction == label.data)
                    
                    if total % 5 ==0:
                        print("total {},train loss:{:.4f},train accuracy:{:.4f}".format(total,sum_loss/total,100*accuracy/total))
                        # write on log
                        log_f.write("total {},train loss:{:.4f},train accuracy:{:.4f}".format(total,sum_loss/total,100*accuracy/total))
                        log_f.write('\n')
                        log_f.flush()
                        
                # write on tensorboard
                writer.add_scalar('loss/train',sum_loss / (i+1),epoch)
                writer.add_scalar('accuracy/train',100.*accuracy / total ,epoch)
                # each epoch ceshi zhunquelv
                print("waiting for test...")
                # zai shangxiawen huanjing zhong qieduan tidu jisuan, zaicimoshixia, meiyibude jisuan jieguozhong requires_grad doushi false, jishi input shezhiwei requires_grad = True
                # guding juanji ceng de canshu, zhi gengxin quanlianjie ceng de canshu 
                with torch.no_grad():
                    accuracy = 0
                    total = 0
                    for data in data_loader_image["val"]:
                        use_model.eval()
                        image,label = data
                        if use_cuda:
                            image,label = Variable(image.to(device)),Variable(label.to(device))
                        else:
                            image,label = Variable(image),Variable(label)
                            
                        label_prediction = use_model(image)
                        _,prediction = torch.max(label_prediction.data,1)
                        total += label.size(0)
                        accuracy += torch.sum(prediction==label.data)
                        
                    # print ceshi zhunquelv
                    #print('ceshi zhunque lv: %.3f%%' % (100*accuracy / total))
                    #print('test accuracy： %.3f%%' % (100*accuracy / total))
                    print("test accuracy:{:.4f}%".format(100*accuracy/total))
                    acc = 100.*accuracy / total
                    
                    # write on tensorboard
                    writer.add_scalar('accuracy/test',acc,epoch)
                    
                    # write test result into file
                    print('saing model...')
                    torch.save(use_model.state_dict(),'%s/net_%3d.pth' % (args.outf,epoch + 1))
                    acc_f.write("epoch = %03d,accuracy = %.3f%%" % (epoch + 1,acc))
                    acc_f.write('\n')
                    acc_f.flush()
                    
                    # log the best ceshi zhunquelv
                    if acc > best_test_acc:
                        print('saving best model...')
                        # baocun zhuangtai
                        state = {'state_dict':use_model.state_dict(),
                                 'acc':acc,
                                 'epoch':epoch +1, }
                        
                        # meiyou jiu chuangjian checkpoint folder
                        if not os.path.isdir('checkpoint'):
                            os.mkdir('checkpoint')
                            
                            torch.save(state,'./checkpoint/ckpt.t7')
                            best_test_acc = acc
                            # write on tensorboard
                            writer.add_scalar('best_accuracy/test',best_test_acc,epoch)
                            
    end_time = time.time - start_time
    print("training time is: {:.0f}m {:.0f}s ".format(end_time // 60,end_time % 60))
    writer.close()
                    
                    

