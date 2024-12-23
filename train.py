import os
import numpy as np
import torch
from torchvision import datasets,transforms
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import resnet_model
from utils.util import train_and_val,plot_loss,plot_acc
from utils.try_gpu import try_gpu

if __name__=="__main__":
    device=try_gpu()
    print("当前调用是",device)

    if not os.path.exists("model"):
        os.makedirs("model")
    #训练前的准备：加载数据，设定数据的处理方式，设置超参数
    BATCH_SIZE=16
    data_transform={
        "train":transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
        "val":transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
    }
    train_dataset = datasets.ImageFolder("data/training", transform=data_transform["train"])  # 训练集
    train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    len_train=len(train_dataset)
    val_dataset = datasets.ImageFolder("data/validation", transform=data_transform["val"])
    val_loader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,num_workers=4)
    len_val=len(val_dataset)
    model=resnet_model.resnet50() #设置模型为手写的resnet50
    loss_fn=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    epochs=30
    history=train_and_val(epochs,model,train_loader,len_train,val_loader,len_val,loss_fn,optimizer,device)
    plot_acc(np.arange(0,epochs),history)
    plot_loss(np.arange(0,epochs),history)
