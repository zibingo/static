import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os


import config
from getdata import TrainDataset
from network import ConvNet,ResConvNet

if torch.cuda.is_available():
    print("使用gpu训练")
else:
    print("使用cpu训练")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = ConvNet().to(device)
model = ResConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=config.init_learn_rate)
#动态学习率
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=config.gamma_rate)
# 定义loss计算方法，cross entropy，交叉熵，可以理解为两者数值越接近其值越小
criterion = torch.nn.CrossEntropyLoss()

if torch.cuda.device_count() > 1:
    print("使用{}张显卡进行训练!".format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)


#获取模型的代数
history_epoch = []
for filename in os.listdir(config.model_path):
    if "pth" in filename:
        history_epoch.append(int(filename.split(".")[0].split("_")[2]))
# 如果有保存的模型，则加载模型，并在其基础上继续训练
end_epoch = -1
if len(history_epoch) == 0:
    end_epoch  = 0
    print('无保存模型,将从头开始训练！')
elif len(history_epoch) > 0 and len(history_epoch) != config.epochs_num:
    end_epoch = np.max(history_epoch)
    checkpoint = torch.load(config.model_path+"checkpoint_model_{}.pth".format(end_epoch))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    end_epoch = checkpoint['epoch']
    print('加载epoch {} 参数成功,继续训练'.format(end_epoch))
else:
    print("已完成训练,退出进程!")
    exit()

def train(train_loader):
    model.train()
    total_loss = 0.0
    flag = True
    for index,(labels,images) in enumerate(train_loader,0):
        images, labels = images.to(device), labels.to(device)
#         print(images.shape)
#         if flag:break
#         print(images, labels)
        outputs = model(images)
        
#         print(outputs.shape)
#         print(outputs)
#         if flag:break

        loss = criterion(outputs,labels)  # 计算损失，也就是网络输出值和实际label的差异，显然差异越小说明网络拟合效果越好，此处需要注意的是第二个参数，必须是一个1维Tensor
        
        total_loss += loss.item()
        
        print("[{}/{}]] - loss:{}".format(index,len(train_loader),loss.item()))

        # 存储的变量梯度清零
        optimizer.zero_grad()
        # 求反向传播的梯度
        loss.backward()
        # 开始优化权重
        optimizer.step()
    return total_loss

def test(train_dataset):
    model.eval()
    correct_num = 0
    for i in range(train_dataset.__len__()):
        label,img = train_dataset.__getitem__(i)
        img = img.unsqueeze(0)
        out = model(img)                      # 输出概率
        num = out.argmax(1).item()
        if label == num:
            correct_num += 1
    return (correct_num/train_dataset.__len__()) * 100



if __name__ == "__main__":
    train_dataset = TrainDataset(config.train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
                          drop_last=True)
    #根据最后一次训练的代数进行训练
    for epoch in range(end_epoch+1,config.epochs_num+1):
        total_loss = train(train_loader)
#         correct_rate = test(train_dataset)
#         scheduler.step()
        if epoch % config.interval1 == 0:
            print('Epoch:{} - Total_train_loss:{}'.format(epoch,total_loss))
#             print("训练集的准确率:{}%".format(correct_rate))
        if epoch % config.interval2 == 0:
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epoch,
                     'total_loss':total_loss}
            torch.save(state, config.model_path+"checkpoint_model_{}.pth".format(epoch))