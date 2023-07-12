import torch
import torch.nn.functional as F

class ConvNet(torch.nn.Module):
    """
    每个卷积核的通道数与原通道数一致

    卷积核的数量与输出通道数一致

    卷积核的大小与图像大小无关
    """
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,16,3, padding=1)
        self.conv2 = torch.nn.Conv2d(16,16,3, padding=1)

        self.FC1 = torch.nn.Linear(7*7*16,128)
        self.FC2 = torch.nn.Linear(128,64)
        self.FC3 = torch.nn.Linear(64,10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size()[0], -1)  # 降维成一位向量
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        x = self.FC3(x)
        return F.softmax(x, dim=1)

class ResBlock(torch.nn.Module):
    def __init__(self,channels):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(channels,channels,kernel_size = 3,padding = 1)
        self.conv2 = torch.nn.Conv2d(channels,channels,kernel_size = 3,padding = 1)
    def forward(self,x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(y+x)


class ResConvNet(torch.nn.Module):
    def __init__(self):
        super(ResConvNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,16,3, padding=1)
        self.conv2 = torch.nn.Conv2d(16,32,3, padding=1)

        self.resblock1 = ResBlock(16)
        self.resblock2 = ResBlock(32)

        self.FC1 = torch.nn.Linear(7*7*32,128)
        self.FC2 = torch.nn.Linear(128,64)
        self.FC3 = torch.nn.Linear(64,10)
    
    def forward(self,x):
        in_size = x.size()[0]

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)

        x = self.resblock1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)

        x = self.resblock2(x)   


        x = x.view(in_size, -1)  # 降维成一位向量
        x = F.relu(self.FC1(x))
        x = F.relu(self.FC2(x))
        x = self.FC3(x)
        return F.softmax(x, dim=1)

if __name__ == "__main__":
    ResConv = ResConvNet()
    x = torch.rand([1,1,28,28])
    # print(x.shape)
    res = ResConv(x)
    print(res.shape)