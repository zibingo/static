"""
配置文件
"""
#batch_size
batch_size = 128
#迭代次数
epochs_num = 2000
#初始学习率
init_learn_rate = 0.0001
#衰减度
gamma_rate = 0.99
# PyTorch读取数据线程数量，一般为2
num_workers = 2
#模型路径
model_path = "./model/"
model_path_drive = "./model/"
#训练数据集路径
train_data_path = "./datasets/train/train.csv"
#测试数据集路径
test_data_path = "./datasets/test/test.csv"
#提交结果文件的路径
submission_path = "./submission/"
#平均损失值的打印输出迭代间隔
interval1 = 1
#保存模型的迭代间隔
interval2 = 1
#是否画损失值和迭代次数的散点图
isShowScatter = False
# #是否展示图片预测结构
# # isShowPhoto = True
# isShowPhoto = False
