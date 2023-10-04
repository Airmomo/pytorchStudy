"""
    简单分类模型
"""

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

"""
    单隐藏层的简单线性分类模型
"""


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # 隐藏层的激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


"""
    创建训练数据集
"""
# 假设您有训练数据和标签
# x_train 是训练数据，每行表示一个图像的像素值
# y_train 是相应的标签，每个标签对应于一个数字类别
x_train = torch.randn(10000, 784)  # 10000 个图像，每个图像有 784 个特征（28x28）
y_train = torch.randint(0, 10, (10000,))  # 10000 个标签，每个标签是 0 到 9 之间的数字

# 将数据转换为 PyTorch Tensor
x_train = x_train.float()  # 转换为浮点型 Tensor
y_train = y_train.long()  # 转换为长整型 Tensor

# 创建 PyTorch 的 Dataset 对象
train_dataset = TensorDataset(x_train, y_train)

# 创建 DataLoader 来批量加载数据
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

"""
    超参数
"""
# 输入数据的特征维度
input_size = 784
# 神经网络中隐藏层的神经元数量
hidden_size = 5
# 分类数量
num_classes = 10
# 学习率，控制模型参数更新步长的超参数
learning_rate = 0.01
# 训练次数
num_epochs = 100

"""
    定义模型、损失函数和优化器
"""
model = MLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 创建 TensorBoard 回调
log_dir = "logs/fit"  # 指定 TensorBoard 日志目录
# 创建一个 PyTorch 的 SummaryWriter 对象
writer = SummaryWriter()

# 训练循环
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # 正向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 记录损失到 TensorBoard
        writer.add_scalar('Loss/train', loss, global_step=epoch)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 关闭 SummaryWriter
writer.close()

