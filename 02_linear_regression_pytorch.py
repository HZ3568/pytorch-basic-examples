import torch

# ======================
# 1. 准备数据集
# ======================
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# ======================
# 2. 定义模型
# ======================
# 继承 torch.nn.Module，自定义线性回归模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # 定义一个线性层：输入1维，输出1维（y = wx + b）
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 前向传播：输入x，输出预测值y_pred
        y_pred = self.linear(x)
        return y_pred


# 实例化模型
model = LinearModel()


# ======================
# 3. 定义损失函数和优化器
# ======================
# 均方误差损失（MSE）
criterion = torch.nn.MSELoss(reduction='sum')

# 随机梯度下降（SGD）优化器，学习率0.025
optimizer = torch.optim.SGD(model.parameters(), lr=0.025)


# ======================
# 4. 训练模型
# ======================
for epoch in range(100):
    # 前向传播：计算预测值
    y_pred = model(x_data)

    # 计算损失（预测值 vs 真实值）
    loss = criterion(y_pred, y_data)

    # 打印当前轮数和损失
    print(epoch, loss.item())

    # 梯度清零（避免梯度累加）
    optimizer.zero_grad()

    # 反向传播：计算梯度
    loss.backward()

    # 参数更新（w 和 b）
    optimizer.step()


# ======================
# 5. 输出训练结果
# ======================
# 输出学到的权重 w 和偏置 b
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())


# ======================
# 6. 测试模型
# ======================
# 测试输入 x=4
x_test = torch.Tensor([[4.0]])

# 预测结果
y_test = model(x_test)

print('y_pred = ', y_test.data)