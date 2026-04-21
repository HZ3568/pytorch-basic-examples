import torch

# ======================
# 1. 准备数据
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
        y_pred = self.linear(x)
        return y_pred


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
for epoch in range(50):
    # 前向传播
    y_pred = model(x_data)

    # 计算损失
    loss = criterion(y_pred, y_data)

    print(f"Epoch = {epoch + 1}, loss = {loss.item():.4f}")

    # 梯度清零, 避免梯度累加
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 参数更新（w 和 b）
    optimizer.step()

# ======================
# 5. 测试模型
# ======================
print(f"w = {model.linear.weight.item():.4f}")
print(f"b = {model.linear.bias.item():.4f}")

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
