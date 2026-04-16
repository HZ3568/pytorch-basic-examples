import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据
x_data = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
y_data = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float32)

# 2. 定义模型
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = LogisticRegressionModel()

# 3. 定义损失和优化器
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 训练
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    print(f"epoch={epoch}, loss={loss.item():.6f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. 测试几个点
x_test = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
with torch.no_grad():
    pred = model(x_test)
    print("\n预测概率：")
    for x, p in zip(x_test, pred):
        print(f"x={x.item():.1f}, probability={p.item():.4f}")

# 6. 画概率曲线
x = np.linspace(0, 10, 200, dtype=np.float32)
x_t = torch.tensor(x).view(200, 1)

with torch.no_grad():
    y_t = model(x_t)

y = y_t.numpy()

plt.plot(x, y, label='logistic regression')
plt.plot([0, 10], [0.5, 0.5], 'r--', label='decision boundary (p=0.5)')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.legend()
plt.show()