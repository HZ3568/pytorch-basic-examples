import numpy as np
import matplotlib.pyplot as plt

# ======================
# 1. 准备数据: y = 2 * x
# ======================
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# y_data = [5.0, 8.0, 11.0]

# 初始化参数 w（模型参数）
w = 0

# 用于记录训练过程中每一轮的 (w, loss)
w_history = []
loss_history = []


# ======================
# 2. 前向传播，线性模型：y = w * x
# ======================
def forward(x):
    return x * w


# ======================
# 3. 定义损失函数（均方误差 MSE）
# ======================
def cost(xs, ys):
    total_loss = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        total_loss += (y_pred - y) ** 2
    return total_loss / len(xs)


# ======================
# 4. 手动计算梯度: loss = (x * w - y) ^ 2  ->  grad = d(loss) / d(w) = 2 * x * (x * w - y)
# ======================
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


# ======================
# 5. 训练前预测
# ======================
print('Predict (before training)', 4, forward(4))

# ======================
# 6. 梯度下降训练过程
# ======================
for epoch in range(100):
    MSE = cost(x_data, y_data)
    # 记录当前的 (w, loss)，用于后续可视化
    w_history.append(w)
    loss_history.append(MSE)

    grad_w = gradient(x_data, y_data)  # 计算梯度
    w -= 0.2 * grad_w  # 更新参数：w = w - 学习率 * 梯度
    print("Epoch:", epoch + 1, "w =", w, "loss =", MSE)

# ======================
# 7. 训练后预测
# ======================
print('Predict (after training)', 4, forward(4))

# ======================
# 8. 计算完整的损失函数曲线（枚举不同的 w，计算对应的 loss）
# ======================
w_range = np.arange(0.0, 4.1, 0.1)
loss_curve = []

for w_temp in w_range:
    total_cost = 0
    for x, y in zip(x_data, y_data):
        y_pred = x * w_temp
        total_cost += (y_pred - y) ** 2
    loss_curve.append(total_cost / len(x_data))

# ======================
# 9. 可视化
# ======================
plt.figure(figsize=(8, 6))
plt.plot(w_range, loss_curve, label='Loss Curve')
plt.scatter(w_history, loss_history, label='Gradient Descent Points')
plt.plot(w_history, loss_history)
plt.xlabel('w')
plt.ylabel('loss')
plt.title('Gradient Descent on Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
