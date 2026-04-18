import numpy as np
import matplotlib.pyplot as plt

# 让 matplotlib 支持 3D 作图
from mpl_toolkits.mplot3d import Axes3D

# ======================
# 1. 准备数据: y = 3 * x + 2
# ======================
x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 8.0, 11.0]

# 初始化参数
w = 0.0
b = 0.0

# 记录梯度下降轨迹
w_history = []
b_history = []
loss_history = []

# ======================
# 2. 前向传播
# ======================
def forward(x):
    return w * x + b

# ======================
# 3. 损失函数 MSE
# ======================
def cost(xs, ys):
    total_loss = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        total_loss += (y_pred - y) ** 2
    return total_loss / len(xs)

# 给画曲面单独写一个函数：传入指定 w,b 计算 loss
def cost_for_params(w_val, b_val, xs, ys):
    total_loss = 0
    for x, y in zip(xs, ys):
        y_pred = w_val * x + b_val
        total_loss += (y_pred - y) ** 2
    return total_loss / len(xs)

# ======================
# 4. 计算梯度
# ======================
def gradient(xs, ys):
    grad_w = 0
    grad_b = 0
    for x, y in zip(xs, ys):
        error = w * x + b - y
        grad_w += 2 * x * error
        grad_b += 2 * error
    grad_w /= len(xs)
    grad_b /= len(xs)
    return grad_w, grad_b

# ======================
# 5. 训练
# ======================
lr = 0.15
epochs = 50

print('Predict (before training)', 4, forward(4))

for epoch in range(epochs):
    mse = cost(x_data, y_data)

    w_history.append(w)
    b_history.append(b)
    loss_history.append(mse)

    grad_w, grad_b = gradient(x_data, y_data)
    w -= lr * grad_w
    b -= lr * grad_b

    print(f"Epoch: {epoch+1}, w = {w:.4f}, b = {b:.4f}, loss = {mse:.6f}")

print('Predict (after training)', 4, forward(4))

# ======================
# 6. 构造 (w, b, loss) 三维损失曲面
# ======================
w_range = np.arange(0.0, 5.1, 0.1)
b_range = np.arange(0.0, 5.1, 0.1)

W, B = np.meshgrid(w_range, b_range)
Loss = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Loss[i, j] = cost_for_params(W[i, j], B[i, j], x_data, y_data)

# ======================
# 7. 画三维损失曲面 + 梯度下降轨迹
# ======================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 曲面
ax.plot_surface(W, B, Loss, alpha=0.7, cmap='viridis')

# 梯度下降轨迹
ax.plot(w_history, b_history, loss_history, color='red', marker='o', label='Gradient Descent Path')

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('loss')
ax.set_title('3D Loss Surface: loss = f(w, b)')
ax.legend()

plt.show()