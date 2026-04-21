import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# ======================
# 1. 准备数据
# ======================
batch_size = 64
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ======================
# 2. 定义模型: 全连接神经网络
# ======================
class FCNet(torch.nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        # x原始形状: [batch_size, 1, 28, 28]
        x = x.view(-1, 28 * 28)  # 拉平成 [batch_size, 784]
        x = F.relu(self.fc1(x))  # 第一层 + ReLU
        x = F.relu(self.fc2(x))  # 第二层 + ReLU
        x = self.fc3(x)  # 输出10个类别分数
        return x

model = FCNet()

# ======================
# 3. 定义损失函数和优化器
# ======================
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.025)


# ======================
# 4. 训练模型
# ======================
def train(epoch):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # 前向传播
        outputs = model(inputs)  # 10个类别的原始分数
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss = {loss.item():.4f}')


# ======================
# 5. 测试模型
# ======================
def test():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    for epoch in range(1, 6):
        train(epoch)
        test()
