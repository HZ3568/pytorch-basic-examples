import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 1. 准备数据
batch_size = 64

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 2. 定义CNN模型
class CNNNet(torch.nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()

        # 卷积层1：输入1通道，输出10通道，卷积核5x5
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)

        # 卷积层2：输入10通道，输出20通道，卷积核5x5
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)

        # 池化层：2x2最大池化
        self.pool = torch.nn.MaxPool2d(kernel_size=2)

        # 全连接层
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # 输入: [batch_size, 1, 28, 28]

        x = self.pool(F.relu(self.conv1(x)))
        # conv1后: [batch_size, 10, 24, 24]
        # pooling后: [batch_size, 10, 12, 12]

        x = self.pool(F.relu(self.conv2(x)))
        # conv2后: [batch_size, 20, 8, 8]
        # pooling后: [batch_size, 20, 4, 4]

        x = x.view(-1, 20 * 4 * 4)   # 展平为 [batch_size, 320]
        x = self.fc(x)               # 输出10类分数

        return x


# 3. 创建模型
model = CNNNet()

# 4. 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# 5. 训练函数
def train(epoch):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss = {loss.item():.4f}')


# 6. 测试函数
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


# 7. 主程序
if __name__ == '__main__':
    for epoch in range(1, 6):
        train(epoch)
        test()