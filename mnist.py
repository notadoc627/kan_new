import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
from model.MLP import MnistMLP
from model.CNN import MnistCNN
from efficient_kan.kan import KAN
from torchinfo import summary

# 计算模型的参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 加载 MNIST 数据集并按 7:1:2 划分
def load_mnist():
    dataset = datasets.MNIST(root='./data', train=True,
                             transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  transform=transforms.ToTensor())

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

# 训练和测试函数
def train_and_test(model, train_loader, val_loader, test_loader, epochs=100, device='cpu'):
    model = model.to(device)  # 将模型移动到指定设备
    criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数移动到指定设备
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # total_params = count_parameters(model)
    # print(f"Model parameters: {total_params}")
    # print(f"Model structure:\n{model}")  # 打印模型结构

    total_train_time = 0
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 28*28).to(device)  # 将数据移动到指定设备
            target = target.to(device)  # 将标签移动到指定设备
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        end_time = time.time()
        if epoch < 30:
            total_train_time += (end_time - start_time)

        # 验证集上的准确率
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.view(-1, 28*28).to(device)  # 将数据移动到指定设备
                target = target.to(device)  # 将标签移动到指定设备
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        val_accuracy = 100 * val_correct / val_total
        print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_accuracy}%")

    if epochs > 30:
        print(f"Total training time for the first 30 epochs: {total_train_time} seconds")
    else:
        print(f"Total training time for {epochs} epochs: {total_train_time} seconds")

    # 测试集上的准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 28*28).to(device)  # 将数据移动到指定设备
            target = target.to(device)  # 将标签移动到指定设备
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

# 主函数
if __name__ == "__main__":
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 MNIST 数据集
    mnist_train_loader, mnist_val_loader, mnist_test_loader = load_mnist()
    input_size = 28 * 28
    hidden_size = 128
    output_size = 10

    # 训练和测试 MLP 模型
    mlp = MnistMLP(input_size, hidden_size, output_size)
    print("Training and testing MLP on MNIST...")
    # train_and_test(mlp, mnist_train_loader, mnist_val_loader, mnist_test_loader)

    # 训练和测试 CNN 模型
    cnn = MnistCNN()
    print("Training and testing CNN on MNIST...")
    #train_and_test(cnn, mnist_train_loader, mnist_val_loader, mnist_test_loader)

    kan = KAN(layers_hidden=[28*28, 64, 10], scale_base=0.3)
    print("Training and testing KAN on MNIST...")
    summary(kan, input_size=(28*28,))
    train_and_test(kan, mnist_train_loader, mnist_val_loader, mnist_test_loader, device=device)