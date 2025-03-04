import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time
from model.MLP import CIFARMLP
from model.CNN import CIFARCNN
from efficient_kan.kan import KAN, KANLinear, KAN_with_flatten
from torchinfo import summary

# 计算模型的参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 加载 CIFAR - 10 数据集并按 7:1:2 划分
def load_cifar10():
    dataset = datasets.CIFAR10(root='./data', train=True,
                               transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
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
def train_and_test(model, train_loader, val_loader, test_loader, epochs=100):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    total_params = count_parameters(model)
    print(f"Model parameters: {total_params}")

    total_train_time = 0
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)  # 将数据移动到指定设备
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
                data = data.to(device)  # 将数据移动到指定设备
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
            data = data.to(device)  # 将数据移动到指定设备
            target = target.to(device)  # 将标签移动到指定设备
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

# class KANBlock(nn.Module):
#     def __init__(self, in_features, out_features, num_layers, **kwargs):
#         super(KANBlock, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_layers = num_layers
#         self.block = nn.Sequential(*[KANLinear(in_features if i == 0 else out_features, out_features, **kwargs) for i in range(num_layers)])
#         self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)

#     def forward(self, x, update_grid=False):
#         identity = x
#         for layer in self.block:
#             if update_grid:
#                 layer.update_grid(x)
#             x = layer(x)
#         x += self.shortcut(identity)
#         return x

# class KANResNet(nn.Module):
#     def __init__(self, block_configs, **kwargs):
#         super(KANResNet, self).__init__()
#         self.blocks = nn.ModuleList()
#         for config in block_configs:
#             in_features, out_features, num_layers = config
#             self.blocks.append(KANBlock(in_features, out_features, num_layers, **kwargs))
    
#     def forward(self, x, update_grid=False):
#         x = x.view(x.size(0), -1)  # 展平输入
#         for block in self.blocks:
#             x = block(x, update_grid)
#         return x

#     def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
#         return sum(
#             block.block[i].regularization_loss(regularize_activation, regularize_entropy)
#             for block in self.blocks
#             for i in range(block.num_layers)
#         )
# 主函数
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 CIFAR - 10 数据集
    cifar10_train_loader, cifar10_val_loader, cifar10_test_loader = load_cifar10()
    input_size = 3 * 32 * 32
    output_size = 10

    # 训练和测试 MLP 模型
    mlp = CIFARMLP()
    print("Training and testing MLP on CIFAR - 10...")
    # train_and_test(mlp, cifar10_train_loader, cifar10_val_loader, cifar10_test_loader)

    # 训练和测试 CNN 模型
    cnn = CIFARCNN()
    print("Training and testing CNN on CIFAR - 10...")
    # train_and_test(cnn, cifar10_train_loader, cifar10_val_loader, cifar10_test_loader)

    # kan = KAN(layers_hidden=[input_size, 128, 64, 10])
    # input_dim = 32 * 32 * 3  # CIFAR-10 图像的展平尺寸
    # num_classes = 10  # CIFAR-10 数据集的类别数
    # block_configs = [
    #     (input_dim, 256, 2),  # 第一个 block
    #     (256, 512, 2),        # 第二个 block
    #     (512, 1024, 2),        # 第三个 block
    #     (1024, num_classes, 1) # 最后一个 block
    # ]

    #kan_model = KANResNet(block_configs, grid_size=5, spline_order=3)
    layers_hidden = [input_size, 128, 64, output_size]  # 根据需要调整隐藏层
    kan = KAN_with_flatten(layers_hidden=[input_size, 128, 64, output_size])
    print("Training and testing KAN on MNIST...")
    #summary(kan_model, input_size=(input_size,))
    train_and_test(kan, cifar10_train_loader, cifar10_val_loader, cifar10_test_loader)