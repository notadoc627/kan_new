import torch
from torchinfo import summary
from efficient_kan.kan import KAN

# 定义一个 KAN 模型实例
kan = KAN(
    layers_hidden=[28*28, 128, 10],  # 输入维度为 3，隐藏层维度为 64 和 128，输出维度为 1
)

# 打印模型摘要
summary(kan, input_size=(28*28,))