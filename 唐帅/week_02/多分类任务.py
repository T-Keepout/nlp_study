import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return self.softmax(x)


# 生成样本
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.01

    # 建立模型
    model = TorchModel(input_size)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

    # 保存模型
    torch.save(model.state_dict(), "model.bin")


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
        predictions = torch.argmax(result, dim=1)

    for vec, pred, prob in zip(input_vec, predictions, result):
        print(
            "输入：%s, 预测类别：%d, 各类别概率：%s"
            % (vec, pred.item(), prob.numpy().round(4))
        )


if __name__ == "__main__":
    main()

    # 测试代码
    test_vec = [
        [0.1, 0.8, 0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.9],
        [0.2, 0.3, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.2, 0.6, 0.1],
    ]

    predict("model.bin", test_vec)
