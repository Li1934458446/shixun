一、train_alex.py 文件学习笔记
1. 功能概述
该文件实现了一个基于 AlexNet 的图像分类训练程序，使用自定义的 ImageTxtDataset 数据集类，并通过 TensorBoard 进行训练过程的可视化。
2. 关键点
（1）数据集加载
自定义数据集类：
使用 ImageTxtDataset，从 train.txt 和 val.txt 文件中读取图像路径和标签。
数据预处理包括 Resize、RandomHorizontalFlip、ToTensor 和 Normalize。
训练集和测试集的路径分别为：
训练集：D:\dataset\train.txt 和 D:\dataset\image2\train
测试集：D:\dataset\train.txt 和 D:\dataset\image2\train（注意：测试集路径应为验证集路径，可能是代码中的错误）。
数据加载器：
使用 DataLoader，训练集设置 shuffle=True，测试集设置 shuffle=False，批量大小均为 64。
（2）模型定义
AlexNet 模型：
定义了一个包含 5 个卷积层和 3 个全连接层的 AlexNet 模型。
卷积层使用了 MaxPool2d 进行下采样。
全连接层的输出维度为 10（对应 CIFAR-10 数据集的类别数）。
（3）训练过程
损失函数和优化器：
使用 CrossEntropyLoss 作为损失函数。
使用 SGD 优化器，学习率为 0.01，动量为 0.9。
训练循环：
训练轮数为 10 轮。
每 500 步记录一次训练损失，并写入 TensorBoard。
每轮训练结束后，计算测试集上的损失和准确率，并写入 TensorBoard。
每轮训练结束后保存模型。
（4）性能评估
测试集评估：
在测试集上计算总损失和准确率。
使用 argmax 获取预测类别，并与真实标签进行比较计算准确率。
时间记录：
记录每轮训练的时间。
（5）模型保存
每轮训练结束后，将模型保存到 model_save 文件夹中，文件名为 alexnet_{i}.pth。
（6）TensorBoard 可视化
使用 SummaryWriter 将训练损失、测试损失和测试准确率写入日志文件，方便通过 TensorBoard 进行可视化。
二、transformer.py 文件学习笔记
1. 功能概述
该文件实现了一个基于 Transformer 的视觉模型（ViT），用于处理序列数据。模型包括多头自注意力机制和前馈网络。
2. 关键点
（1）模块定义
FeedForward 模块：
包括 LayerNorm、Linear、GELU 和 Dropout。
用于 Transformer 中的前馈网络部分。
Attention 模块：
实现了多头自注意力机制。
包括查询（Q）、键（K）和值（V）的线性变换，以及 Softmax 和 Dropout。
使用 einops 的 rearrange 函数对张量进行重新排列。
Transformer 模块：
包含多个 Transformer 层，每层由一个注意力模块和一个前馈模块组成。
每个模块的输出会与输入相加（残差连接）。
ViT 模块：
定义了一个视觉 Transformer 模型。
包括：
将输入序列划分为固定大小的块（patch），并将其嵌入到高维空间。
添加位置嵌入和类别令牌（cls token）。
通过 Transformer 编码器处理序列。
使用全连接层输出最终的类别预测。
（2）模型结构
输入处理：
使用 Rearrange 将输入序列划分为块（patch）。
使用 LayerNorm 和 Linear 将块嵌入到高维空间。
位置嵌入：
添加可学习的位置嵌入，以保留序列中的位置信息。
Transformer 编码器：
包含多个 Transformer 层，每层由多头自注意力模块和前馈模块组成。
输出层：
使用 LayerNorm 和 Linear 将 Transformer 的输出映射到类别空间。
（3）测试代码
在 __main__ 中创建了一个 ViT 模型实例，并测试了其对随机时间序列数据的处理。
输入形状为 (4, 3, 256)，表示 4 个样本，每个样本有 3 个通道，序列长度为 256。
输出形状为 (4, 1000)，表示每个样本的类别预测概率。
总结
今天的学习内容涵盖了两个主要部分：
AlexNet 训练程序：
学习了如何使用 PyTorch 实现一个完整的训练流程，包括数据加载、模型定义、训练循环、性能评估和模型保存。
掌握了如何使用 TensorBoard 进行训练过程的可视化。
了解了 AlexNet 的网络结构及其在图像分类任务中的应用。
Transformer 模型实现：
学习了 Transformer 的基本概念，包括多头自注意力机制和前馈网络。
掌握了如何使用 einops 库进行张量操作。
了解了 ViT 模型的结构及其在处理序列数据中的应用。
通过这两个文件的学习，你不仅加深了对深度学习模型的理解，还掌握了如何实现和训练这些模型，为后续的项目开发和研究打下了坚实的基础。