一、dataset.py 文件学习笔记  
功能概述：定义了一个名为 ImageTxtDataset 的 PyTorch 数据集类，用于加载图像数据和对应的标签。  
关键点：  
初始化方法：  
接收 txt_path（包含图像路径和标签的文本文件路径）、folder_name（图像所在文件夹名称）和 transform（图像预处理操作）作为参数。  
通过读取 txt_path 文件，将图像路径和标签分别存储在 self.imgs_path 和 self.labels 列表中。  
注释掉的 img_path = os.path.join(self.data_dir, self.folder_name, img_path) 行，可能是之前用于构建完整图像路径的代码，但目前未被使用。  
__len__ 方法：返回数据集中的图像数量，即 self.imgs_path 列表的长度。  
__getitem__ 方法：  
根据索引 i 获取对应的图像路径和标签。  
使用 PIL.Image.open 打开图像，并将其转换为 RGB 模式。  
如果定义了 transform，则对图像应用预处理操作。  
返回处理后的图像及其标签。  
二、deal_with_datasets.py 文件学习笔记  
功能概述：用于将原始数据集划分为训练集和验证集，并将相应的图像移动到对应的文件夹中。  
关键点：  
随机种子设置：通过 random.seed(42) 确保每次运行代码时划分结果具有可重复性。  
路径设置：  
dataset_dir 是原始数据集的路径。  
train_dir 和 val_dir 分别是训练集和验证集的输出路径。  
训练集和验证集划分：  
遍历 dataset_dir 下的每个类别文件夹。  
获取每个类别文件夹中的所有图片，并将其路径存储在 images 列表中。  
使用 train_test_split 函数按照指定的比例（train_ratio）划分训练集和验证集。  
文件夹创建与图片移动：  
为每个类别在训练集和验证集目录下创建子文件夹。  
将划分后的训练集和验证集图片分别移动到对应的文件夹中。  
最后删除原始类别文件夹。  
三、nn_relu.py 文件学习笔记  
功能概述：定义了一个简单的神经网络模型，使用 ReLU 激活函数，并通过 TensorBoard 可视化输入和输出图像。  
关键点：  
数据加载：  
使用 torchvision.datasets.CIFAR10 加载 CIFAR-10 数据集，设置为非训练模式，并应用 torchvision.transforms.ToTensor() 转换。  
创建 DataLoader，设置批量大小为 64。  
输入张量定义：  
定义了一个形状为 (1, 1, 2, 2) 的输入张量 input，用于测试模型。  
模型定义：  
定义了一个名为 Chen 的神经网络类，继承自 torch.nn.Module。  
在模型中定义了 ReLU 激活函数。  
在 forward 方法中，将输入数据通过 ReLU 激活函数。  
TensorBoard 可视化：  
创建 SummaryWriter，用于将数据写入 TensorBoard 日志。  
遍历 dataloader，将输入图像和经过模型处理后的输出图像写入 TensorBoard。  
模型测试：  
使用定义的输入张量 input 测试模型，打印输出结果。  
四、prepare.py 文件学习笔记  
功能概述：创建包含图像路径和标签的文本文件，分别用于训练集和验证集。  
关键点：  
函数定义：  
定义了 create_txt_file 函数，接收 root_dir（数据集根目录）和 txt_filename（输出文本文件名）作为参数。  
遍历 root_dir 下的每个类别文件夹。  
对于每个类别文件夹中的每张图片，将其路径和对应的标签（类别索引）写入文本文件。  
调用函数：  
分别为训练集和验证集调用 create_txt_file 函数，生成 train.txt 和 val.txt 文件。  
总结  
今天的学习内容主要涉及了 PyTorch 数据集的自定义、数据集的划分、简单神经网络模型的定义与可视化，以及数据准备的相关操作。通过这些代码，你掌握了如何加载和处理图像数据、构建简单的神经网络模型，并使用 TensorBoard 进行可视化，同时也学会了如何为数据集生成路径和标签信息的文本文件，为后续的模型训练和评估做好了准备。  