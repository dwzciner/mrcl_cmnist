import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
def filter_mnist(dataset, num_per_class=20):
    data = dataset.data
    targets = dataset.targets
    selected_data = []
    selected_labels = []
    class_count = {i: 0 for i in range(10)}

    for i in range(len(targets)):
        label = targets[i].item()
        if class_count[label] < num_per_class:
            selected_data.append(data[i])
            selected_labels.append(label)
            class_count[label] += 1
        if all(count == num_per_class for count in class_count.values()):
            break  # 每类达到 num_per_class 张后停止

    # 转换为 Tensor 并创建新的 MNIST 数据集
    selected_data = torch.stack(selected_data)
    selected_labels = torch.tensor(selected_labels)

    new_dataset = datasets.MNIST(
        root=dataset.root,
        train=dataset.train,
        transform=dataset.transform,
        target_transform=dataset.target_transform,
    )
    new_dataset.data = selected_data
    new_dataset.targets = selected_labels

    return new_dataset

# 定义转换，将单通道图像转换为三通道
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 复制 3 次变成 RGB
    transforms.Lambda(lambda x: x * torch.tensor([1.0, 0.1, 0.1]).view(3, 1, 1)),  # 颜色变换
])

# 加载 MNIST 训练集
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
mnist_train = filter_mnist(mnist_train,20)
# 创建 DataLoader
iterator_train = torch.utils.data.DataLoader(mnist_train, batch_size=5, shuffle=True)

# 取出一张图片
for img, _ in iterator_train:
    print(img.size())  # (5,3,28,28)

    img = img[0]  # 取第一张图
    img = img.permute(1, 2, 0)  # 调整维度顺序 (C, H, W) -> (H, W, C)

    # 确保数据是 float 类型，并且在 0-1 之间
    img = img.numpy().astype("float32")

    # 显示彩色图片
    plt.imshow(img)
    plt.axis("off")  # 关闭坐标轴
    plt.show()
    break  # 只显示一张图片
