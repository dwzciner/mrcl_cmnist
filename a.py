import torch
import torchvision.transforms as transforms
from torchvision.datasets import Omniglot
from torch.utils.data import Dataset, DataLoader, random_split
import argparse
import logging

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import datasets.datasetfactory as df
import datasets.task_sampler as ts
import configs.classification.class_parser as class_parser
import model.modelfactory as mf
import utils.utils as utils
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification

logger = logging.getLogger('experiment')

# ----------------------
# 1. 加载 Omniglot 数据集
# ----------------------
def load_omniglot(root='./data', background=True):
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])
    dataset = Omniglot(root=root, background=background, transform=transform, download=True)
    return dataset

# ----------------------
# 2. 创建数据增强环境
# ----------------------
def make_environment(images, labels, e):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()

    # 生成二分类标签（假设前 800 类设为 0，后 823 类设为 1）
    labels = (labels >= 450).int()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))  # 25% 误标

    # 颜色翻转，概率 e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))

    # 创建彩色图像通道
    images = torch.stack([images, images], dim=1)  # [batch_size, 2, 84, 84]
    images[torch.arange(len(images)), (1 - colors).long(), :, :] *= 0

    return images.float(), labels[:, None]

# ----------------------
# 3. 定义 Dataset 类
# ----------------------
class OmniglotDataset(Dataset):
    def __init__(self, dataset, e=0.2):
        self.e = e
        self.images, self.labels = self.process_dataset(dataset)

    def process_dataset(self, dataset):
        images, labels = zip(*[dataset[i] for i in range(len(dataset))])
        images = torch.stack(images).squeeze(1) * 255  # 转换到 [0,255]
        labels = torch.tensor(labels)
        images, labels = make_environment(images, labels, self.e)  # 处理环境
        return images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]




def main():
    p = class_parser.Parser()
    total_seeds = len(p.parse_known_args()[0].seed)
    rank = p.parse_known_args()[0].rank
    all_args = vars(p.parse_known_args()[0])
    print("All args = ", all_args)

    args = utils.get_run(vars(p.parse_known_args()[0]), rank)

    utils.set_seed(args['seed'])

    my_experiment = experiment(args['name'], args, "../results/", commit_changes=False, rank=0, seed=1)
    writer = SummaryWriter(my_experiment.path + "tensorboard")

    logger = logging.getLogger('experiment')

    # ----------------------
    # 4. 划分训练集、验证集、测试集
    # ----------------------
    omniglot_train = load_omniglot(background=True)   # 训练集（964 类，每类 20 张）
    omniglot_test = load_omniglot(background=False)   # 测试集（659 类，每类 20 张）

    # **随机划分训练集和验证集**
    # train_size = int(0.75 * len(range(100)))
    train_size = int(0.75 * len(omniglot_train))
    val_size = len(omniglot_train) - train_size
    train_dataset, val_dataset = random_split(omniglot_train, [train_size, val_size])


    # **封装数据集**
    train_dataset = OmniglotDataset(train_dataset, e=0.2)
    val_dataset = OmniglotDataset(val_dataset, e=0.1)
    test_dataset = OmniglotDataset(omniglot_test, e=0.9)

    # ----------------------
    # 5. 创建 DataLoader
    # ----------------------
    batch_size = 20

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # for img, data in train_loader:
    #     print(img,data)
    #     print(data[0].item())
    #     break
    config = mf.ModelFactory.get_model("na", 'comni', output_dimension=1000)

    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')

    maml = MetaLearingClassification(args, config).to(device)

    for step in range(args['steps']):
        x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(train_loader, val_loader,
                                                               steps=args['update_step'], reset=not args['no_reset'])
        if torch.cuda.is_available():
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

        accs, loss = maml(x_spt, y_spt, x_qry, y_qry)

        # Evaluation during training for sanity checks
        if step % 40 == 5:
            writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)
            logger.info('step: %d \t training acc %s', step, str(accs))
        if step % 300 == 3:
            utils.log_accuracy(maml, my_experiment, test_loader, device, writer, step)
            utils.log_accuracy(maml, my_experiment, train_loader, device, writer, step)



if '__main__' == main():
    main()




