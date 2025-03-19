import argparse
import logging
from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import datasets.datasetfactory as df
import datasets.task_sampler as ts
import configs.classification.class_parser as class_parser
import model.modelfactory as mf
import utils.utils as utils
from experiment.experiment import experiment
from model.meta_learner import MetaLearingClassification
import model.learner as Learner
logger = logging.getLogger('experiment')

# 筛选数据集，每类保留 num_per_class 张图片，返回 MNIST 类型
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

class RepeatChannel:
    def __init__(self, train):
        self.train = train
    def __call__(self, x):
        if self.train:
            return x.repeat(3, 1, 1) * torch.tensor([0.1, 0.1, 1.0]).view(3, 1, 1)
        else:
            return x.repeat(3, 1, 1) * torch.tensor([1.0, 0.1, 0.1]).view(3, 1, 1)

def load_model(args, config):
    if args['model_path'] is not None:
        net_old = Learner.Learner(config)
        # logger.info("Loading model from path %s", args["model_path"])
        net = torch.load(args['model_path'],
                         map_location="cpu")

        for (n1, old_model), (n2, loaded_model) in zip(net_old.named_parameters(), net.named_parameters()):
            # print(n1, n2, old_model.adaptation, old_model.meta)
            loaded_model.adaptation = old_model.adaptation
            loaded_model.meta = old_model.meta

        net.reset_vars()
    else:
        net = Learner.Learner(config)
    return net


def eval_iterator(iterator, device, maml):
    correct = 0
    for img, target in iterator:
        img = img.to(device)
        target = target.to(device)
        logits_q = maml(img)

        pred_q = (logits_q).argmax(dim=1)

        correct += torch.eq(pred_q, target).sum().item() / len(img)
    return correct / len(iterator)


def test(args, config, device,test_iterator):

    maml = load_model(args, config)
    maml.to(device)
    correct = eval_iterator(test_iterator,device,maml)
    return correct


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

    # Using first 963 classes of the omniglot as the meta-training set
    args['classes'] = list(range(10))
    # 3 4 5
    args['traj_classes'] = list(range(5, 10))

    # dataset = df.DatasetFactory.get_dataset(args['dataset'], background=True, train=True, path=args["path"], all=True)
    # dataset_test = df.DatasetFactory.get_dataset(args['dataset'], background=True, train=False, path=args["path"],
    #                                              all=True)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        RepeatChannel(train=True),  # 用类代替 Lambda
        # transforms.Lambda(lambda x: x * torch.tensor([0.1, 0.1, 1.0]).view(3, 1, 1)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        RepeatChannel(train=False),  # 用类代替 Lambda
        # transforms.Lambda(lambda x: x * torch.tensor([1.0, 0.1, 0.1]).view(3, 1, 1)),
    ])
    # 加载 MNIST 数据集
    mnist_train = datasets.MNIST(root="../data", train=True, download=True, transform=transform_train)
    mnist_test = datasets.MNIST(root="../data", train=False, download=True, transform=transform_test)

    train_size = int(0.75 * len(mnist_train))  # 80% 作为训练集
    val_size = len(mnist_train) - train_size  # 剩下的 20% 作为验证集

    # 随机划分数据集
    mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [train_size, val_size])

    # 过滤数据，每类保留 20 张
    mnist_train = filter_mnist(mnist_train.dataset, num_per_class=20)
    mnist_val = filter_mnist(mnist_val.dataset, num_per_class=20)
    mnist_test = filter_mnist(mnist_test, num_per_class=20)





    # Iterators used for evaluation
    iterator_val = torch.utils.data.DataLoader(mnist_val, batch_size=5,
                                                shuffle=True, num_workers=1)

    iterator_train = torch.utils.data.DataLoader(mnist_train, batch_size=5,
                                                 shuffle=True, num_workers=1)

    iterator_test = torch.utils.data.DataLoader(mnist_test, batch_size=32,
                                                shuffle=True, num_workers=1)


    sampler = ts.SamplerFactory.get_sampler(args['dataset'], args['classes'], mnist_train, mnist_val)

    config = mf.ModelFactory.get_model("na", args['dataset'], output_dimension=10)

    gpu_to_use = rank % args["gpus"]
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')

    maml = MetaLearingClassification(args, config).to(device)

    for step in range(args['steps']):

        t1 = np.random.choice(args['traj_classes'], args['tasks'], replace=False)

        d_traj_iterators = []
        for t in t1:
            d_traj_iterators.append(sampler.sample_task([t]))

        d_rand_iterator = sampler.get_complete_iterator()

        x_spt, y_spt, x_qry, y_qry = maml.sample_training_data(d_traj_iterators, d_rand_iterator,
                                                               steps=args['update_step'], reset=not args['no_reset'])
        if torch.cuda.is_available():
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

        accs, loss = maml(x_spt, y_spt, x_qry, y_qry)
        writer.add_scalar('/metatrain/train/loss', loss[-1], step)
        # Evaluation during training for sanity checks
        if step % 10 == 5:
            writer.add_scalar('/metatrain/train/accuracy', accs[-1], step)

            logger.info('step: %d \t training acc %s', step, str(accs))
        if step % 20 == 3:
            # utils.log_accuracy(maml, my_experiment, iterator_test, device, writer, step)
            utils.log_accuracy(maml, my_experiment, iterator_train, device, writer, step)

    correct = test(args, config, device, iterator_test)
    logger.info('step: %d \t test acc %s', step, str(correct))


if __name__ == '__main__':
    main()
