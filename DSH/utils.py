import time
from functools import wraps
import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.vision.datasets import Cifar10
import paddle.vision.transforms as transforms


def init_cifar_dataloader(root, batchSize):
    """load dataset"""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_loader = DataLoader(Cifar10(mode='train', transform=transform_train, download=True),
                              batch_size=batchSize, shuffle=True, num_workers=4)
    print(f'train set: {len(train_loader.dataset)}')
    test_loader = DataLoader(Cifar10(mode='train', transform=transform_test, download=True),
                             batch_size=batchSize * 8, shuffle=False, num_workers=4)
    print(f'val set: {len(test_loader.dataset)}')

    return train_loader, test_loader


def timing(f):
    """print time used for function f"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        time_start = time.time()
        ret = f(*args, **kwargs)
        print(f'total time = {time.time() - time_start:.4f}')
        return ret

    return wrapper


def compute_result(dataloader, net):
    bs, clses = [], []
    net.eval()
    for img, cls in dataloader:
        clses.append(cls)
        bs.append(net(paddle.to_tensor(img).stop_gradient).data.cpu())
    return paddle.sign(paddle.cat(bs)), paddle.cat(clses)


@timing
def compute_mAP(trn_binary, tst_binary, trn_label, tst_label):
    """
    compute mAP by searching testset from trainset
    https://github.com/flyingpot/pypaddle_deephash
    """
    for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

    AP = []
    Ns = paddle.arange(1, trn_binary.size(0) + 1)
    for i in range(tst_binary.size(0)):
        query_label, query_binary = tst_label[i], tst_binary[i]
        _, query_result = paddle.sum((query_binary != trn_binary).long(), dim=1).sort()
        correct = (query_label == trn_label[query_result]).float()
        P = paddle.cumsum(correct, dim=0) / Ns
        AP.append(paddle.sum(P * correct) / paddle.sum(correct))
    mAP = paddle.mean(paddle.to_tensor(AP))
    return mAP


def choose_gpu(i_gpu):
    """choose current CUDA device"""
    paddle.device.set_device(f"gpu:{i_gpu}")
    # cudnn.benchmark = True


def feed_random_seed(seed=np.random.randint(1, 10000)):
    """feed random seed"""
    np.random.seed(seed)
    paddle.seed(seed)
    # paddle.cuda.manual_seed(seed)
