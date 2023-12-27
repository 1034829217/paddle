import os
import math
import argparse
import paddle
import paddle.optimizer as optim
from tensorboardX import SummaryWriter

from model import *
from utils import *


def hashing_loss(b, cls, m, alpha):
    """
    compute hashing loss
    automatically consider all n^2 pairs
    """
    y = paddle.reshape((cls.unsqueeze(0) != cls.unsqueeze(1)).astype('float32'), [-1])
    dist = ((b.unsqueeze(0) - b.unsqueeze(1)) ** 2).sum(axis=2).reshape([-1])
    loss = (1 - y) / 2 * dist + y / 2 * paddle.clip(m - dist, min=0)
    loss = loss.mean() + alpha * paddle.sum(paddle.abs(b.abs() - 1), axis=1).mean() * 2
    return loss


def train(epoch, dataloader, net, optimizer, m, alpha):
    accum_loss = 0
    loss_print = 0
    net.train()
    for i, (img, cls) in enumerate(dataloader):
        img, cls = [paddle.to_tensor(x) for x in (img, cls)]

        optimizer.clear_grad()
        b = net(img)
        loss = hashing_loss(b, cls, m, alpha)


        loss.backward()
        optimizer.step()
        # print(loss)
        if not math.isnan(float(loss)) :
            accum_loss += float(loss)
            loss_print = float(loss)


        print(f'[{epoch}][{i}/{len(dataloader)}] loss: {loss_print:.4f}')
    return accum_loss / len(dataloader)


def test(epoch, dataloader, net, m, alpha):
    accum_loss = 0
    net.eval()
    for img, cls in dataloader:
        img, cls = [paddle.to_tensor(x) for x in (img, cls)]

        b = net(img)
        loss = hashing_loss(b, cls, m, alpha)
        if not math.isnan(float(loss)) :
            accum_loss += float(loss)


    accum_loss /= len(dataloader)
    print(f'[{epoch}] val loss: {accum_loss:.4f}')
    return accum_loss


def main():
    parser = argparse.ArgumentParser(description='train DSH')
    parser.add_argument('--cifar', default='../dataset/cifar', help='path to cifar')
    parser.add_argument('--weights', default='', help="path to weight (to continue training)")
    parser.add_argument('--outf', default='checkpoints', help='folder to output model checkpoints')
    parser.add_argument('--checkpoint', type=int, default=50, help='checkpointing after batches')

    parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
    parser.add_argument('--ngpu', type=int, default=6, help='which GPU to use')

    parser.add_argument('--binary_bits', type=int, default=12, help='length of hashing binary')
    parser.add_argument('--alpha', type=float, default=0.01, help='weighting of regularizer')

    parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.outf, exist_ok=True)
    choose_gpu(opt.ngpu)
    feed_random_seed()
    train_loader, test_loader = init_cifar_dataloader(opt.cifar, opt.batchSize)
    logger = SummaryWriter()

    # setup net
    net = DSH(opt.binary_bits)
    resume_epoch = 0
    # print(net)
    if opt.weights:
        print(f'loading weight form {opt.weights}')
        resume_epoch = int(os.path.basename(opt.weights)[:-4])
        net.load_state_dict(paddle.load(opt.weights, map_location=lambda storage, location: storage))

    # net.cuda()

    # setup optimizer
    optimizer = optim.Adam(learning_rate=opt.lr, parameters=net.parameters(), weight_decay=0.004)

    for epoch in range(resume_epoch, opt.niter):
        train_loss = train(epoch, train_loader, net, optimizer, 2 * opt.binary_bits, opt.alpha)
        logger.add_scalar('train_loss', train_loss, epoch)

        test_loss = test(epoch, test_loader, net, 2 * opt.binary_bits, opt.alpha)
        logger.add_scalar('test_loss', test_loss, epoch)

        # if epoch % opt.checkpoint == 0:
        #     # compute mAP by searching testset images from trainset
        #     trn_binary, trn_label = compute_result(train_loader, net)
        #     tst_binary, tst_label = compute_result(test_loader, net)
        #     mAP = compute_mAP(trn_binary, tst_binary, trn_label, tst_label)
        #     print(f'[{epoch}] retrieval mAP: {mAP:.4f}')
        #     logger.add_scalar('retrieval_mAP', mAP, epoch)

        #     # save checkpoints
        #     paddle.save(net.state_dict(), os.path.join(opt.outf, f'{epoch:03d}.pth'))


if __name__ == '__main__':
    main()
