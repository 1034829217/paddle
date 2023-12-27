import os
import numpy as np
from tqdm import tqdm
import paddle
import torch
import paddle.optimizer as optim
from datasets import *
from utils_paddle import *
from models.VDSH_paddle import VDSH
import argparse

##################################################################################################

parser = argparse.ArgumentParser()
# parser.add_argument("-g", "--gpunum", help="GPU number to train the model.")
parser.add_argument("-d", "--dataset", help="Name of the dataset.")
parser.add_argument("-b", "--nbits", help="Number of bits of the embedded vector.", type=int)
parser.add_argument("--dropout", help="Dropout probability (0 means no dropout)", default=0.1, type=float)
parser.add_argument("--train_batch_size", default=100, type=int)
parser.add_argument("--test_batch_size", default=100, type=int)
parser.add_argument("--transform_batch_size", default=100, type=int)
parser.add_argument("--num_epochs", default=5, type=int)
parser.add_argument("--lr", default=0.001, type=float)

args = parser.parse_args()

# if not args.gpunum:
#     parser.error("Need to provide the GPU number.")
    
if not args.dataset:
    parser.error("Need to provide the dataset.")

if not args.nbits:
    parser.error("Need to provide the number of bits.")
        
##################################################################################################

# os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum
# print("detection>>>>>>>>", paddle.fluid.is_compiled_with_cuda())
device = paddle.device.set_device("cpu")

#########################################################################################################

# print(args.dataset)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>", paddle.device.is_compiled_with_cuda())

dataset, data_fmt = args.dataset.split('.')

if dataset in ['reuters', 'tmc', 'rcv1']:
    single_label_flag = False
else:
    single_label_flag = True
        
if single_label_flag:
    train_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
    test_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)
else:
    train_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
    test_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)

# print('计算机视觉（CV）相关数据集：', paddle.vision.datasets.__all__)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=True)

#########################################################################################################
y_dim = train_set.num_classes()
num_bits = args.nbits
num_features = train_set[0][0].size(0)

print("Train VDSH-S model ...")
print("dataset: {}".format(args.dataset))
print("numbits: {}".format(args.nbits))
# print("gpu id:  {}".format(args.gpunum))
print("dropout probability: {}".format(args.dropout))
if single_label_flag:
    print("single-label prediction.")
else:
    print("multi-label prediction.")
print("num epochs: {}".format(args.num_epochs))
print("learning rate: {}".format(args.lr))
print("num train: {} num test: {}".format(len(train_set), len(test_set)))

#########################################################################################################

model = VDSH(dataset, num_features, num_bits, dropoutProb=0.1, device=device)
print("heeeeeeeeeeeee!", type(model))
# model.to(device)


num_epochs = args.num_epochs

print(type(model.parameters()))
optimizer = optim.Adam(learning_rate=args.lr, parameters=model.parameters())
kl_weight = 0.
kl_step = 1 / 5000.

best_precision = 0
best_precision_epoch = 0

with open('logs/VDSH/loss.log.txt', 'w') as log_handle:
    log_handle.write('epoch,step,loss,reconstr_loss,kl_loss\n')
    
    for epoch in range(num_epochs):
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            print("<<<<<<<<step>>>>>>>>>", step)
            # print("xxxxxxxxxxxxxb>>>>>>>>>>>", xb, type(xb), yb, type(yb))

            # xb = xb.to(device)
            # yb = yb.to(device)

            temp1 = xb.numpy()
            temp2 = yb.numpy()
            xb = paddle.to_tensor(temp1)
            yb = paddle.to_tensor(temp2)
            # print("yyyyyyyyyyyyyyyb>>>>>>>>>>>", xb, type(xb), yb, type(yb))
            logprob_w, mu, logvar = model(xb)
            kl_loss = VDSH.calculate_KL_loss(mu, logvar)
            reconstr_loss = VDSH.compute_reconstr_loss(logprob_w, xb)
            
            loss = reconstr_loss + kl_weight * kl_loss

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

            kl_weight = min(kl_weight + kl_step, 1.)
            avg_loss.append(loss.item())
            
            log_handle.write('{},{},{:.4f},{:.4f},{:.4f}'.format(epoch, step, loss.item(), 
                                                                 reconstr_loss.item(), kl_loss.item()))
        print('{} epoch:{} loss:{:.4f} Best Precision:({}){:.3f}'.format(model.get_name(), epoch+1, np.mean(avg_loss), best_precision_epoch, best_precision))
        
        with paddle.no_grad():
            train_b, test_b, train_y, test_y = model.get_binary_code(train_loader, test_loader)
            # print("type>>>>", type(train_b), type(test_b), type(train_y), type(test_y))
            temp3 = train_b.numpy()
            temp4 = train_y.numpy()
            temp5 = test_b.numpy()
            temp6 = test_y.numpy()
            train_b = torch.tensor(temp3)
            train_y = torch.tensor(temp4)
            test_b = torch.tensor(temp5)
            test_y = torch.tensor(temp6)
            # print("typeeeeeeeeeeeeeee>>>>", type(train_b), type(test_b), type(train_y), type(test_y))
            retrieved_indices = retrieve_topk(test_b, train_b, topK=100)
            prec = compute_precision_at_k(retrieved_indices, test_y, train_y, topK=100, is_single_label=single_label_flag)
            print("precision at 100: {:.4f}".format(prec.item()))

            if prec.item() > best_precision:
                best_precision = prec.item()
                best_precision_epoch = epoch + 1
        
#########################################################################################################
with open('logs/VDSH/result.txt', 'a') as handle:
    handle.write('{},{},{},{},{}\n'.format(dataset, data_fmt, args.nbits, best_precision_epoch, best_precision))