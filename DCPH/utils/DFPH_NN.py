#-*- coding : utf-8-*-
# coding:unicode_escape

import pickle
import os
import argparse
import logging
import torch
import time
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms

from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
import random

def setRandomSeed(seed=0):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    setRandomSeed(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True

setRandomSeed()
import utils.data_processing as dp  #相比data_processing_o多了feature特征
import utils.hash_model as image_hash_model
import utils.label_hash_model as label_hash_model
from utils.resnet import resnet34
import utils.txt_hash_model as txt_hash_model
import utils.calc_hr as calc_hr
import torch.nn as nn
#gpu_id = [1, 2, 3, 4, 5, 6]
#torch.cuda.set_device(gpu_id)


def _logging():
    os.mkdir(logdir)
    global logger
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return

def _record():
    global record
    record = {}
    record['train loss'] = []
    record['iter time'] = []
    record['param'] = {}
    return

def _save_record(record, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(record, fp)
    return

def load_label(label_filename, ind, DATA_DIR):
    label_filepath = os.path.join(DATA_DIR, label_filename)
    label = np.loadtxt(label_filepath, dtype=np.int64)
    ind_filepath = os.path.join(DATA_DIR, ind)
    fp = open(ind_filepath, 'r')
    ind_list = [x.strip() for x in fp]
    fp.close()
    ind_np = np.asarray(ind_list, dtype=np.int32)
    ind_np = ind_np - 1
    ind_label = label[ind_np, :]
    return torch.from_numpy(ind_label)


def calc_sim(database_label, train_label):
    S = (train_label.mm(database_label.t()) > 0).type(torch.FloatTensor)
    '''
    soft constraint
    '''
    #r = S.sum() / (1-S).sum()
    #S = S*(1+r) - r
    return S

def calc_loss(V, U, G, S, code_length, select_index, gamma, alpha=1):
    num_database = V.shape[0]
    square_loss = (U.dot(V.transpose()) - code_length*S) ** 2
    V_omega = V[select_index, :]
    quantization_loss = (0.5 * (U + G) - V_omega) ** 2
    loss = (alpha * square_loss.sum() + gamma * quantization_loss.sum()) / (opt.num_samples * num_database)
    return loss

def GenerateCode(model_hash, model_hash_f,data_loader, num_data, bit, k=0):
    B = np.zeros((num_data, bit), dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_img, data_txt, data_feature, data_label, data_ind = data
        #data_img, data_txt, data_label, data_ind = data
        data_img = Variable(data_img.cuda())
        # data_label = torch.squeeze(data_label)
        data_label = Variable(data_label.type(torch.FloatTensor).cuda())
        data_txt = Variable(data_txt.type(torch.FloatTensor).cuda())
        data_feature = Variable(data_feature.type(torch.FloatTensor).cuda())
        if k == 0:
            out = model_hash(data_img)
            B[data_ind.numpy(), :] = torch.sign(out.data.cpu()).numpy()
        if k == 1:
            out_t= model_hash(data_txt)
            out_f = model_hash_f(data_feature)
            out=out_t+out_f
            B[data_ind.numpy(), :] = torch.sign(out.data.cpu()).numpy()
        if k == 2:
            out = model_hash(data_label)
            B[data_ind.numpy(), :] = torch.sign(out.data.cpu()).numpy()
        if k == 3:
            out= model_hash(data_feature)
            #out_f = model_hash_t(data_feature)
            #out=out_t+out_f
            B[data_ind.numpy(), :] = torch.sign(out.data.cpu()).numpy()
    return B

def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def Logtrick(x):
    lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.]).cuda()))
    return lt

def cossim(x, y):
    normal_x = x / torch.sqrt((x.pow(2)).sum(1)).view(-1, 1).expand(x.size()[0], x.size()[1])
    normal_y = y / torch.sqrt((y.pow(2)).sum(0)).view(1, -1).expand(y.size()[0], y.size()[1])
    cos = normal_x.mm(normal_y)
    return cos


def adsh_algo(code_length, param, dataset):
# def adsh_algo(code_length, tt):
    d = dataset#3
    if d == 0:
        data_set = 'nus_random_alex_1'
        DATA_DIR = '../nus_wide_21_txt'
        # DATA_DIR = '/home2/public/trc/NUS-WIDE-TC21/nus_wide_21_txt'
        # DATA_DIR = '/data/trc/nus_wide_21_txt'
        LABEL_FILE = 'label_hot.txt'
        IMAGE_FILE = 'images_name.txt'
        TXT_FILE = 'y_vector.txt'
        DATABASE_FILE = 'database_ind_ph.txt'
        TRAIN_FILE = 'train_ind_ph.txt'
        TEST_FILE = 'test_ind_ph.txt'
        # DATABASE_FILE = 'database_ind_original.txt'
        # TRAIN_FILE = 'train_ind_original.txt'
        # TEST_FILE = 'test_ind_original.txt'
        category = 'class_weight_nus.pkl'
        data_name = './label_code/nus'
        y_dim = 1000
    elif d == 1:
        data_set = 'iaprtc'
        DATA_DIR = '/data/home/trc/IAPRTC12'
        LABEL_FILE = 'label_hot.txt'
        IMAGE_FILE = 'images_name.txt'
        TXT_FILE = 'y_vector.txt'
        DATABASE_FILE = 'database_ind.txt'
        TRAIN_FILE = 'train_ind.txt'
        TEST_FILE = 'test_ind.txt'
        y_dim = 2000
        category = 'class_weight_iaprtc.pkl'
        data_name = 'iaprtc'
    elif d == 2:
        data_set = 'mir'
        DATA_DIR = '/s2_md0/leiji/v-rtu/2dtan/MIRFlickr-25K'
        LABEL_FILE = 'label_hot.txt'
        IMAGE_FILE = 'images_name.txt'
        TXT_FILE = 'y_vector.txt'
        DATABASE_FILE = 'database_ind_mh.txt'
        TRAIN_FILE = 'train_ind_mh.txt'
        TEST_FILE = 'test_ind_mh.txt'
        y_dim = 1386
        data_name = './label_code/mir'
        category = 'class_weight_mir.pkl'
    else:
        category = 'class_weight_coco.pkl'
        data_set = 'coco1_1'
        #DATA_DIR = '/mnt/my_storage/data/COCO'
        #DATA_DIR = '/s2_md0/leiji/v-rtu/2dtan/COCO'
        # DATA_DIR = '/data/trc/COCO'
        # DATA_DIR = '/home2/public/trc/COCO'
        DATA_DIR = '/data1/home/jwj/dataset/COCO'
        LABEL_FILE = 'label_hot.txt'
        IMAGE_FILE = 'images_name.txt'
        # TXT_FILE = 'y_vector.txt'
        TXT_FILE = 'YvectorLast1117.txt'
        NN_FILE = 'YvectorDiff1117.txt'
        #TXT_FILE = 'YvectorDiff1117.txt'
        DATABASE_FILE = 'database_ind_ph.txt'
        TRAIN_FILE = 'train_ind_ph.txt'
        TEST_FILE = 'test_ind_ph.txt'
        data_name = './label_code/coco1'
        y_dim = 2000
        f_dim = 986





    batch_size = 40
    # code_length = 32
    # epochs = 150
    # learning_rate = 0.001 #0.05cd 
    #nus
    if d==0:
        lri = {16: 0.0015, 32: 0.0012, 48: 0.001, 64: 0.001}
        learning_ratei = lri[code_length]
        lrt = {16: 0.008, 32: 0.008, 48: 0.006, 64: 0.005}
        learning_ratet = lrt[code_length]


    #coco
    elif d==3:
        lri = {16: 0.0008, 32: 0.0006, 48: 0.0005, 64: 0.0003}
        learning_ratei = lri[code_length]
        lrt = {16: 0.002, 32: 0.0017, 48: 0.001, 64: 0.001}
        learning_ratet = lrt[code_length]





    # learning_ratet = 0.0316

    learning_ratef = 0.01

    weight_decay = 10 ** -5
    model_name = 'alexnet'

    alpha = 0.05
    beta = 1
    lamda = 0.1 #50
    gamma = 0.3
    sigma = code_length * 0.3
    cro = 0.1

    print("*"*10,  learning_ratei, learning_ratet, lamda, alpha, beta, code_length, sigma, gamma, cro, "*"*10)
    ### data processing

    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #dset_database = dp.DatasetProcessingNUS_WIDE(DATA_DIR, IMAGE_FILE, TXT_FILE, LABEL_FILE, DATABASE_FILE, transformations)

    #dset_train = dp.DatasetProcessingNUS_WIDE(DATA_DIR, IMAGE_FILE, TXT_FILE, LABEL_FILE, TRAIN_FILE, transformations)

    #dset_test = dp.DatasetProcessingNUS_WIDE(DATA_DIR, IMAGE_FILE, TXT_FILE, LABEL_FILE, TEST_FILE, transformations)
    
    
    dset_database = dp.DatasetProcessingNUS_WIDE(DATA_DIR, IMAGE_FILE, TXT_FILE, NN_FILE, LABEL_FILE, DATABASE_FILE, transformations)

    dset_train = dp.DatasetProcessingNUS_WIDE(DATA_DIR, IMAGE_FILE, TXT_FILE, NN_FILE, LABEL_FILE, TRAIN_FILE, transformations)

    dset_test = dp.DatasetProcessingNUS_WIDE(DATA_DIR, IMAGE_FILE, TXT_FILE, NN_FILE, LABEL_FILE, TEST_FILE, transformations)
    
    
    # label_set = dp.DatasetProcessingNUS_WIDE_label(DATA_DIR, 'label_category.txt')
    # label_set_train = dp.DatasetProcessingNUS_WIDE_label(DATA_DIR, Category)

    num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)
    # print("*"*10,"num_database:",num_database,"num_train:",num_train,"num_test:",num_test,"*"*10)
    database_loader = DataLoader(dset_database,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4
                             )

    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )
    # train_loader1 = DataLoader(label_set,
    #                           batch_size=128,
    #                           shuffle=True,
    #                           num_workers=4
    #                           )
    # train_loader2 = DataLoader(label_set_train,
    #                           batch_size=batch_size,
    #                           shuffle=True,
    #                           num_workers=4
    #                           )
    test_loader = DataLoader(dset_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4
                             )
    train_labels = load_label(LABEL_FILE, TRAIN_FILE, DATA_DIR)
    database_labels = load_label(LABEL_FILE, DATABASE_FILE, DATA_DIR)
    test_labels = load_label(LABEL_FILE, TEST_FILE, DATA_DIR)
    # label_categories = np.loadtxt(DATA_DIR + '/' + Category)
    # label_categories = torch.from_numpy(label_categories).type(torch.FloatTensor)

    hash_model = image_hash_model.HASH_Net(model_name, code_length)
    # hash_model = resnet34(code_length)
    # hash_model = nn.parallel.DistributedDataParallel(hash_model,device_ids=[1,2,3,4])
    # print("Done")
    hash_model.cuda()
    txt_model = txt_hash_model.TxtNet(y_dim, code_length)
    txt_model.cuda()
    txt_model_f = txt_hash_model.TxtNet(f_dim, code_length)
    txt_model_f.cuda()
    label_model = label_hash_model.Label_net(test_labels.shape[1], code_length)
    label_model.cuda()
    nclass = test_labels.shape[1]



    # optimizer_label = optim.SGD(label_model.parameters(), lr=0.01, weight_decay=weight_decay)
    # scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_label, step_size=150, gamma=0.5, last_epoch=-1)

    optimizer_hash = optim.SGD(hash_model.parameters(), lr=learning_ratei, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_hash, step_size=50, gamma=0.5, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_hash, lr_lambda=lambda epoch: (121 - epoch) / (120 + 1))

    optimizer_txt = optim.SGD(txt_model.parameters(), lr=learning_ratet, weight_decay=weight_decay)
    # scheduler_t = torch.optim.lr_scheduler.StepLR(optimizer_txt, step_size=200, gamma=0.5, last_epoch=-1)
    # scheduler_t = torch.optim.lr_scheduler.LambdaLR(optimizer_txt, lr_lambda=lambda epoch: (81 - epoch) / (80 + 1))
    scheduler_t = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_txt,
                                                                 T_0=20, 
                                                                T_mult=1)

    optimizer_feature = optim.SGD(txt_model_f.parameters(), lr=learning_ratef, weight_decay=weight_decay)
    scheduler_f = torch.optim.lr_scheduler.LambdaLR(optimizer_feature, lr_lambda=lambda epoch: (81 - epoch) / (80 + 1))
    '''
    model construction
    '''


    ### training phase
    # parameters setting
    # U_l = torch.sign(torch.randn(num_label, bit)).type(torch.FloatTensor)
    # time_map = {'used_time': [0.0], 'map': [{'i2t': 0., 't2i': 0.}]}
    # used_time = 0.0
    # a = 1
    # with open(category, 'rb') as f:
    #     class_weight = pickle.load(f)
    # # print(class_weight.keys())
    # a = list(class_weight.keys())
    # print(a[0])
    iii = 1
    with open(data_name + '_label_code_' + str(code_length) + 'beta_' + str(beta) + '_1.pkl', 'rb') as f:
        label_code = pickle.load(f)
    B_xyz = torch.sign(torch.randn(num_train, code_length).type(torch.FloatTensor))#��size��2����ʱ������һ��num_train��code_length�е���������СԪ�ص�ÿһ�з�����̬�ֲ�##���ط���
    B_x = torch.randn(num_train, code_length).type(torch.FloatTensor)
    B_y = torch.randn(num_train, code_length).type(torch.FloatTensor)
    B_z = torch.randn(num_train, code_length).type(torch.FloatTensor)

    best_score = 0
    best_iter = 0
    best_ti_score = 0
    best_ti_iter = 0
    best_it_score = 0
    best_it_iter = 0
    for it in range(4):
        for z in range(20):
            epochi = 1
            epocht = 1
            # if iii:
            #     epocht = 1
            # else:
            #     epocht = 2
            #     iii = 0
            for epoch in range(epochi):
                
                epoch_loss = 0.0
                epoch_loss_t = 0.0
                epoch_loss_t_xz=0.0
                epoch_loss_t_yz=0.0
                epoch_loss_xz=0.0
                epoch_loss_yz=0.0
                ## training epoch
                # print("train_loader:",len(train_loader))
                for iter, traindata in enumerate(train_loader, 0):
                    #train_img, train_txt, train_label, batch_ind = traindata
                    train_img, train_txt, train_NN, train_label, batch_ind = traindata
                    
                    # train_label = torch.squeeze(train_label)
                    # train_txt = torch.squeeze(train_txt)
                    train_img = Variable(train_img.cuda())
                    train_label = Variable(train_label.type(torch.FloatTensor).cuda())
                    the_batch = len(batch_ind)

                    hash_out = hash_model(train_img)

                    B_x[batch_ind] = hash_out.data.cpu()
                    textb = Variable(torch.sign(B_y[batch_ind]).type(torch.FloatTensor).cuda())
                    
                    
                    
                    logitt = textb.mm(label_code.t())
                    logit = hash_out.mm(label_code.t())
                    
                    
                    
                    nits = train_label.sum(1)
                    # print("nits:",nits)
                    our_logitt = torch.exp(((logitt * train_label).sum(1) / nits - sigma) * gamma)
                    
                    
                    
                    # print("our_logitt:",our_logitt)
                    mu_logitt = torch.exp(logitt * (1 - train_label) * gamma).sum(1) + our_logitt
                    
                   
                    
                    # print("mu_logitt:",mu_logitt)
                    our_logit = torch.exp(((logit * train_label).sum(1) / nits - sigma) * gamma)
                    
                    
                    
                    # print("our_logit2:",our_logit)
                    mu_logit = torch.exp(logit * (1 - train_label) * gamma).sum(1) + our_logit
                    
                    
                    
                    # print("mu_logit2:",mu_logit)
                    loss = - (torch.log(our_logit / mu_logit)).sum() - cro * (torch.log(our_logit / mu_logitt)).sum()
                    
                    
                    
                    
                    # print('[Loss: %3.5f]' % (loss))

                    Bbatch = Variable(B_xyz[batch_ind].cuda())
                    regterm = (Bbatch - hash_out).pow(2).sum() / the_batch
                    lossa = loss / the_batch
                    loss_all = lossa + regterm * lamda
                    optimizer_hash.zero_grad()
                    # optimizer_label.zero_grad()
                    loss_all.backward(retain_graph=True)
                    optimizer_hash.step()
                    epoch_loss += lossa.item()
                    epoch_loss_t += regterm.item()
                scheduler.step()
                # print("train_loader:",train_loader)
                # print("len(train_loader):",len(train_loader))
                # print('[Train Phase][Epoch: %3d/%3d][Loss_i: %3.5f][Loss_r: %3.5f]' % (z + 1, epochs, epoch_loss / len(train_loader), epoch_loss_t / len(train_loader)))

            for epoch in range(epocht):
                
                epoch_loss = 0.0
                epoch_loss_t = 0.0
                epoch_loss_t_xz=0.0
                epoch_loss_t_yz=0.0
                epoch_loss_xz=0.0
                epoch_loss_yz=0.0
                ## training epoch
                for iter, traindata in enumerate(train_loader, 0):
                    #train_img, train_txt, train_label, batch_ind = traindata
                    train_img, train_txt, train_NN, train_label, batch_ind = traindata
                    train_txt = Variable(train_txt.type(torch.FloatTensor).cuda())
                    train_NN = Variable(train_NN.type(torch.FloatTensor).cuda())
                    train_label = Variable(train_label.type(torch.FloatTensor).cuda())
                    the_batch = len(batch_ind)
                    # train_New= torch.cat((train_txt,train_NN), 2)
                    # print("********************train_txt:",train_txt.size())
                    # print("********************train_NN:",train_NN.size())
                    # print("********************train_New:",train_New.size())
                    # hash_out = txt_model(train_txt,train_NN)
                    hash_out_t = txt_model(train_txt)
                    hash_out_f = txt_model_f(train_NN)
                    hash_out=hash_out_t+hash_out_f+hash_out_f
                    B_y[batch_ind] = hash_out.data.cpu()
                    imgb = Variable(torch.sign(B_x[batch_ind]).type(torch.FloatTensor).cuda())
                    logiti = imgb.mm(label_code.t())
                    logit = hash_out.mm(label_code.t())
                    nits = train_label.sum(1)
                    our_logiti = torch.exp(((logiti * train_label).sum(1) / nits - sigma) * gamma)
                    mu_logiti = torch.exp(logiti * (1 - train_label) * gamma).sum(1) + our_logiti
                    
                    our_logit = torch.exp(((logit * train_label).sum(1) / nits - sigma) * gamma)
                    
                    mu_logit = torch.exp(logit * (1 - train_label) * gamma).sum(1) + our_logit
                    
                    loss = - (torch.log(our_logit / mu_logit)).sum() - cro * (torch.log(our_logit / mu_logiti)).sum()

                    Bbatch = Variable(B_xyz[batch_ind].cuda())
                    regterm = (Bbatch - hash_out).pow(2).sum() / the_batch
                    #修改了一下归一化hash_out.sign()
                    lossa = loss / the_batch
                    # lossa_yz = loss_yz / the_batch
                    
                    loss_allt = lossa + regterm * lamda
                    # loss_allt_yz = lossa_yz
                    #print('[loss_allt: %3.5f]' % (loss_allt))
                    optimizer_txt.zero_grad()
                    optimizer_feature.zero_grad()
                    loss_allt.backward(retain_graph=True)
                    optimizer_txt.step()
                    optimizer_feature.step()
                    
                    epoch_loss_t += lossa.item()
                    epoch_loss += regterm.item()
                scheduler_t.step()
                #print('[Train Phase][Epoch: %3d/%3d][Loss_t: %3.5f][Loss_r: %3.5f]' % ((z + 1) * (it + 1) * (epoch + 1) + 1, epochs, epoch_loss_t / len(train_loader), epoch_loss / len(train_loader)))
            B_xyz = torch.sign(B_x+B_y+B_z)
        hash_model.eval()
        txt_model.eval()
        txt_model_f.eval()
        qt = GenerateCode(txt_model,  txt_model_f,test_loader, num_test, code_length, 1)
        qi = GenerateCode(hash_model, txt_model_f,test_loader, num_test, code_length)
        ri = GenerateCode(hash_model, txt_model_f,database_loader, num_database, code_length)
        rt = GenerateCode(txt_model,  txt_model_f,database_loader, num_database, code_length, 1)
        hash_model.train()
        txt_model.train()
        txt_model_f.train()
        # map_ti = calc_hr.calc_topMap(qt, ri, test_labels.numpy(), database_labels.numpy(), 5000)
        # print('txt_i_map5000:', map_ti)
        # map_it = calc_hr.calc_topMap(qi, rt, test_labels.numpy(), database_labels.numpy(), 5000)
        # print('i_t_map5000:', map_it)
        map_ti = calc_hr.calc_map(qt, ri, test_labels.numpy(), database_labels.numpy())
        print("*"*10,"--OUTPLUSNN--",learning_ratei, learning_ratet,"*"*10)
        print('txt_i_map:', map_ti)
        map_it = calc_hr.calc_map(qi, rt, test_labels.numpy(), database_labels.numpy())
        print('i_t_map:', map_it)
        score = (map_ti+map_it)/2
        if score>best_score:
            best_score = score 
            best_iter = it
        if map_ti>best_ti_score:
            best_ti_score = map_ti 
            best_ti_iter = it
        if map_it>best_it_score:
            best_it_score = map_it 
            best_it_iter = it


    print("Best score:%f, best iter: %d" % (best_score,best_iter))
    print("best txt_i_map:%f, best iter: %d" % (best_ti_score,best_ti_iter))
    print("best i_t_map:%f, best iter: %d" % (best_it_score,best_it_iter))



def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=float, default=0.01)#add_argument ǰ����������֮ǰ���ϡ�- -�������ܽ�֮��Ϊ��ѡ������
    # parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()# ��parser�����õ�����"add_argument"�����ص�args����ʵ������
    # print(opt.param)
    if (float(opt.param) < 0.000001) | (float(opt.param) > 0.9):
        param = int(opt.param)
    else:
        param = int(opt.param)
    #adsh_algo(32, param)
    # bits = [64]
    # # sigmas = [0.2, 0.25, 0.35, 0.4]
    # # sigmas = [0.45, 0.5, 0.55, 0.6]
    # # sigmas = [0.6, 0.7, 0.8, 0.9, 1.]
    # # sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
    # # sigmas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    # sigmas = [1]
    # for bit in bits:
    #     print('*' * 10, bit, '*' * 10)
    #     for sigam in sigmas:
    #         print('*' * 10, sigam, '*' * 10)
    #         adsh_algo(bit, sigam)
    
    #test lr 0.05、0.01、0.005，0.005、0.0001、0.00001
    # for index in range(1):
    #     #print('*' * 10, bit, '*' * 10)
    #     adsh_algo(index,32, param)
    dataset = 3 #0 for nus    3 for coco
    nbit = 64
    adsh_algo(nbit, param, dataset)

if __name__=="__main__":
    main()