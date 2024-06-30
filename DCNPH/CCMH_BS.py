#-*- coding : utf-8-*-
# coding:unicode_escape
import pickle
import os
import argparse
import logging
import paddle
import time
import numpy as np
import paddle.optimizer as optim
import paddle.vision.transforms as transforms

from datetime import datetime
from paddle.io import DataLoader
from paddle.static import InputSpec

import utils.data_processing_o as dp
# import utils.data_processing_clip as dp
import utils.hash_model as image_hash_model
import utils.label_hash_model as label_hash_model
from utils.resnet import resnet34
import utils.txt_hash_model as txt_hash_model
import utils.calc_hr as calc_hr
import paddle.nn as nn

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
    ind_np = np.asarray(ind_list, dtype=np.int_)
    ind_np = ind_np - 1
    ind_label = label[ind_np, :]
    return paddle.to_tensor(ind_label)


def calc_sim(database_label, train_label):
    S = (train_label.mm(database_label.t()) > 0).type(paddle.FloatTensor)
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

def GenerateCode(model_hash, data_loader, num_data, bit, k=0):
    B = np.zeros((num_data, bit), dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_img, data_txt, data_label, data_ind = data
        data_img = data_img.cuda()
        # data_label = paddle.squeeze(data_label)
        data_label = paddle.cast(data_label, dtype='float32')
        data_txt = paddle.cast(data_txt, dtype='float32')
        if k == 0:
            out = model_hash(data_img)
            B[data_ind.numpy(), :] = paddle.sign(out.data.cpu()).numpy()
        if k == 1:
            out = model_hash(data_txt)
            B[data_ind.numpy(), :] = paddle.sign(out.data.cpu()).numpy()
        if k == 2:
            out = model_hash(data_label)
            B[data_ind.numpy(), :] = paddle.sign(out.data.cpu()).numpy()
    return B

def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def Logtrick(x):
    lt = paddle.log(1+paddle.exp(-paddle.abs(x))) + paddle.max(x, paddle.FloatTensor([0.]).cuda())
    return lt

def cossim(x, y):
    normal_x = x / paddle.sqrt((x.pow(2)).sum(1)).view(-1, 1).expand(x.size()[0], x.size()[1])
    normal_y = y / paddle.sqrt((y.pow(2)).sum(0)).view(1, -1).expand(y.size()[0], y.size()[1])
    cos = normal_x.mm(normal_y)
    return cos
def adsh_algo(code_length, tt):

    d = 3
    if d == 0:
        data_set = 'nus_random_alex_1'
        DATA_DIR = '/data/home/trc/nus_wide_txt/nus_wide_21_txt'
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
        data_name = 'nus_random_continuous'
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
        DATA_DIR = '/data/home/trc/MIRFlickr-25K'
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
        DATA_DIR = '/home/zgq/COCO'
        LABEL_FILE = 'label_hot.txt'
        IMAGE_FILE = 'images_name.txt'
        # TXT_FILE = 'y_vector.txt'
        TXT_FILE = 'YvectorLast1117.txt'
        # NN_FILE = 'YvectorDiff1117.txt'
        #TXT_FILE = 'YvectorDiff1117.txt'
        DATABASE_FILE = 'database_ind_ph.txt'
        TRAIN_FILE = 'train_ind_ph.txt'
        TEST_FILE = 'test_ind_ph.txt'
        data_name = './label_code/coco1'
        y_dim = 2000
        f_dim = 986





    batch_size = 40
    # code_length = 32
    epochs = 150
    learning_rate = 0.0015 #0.05cd 
    #nus
    # lri = {16: 0.0015, 32: 0.0012, 48: 0.001, 64: 0.001}
    # learning_ratei = lri[code_length]
    # lrt = {16: 0.008, 32: 0.008, 48: 0.006, 64: 0.005}
    # learning_ratet = lrt[code_length]

    # nus
    # lri = {16: 0.0015, 32: 0.0012, 48: 0.001, 64: 0.001, 96: 0.001, 128: 0.001}
    # learning_ratei = lri[code_length]
    # lrt = {16: 0.008, 32: 0.008, 48: 0.006, 64: 0.005, 96: 0.005, 128: 0.005}
    # learning_ratet = lrt[code_length]

    #coco
    lri = {16: 0.0008, 32: 0.0006, 48: 0.0005, 64: 0.0003, 96: 0.002}
    learning_ratei = lri[code_length]
    lrt = {16: 0.002, 32: 0.0017, 48: 0.001, 64: 0.001, 96: 0.001}
    learning_ratet = lrt[code_length]
    learning_ratet = 10 ** (-1.5)
    weight_decay = 10 ** -5
    model_name = 'alexnet'

    alpha = 0.05
    beta = 1
    lamda = 0.1 #50
    gamma = 0.3
    sigma = code_length * 0.3
    cro = 0.1

    print("*"*10, learning_rate, learning_ratei, learning_ratet, lamda, alpha, beta, code_length, sigma, gamma, cro, "*"*10)
    ### data processing
    print("WELL DOWN!1")
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print("WELL DOWN!2", DATA_DIR)
    dset_database = dp.DatasetProcessingNUS_WIDE(
        DATA_DIR, IMAGE_FILE, TXT_FILE, LABEL_FILE, DATABASE_FILE, transformations)

    dset_train = dp.DatasetProcessingNUS_WIDE(
        DATA_DIR, IMAGE_FILE, TXT_FILE, LABEL_FILE, TRAIN_FILE, transformations)

    dset_test = dp.DatasetProcessingNUS_WIDE(
        DATA_DIR, IMAGE_FILE, TXT_FILE, LABEL_FILE, TEST_FILE, transformations)
    # label_set = dp.DatasetProcessingNUS_WIDE_label(DATA_DIR, 'label_category.txt')
    # label_set_train = dp.DatasetProcessingNUS_WIDE_label(DATA_DIR, Category)

    num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)
    print("*"*10,"num_database:",num_database,"num_train:",num_train,"num_test:",num_test,"*"*10)
    database_loader = DataLoader(dset_database,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=4
                             )

    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0
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
    # label_categories = paddle.from_numpy(label_categories).type(paddle.FloatTensor)
    # hash_model = image_hash_model.HASH_Net(model_name, code_length)
    hash_model = image_hash_model.HASH_Net(model_name, code_length)
    txt_model = txt_hash_model.TxtNet(y_dim, code_length)
    label_model = label_hash_model.Label_net(test_labels.shape[1], code_length)
    nclass = test_labels.shape[1]



    # optimizer_label = optim.SGD(label_model.parameters(), lr=0.01, weight_decay=weight_decay)
    # scheduler_l = paddle.optim.lr_scheduler.StepLR(optimizer_label, step_size=150, gamma=0.5, last_epoch=-1)

    optimizer_hash = optim.SGD(parameters=hash_model.parameters(), learning_rate=learning_ratei, weight_decay=weight_decay)
    # scheduler = paddle.optim.lr_scheduler.StepLR(optimizer_hash, step_size=200, gamma=0.5, last_epoch=-1)
    scheduler = paddle.optimizer.lr.StepDecay(learning_rate=optimizer_hash.get_lr(), step_size=200, gamma=0.5, last_epoch=-1)
    optimizer_hash.set_lr_scheduler(scheduler)
    # scheduler = paddle.optim.lr_scheduler.LambdaLR(optimizer_hash, lr_lambda=lambda epoch: (121 - epoch) / (120 + 1))

    optimizer_txt = optim.SGD(parameters=txt_model.parameters(), learning_rate=learning_ratet, weight_decay=weight_decay)
    # scheduler_t = paddle.optim.lr_scheduler.StepLR(optimizer_txt, step_size=200, gamma=0.5, last_epoch=-1)
    # scheduler_t = paddle.optim.lr_scheduler.LambdaLR(optimizer_txt, lr_lambda=lambda epoch: (121 - epoch) / (120 + 1))
    scheduler_t = paddle.optimizer.lr.LambdaDecay(learning_rate=optimizer_txt.get_lr(), lr_lambda=lambda epoch: (121 - epoch) / (120 + 1))
    optimizer_txt.set_lr_scheduler(scheduler)

    '''
    model construction
    '''


    ### training phase
    # parameters setting
    # U_l = paddle.sign(paddle.randn(num_label, bit)).type(paddle.FloatTensor)
    # time_map = {'used_time': [0.0], 'map': [{'i2t': 0., 't2i': 0.}]}
    # used_time = 0.0
    # a = 1
    # with open(category, 'rb') as f:
    #     class_weight = pickle.load(f)
    # # print(class_weight.keys())
    # a = list(class_weight.keys())
    # print(a[0])
    print("WELL DOWN!3")
    iii = 1
    with open(data_name + '_label_code_' + str(code_length) + 'beta_' + str(beta) + '_1.pkl', 'rb') as f:
        label_code = pickle.load(f)
    B_xy = paddle.sign(paddle.randn([num_train, code_length], dtype='float32'))
    B_x = paddle.randn([num_train, code_length], dtype='float32')
    B_y = paddle.randn([num_train, code_length], dtype='float32')
    print("111111")
    for it in range(6):


        for z in range(2):
            epochi = 1
            epocht = 1
            # if iii:
            #     epocht = 1
            # else:
            #     epocht = 2
            #     iii = 0
            for epoch in range(epochi):
                scheduler.step()
                epoch_loss = 0.0
                epoch_loss_t = 0.0
                ## training epoch
                # print("train_loader:",len(train_loader))
                for iter, traindata in enumerate(train_loader, 0):
                    train_img, train_txt, train_label, batch_ind = traindata
                    # train_label = paddle.squeeze(train_label)
                    # train_txt = paddle.squeeze(train_txt)
                    # print(">>>>>3333")
                    train_img = train_img.cuda()
                    #print("******#train_img:",train_img.shape)#train_img: paddle.Size([40, 3, 224, 224])batchsize
                    
                    # print("22222")
                    train_label = train_label.astype('float32')
                    train_label = paddle.to_tensor(train_label, place=paddle.CUDAPlace(0))
                    the_batch = len(batch_ind)

                    hash_out = hash_model(train_img)
                    hash_out_cpu = hash_out.data.cpu().detach()

                    # 将 hash_out_cpu 的数据赋值给 B_x 的指定索引处
                    B_x[batch_ind] = hash_out_cpu.numpy()
                    # B_x[batch_ind] = hash_out.data.cpu()
                    textb = paddle.sign(B_y[batch_ind]).astype('float32')

                    textb = paddle.to_tensor(textb, place=paddle.CUDAPlace(0))
                    label_code_np = label_code.cpu().detach().numpy()
                    label_code = paddle.to_tensor(label_code_np)
                    logitt = paddle.matmul(textb, paddle.t(label_code))
                    logit = paddle.matmul(hash_out, paddle.t(label_code))
                    # print("logit:",logitt)

                    nits = paddle.sum(train_label, axis=1)

                    # assert (nits == 0).sum()
                    #print("nits:",nits)
                    # print("train_label:",train_label)
                    our_logitt = paddle.exp(((logitt * train_label).sum(1) / nits - sigma) * gamma)
                    #print("our_logitt:",our_logitt)
                    mu_logitt = paddle.exp(logitt * (1 - train_label) * gamma).sum(1) + our_logitt
                    #print("mu_logitt:",mu_logitt)
                    our_logit = paddle.exp(((logit * train_label).sum(1) / nits - sigma) * gamma)
                    #print("our_logit2:",our_logit)
                    mu_logit = paddle.exp(logit * (1 - train_label) * gamma).sum(1) + our_logit
                    #print("mu_logit2:",mu_logit)
                    loss = - (paddle.log(our_logit / mu_logit)).sum() - cro * (paddle.log(our_logit / mu_logitt)).sum()
                    #print('[Loss: %3.5f]' % (loss))

                    Bbatch = B_xy[batch_ind].cuda()
                    regterm = paddle.sum((Bbatch - hash_out).pow(2)) / the_batch
                    lossa = loss / the_batch
                    loss_all = lossa + regterm * lamda
                    optimizer_hash.clear_grad()
                    loss_all.backward(retain_graph=True)
                    optimizer_hash.step()
                    
                    epoch_loss += lossa.numpy()
                    epoch_loss_t += regterm.numpy()

                #print("train_loader:",train_loader)
                #print("len(train_loader):",len(train_loader))
                # print('[Train Phase][Epoch: %3d/%3d][Loss_i: %3.5f][Loss_r: %3.5f]' % (z + 1, epochs, epoch_loss / len(train_loader), epoch_loss_t / len(train_loader)))

            for epoch in range(epocht):
                print(">>>>>>>>>>>>", epoch, epocht)
                scheduler_t.step()
                epoch_loss = 0.0
                epoch_loss_t = 0.0
                ## training epoch
                for iter, traindata in enumerate(train_loader, 0):
                    train_img, train_txt, train_label, batch_ind = traindata
                    # train_label = paddle.squeeze(train_label)
                    # train_txt = paddle.squeeze(train_txt)
                    train_txt = train_txt.astype('float32')
                    train_txt = paddle.to_tensor(train_txt, place=paddle.CUDAPlace(0))
                    train_label = train_label.astype('float32')
                    train_label = paddle.to_tensor(train_label, place=paddle.CUDAPlace(0))
                    the_batch = len(batch_ind)

                    hash_out = txt_model(train_txt)
                    hash_out_cpu = hash_out.data.cpu().detach()

                    # 将 hash_out_cpu 的数据赋值给 B_x 的指定索引处
                    B_y[batch_ind] = hash_out_cpu.numpy()
                    # B_y[batch_ind] = hash_out.data.cpu()
                    # imgb = paddle.sign(B_x[batch_ind]).type(paddle.FloatTensor).cuda()
                    imgb = paddle.sign(B_y[batch_ind]).astype('float32')

                    imgb = paddle.to_tensor(imgb, place=paddle.CUDAPlace(0))
                    logiti = paddle.matmul(imgb, paddle.t(label_code))
                    logit = paddle.matmul(hash_out, paddle.t(label_code))

                    nits = paddle.sum(train_label, axis=1)

                    our_logiti = paddle.exp(((logiti * train_label).sum(1) / nits - sigma) * gamma)
                    mu_logiti = paddle.exp(logiti * (1 - train_label) * gamma).sum(1) + our_logiti
                    our_logit = paddle.exp(((logit * train_label).sum(1) / nits - sigma) * gamma)
                    mu_logit = paddle.exp(logit * (1 - train_label) * gamma).sum(1) + our_logit
                    loss = - (paddle.log(our_logit / mu_logit)).sum() - cro * (paddle.log(our_logit / mu_logitt)).sum()

                    Bbatch = B_xy[batch_ind]
                    regterm = paddle.sum((Bbatch - hash_out).pow(2)) / the_batch
                    lossa = loss / the_batch
                    loss_allt = lossa + regterm * lamda

                    optimizer_txt.clear_grad()
                    loss_allt.backward(retain_graph=True)
                    optimizer_txt.step()

                    epoch_loss_t += lossa.item()
                    epoch_loss += regterm.item()

                # print('[Train Phase][Epoch: %3d/%3d][Loss_t: %3.5f][Loss_r: %3.5f]' % ((z + 1) * (it + 1) * (epoch + 1) + 1, epochs, epoch_loss_t / len(train_loader), epoch_loss / len(train_loader)))
            B_xy = paddle.sign(B_x + B_y)
        print("loop over!!")
        hash_model.eval()
        txt_model.eval()
        print("111111")
        qt = GenerateCode(txt_model, test_loader, num_test, code_length, 1)
        qi = GenerateCode(hash_model, test_loader, num_test, code_length)
        ri = GenerateCode(hash_model, database_loader, num_database, code_length)
        rt = GenerateCode(txt_model, database_loader, num_database, code_length, 1)
        print("222222")
        hash_model.train()
        txt_model.train()
        print("3333333")
        # map_ti = calc_hr.calc_topMap(qt, ri, test_labels.numpy(), database_labels.numpy(), 5000)
        # print('txt_i_map5000:', map_ti)
        # map_it = calc_hr.calc_topMap(qi, rt, test_labels.numpy(), database_labels.numpy(), 5000)
        # print('i_t_map5000:', map_it)
        if isinstance(test_labels, paddle.static.InputSpec):
            test_labels = paddle.to_tensor(test_labels)
        if isinstance(database_labels, paddle.static.InputSpec):
            database_labels = paddle.to_tensor(database_labels)
        map_ti = calc_hr.calc_map(qt, ri, test_labels.numpy(), database_labels.numpy())
        print("*"*10,"--Baseline--","*"*10)
        print('txt_i_map:', map_ti)
        map_it = calc_hr.calc_map(qi, rt, test_labels.numpy(), database_labels.numpy())
        print('i_t_map:', map_it)
    '''
    training procedure finishes, evaluation
    '''
def main():
    # paddle.backends.cudnn.enabled = False
    # paddle.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=float)#add_argument ǰ����������֮ǰ���ϡ�- -�������ܽ�֮��Ϊ��ѡ������
    opt = parser.parse_args()# ��parser�����õ�����"add_argument"�����ص�args����ʵ������
    print(opt.param)
    if (float(opt.param) < 0.000001) | (float(opt.param) > 0.9):
        param = int(opt.param)
    else:
        param = int(opt.param)
    adsh_algo(96, param)
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

if __name__=="__main__":
    main()