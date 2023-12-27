import paddle
import paddle.optimizer as optim
from model import *
from train_model import train_model
from load_data import get_loader
from evaluate import fx_calc_map_label

######################################################################
# Start running

import time
import os

if __name__ == '__main__':

    dataset = 'pascnn'

    DATA_DIR = 'data/' + dataset + '/'
    MAX_EPOCH = 80
    MAX_EPOCHGAN = 100
    batch_size = 100
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.999
    weight_decay = 0.01

    print('...Data loading is beginning...')

    data_loader, input_data_par = get_loader(DATA_DIR, batch_size)

    print('...Data loading is completed...')

    model_ft = CrossGCN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],
                        output_dim=input_data_par['num_class'])

    dis_ft = DiscriminatorV()
    params_to_update = list(model_ft.parameters())
    params_dis = list(dis_ft.parameters())

    gen_ft = GeneratorV()
    params_gen = list(gen_ft.parameters())

    #########
    disT_ft = DiscriminatorT()
    params_disT = list(disT_ft.parameters())

    genT_ft = GeneratorT()
    params_genT = list(genT_ft.parameters())


    # Observe that all parameters are being optimized
    optimizer = optim.Adam(learning_rate=lr, parameters=params_to_update, beta1=beta1, beta2=beta2)
    optimizer_dis = optim.Adam(learning_rate=lr, parameters=params_dis, beta1=beta1, beta2=beta2)
    optimizer_gen = optim.Adam(learning_rate=lr, parameters=params_gen, beta1=beta1, beta2=beta2, weight_decay=weight_decay)
    optimizer_disT = optim.Adam(learning_rate=lr, parameters=params_disT, beta1=beta1, beta2=beta2)
    optimizer_genT = optim.Adam(learning_rate=lr, parameters=params_genT, beta1=beta1, beta2=beta2, weight_decay=weight_decay)

    print('...Training is beginning...')
    # Train and evaluate
    start = time.time()
    genT_ft, gen_ft, model_ft = train_model(genT_ft, disT_ft, gen_ft, dis_ft,model_ft,
                                            data_loader, optimizer_genT, optimizer_disT, optimizer_gen, optimizer_dis, optimizer, num_epochs=MAX_EPOCH,num_epochsGAN=MAX_EPOCHGAN)
    end = time.time()
    print('Total train time:', end-start)
    print('...Training is completed...')
    checkpoint='checkpoint.model'
    paddle.save(model_ft.state_dict(), checkpoint)





