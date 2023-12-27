import paddle

from datetime import datetime
import paddle.optimizer as optim
import matplotlib.pyplot as plt
from model import IDCM_NN
from train_model import train_model
from load_data import get_loader
from evaluate import fx_calc_map_label
from paddle.vision.transforms import functional as F
######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    dataset = 'pascal'
    # data parameters
    DATA_DIR = 'data/' + dataset + '/'
    alpha = 1e-3
    beta = 1e-1
    MAX_EPOCH = 500
    batch_size = 100
    # batch_size = 512
    lr = 1e-4
    beta1 = 0.5 
    beta2 = 0.999
    weight_decay = 0

    print('...Data loading is beginning...')

    data_loader, input_data_par = get_loader(DATA_DIR, batch_size)

    print('...Data loading is completed...')

    model_ft = IDCM_NN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'], output_dim=input_data_par['num_class'])
    params_to_update = list(model_ft.parameters())

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(learning_rate=lr, parameters=params_to_update,  beta1=beta1, beta2=beta2)

    print('...Training is beginning...')
    # Train and evaluate
    model_ft, img_acc_hist, txt_acc_hist, loss_hist = train_model(model_ft, data_loader, optimizer, alpha, beta, MAX_EPOCH)
    print('...Training is completed...')
