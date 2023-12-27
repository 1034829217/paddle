from __future__ import print_function
from __future__ import division
import paddle
import paddle.nn as nn
import paddle.vision
import time
import copy
from evaluate import fx_calc_map_label
import paddle.nn.functional as F
from evaluate import fx_calc_map_label
import numpy as np
print("Paddle Version: ", paddle.__version__)


def calc_label_sim(label_1, label_2):
    Sim = label_1.astype('float64').mm(label_2.astype('float64').t())
    return Sim

# def cos(x, y):
#     return x.mm(y.t())

def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1, labels_2, alpha, beta):
    term1 = ((view1_predict-labels_1.astype('float64'))**2).sum(1).sqrt().mean() + ((view2_predict-labels_2.astype('float64'))**2).sum(1).sqrt().mean()

    cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clip(min=1e-6) / 2.
    theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = calc_label_sim(labels_1, labels_1).astype('float64')
    Sim12 = calc_label_sim(labels_1, labels_2).astype('float64')
    Sim22 = calc_label_sim(labels_2, labels_2).astype('float64')
    term21 = ((1 + paddle.exp(theta11)).log() - Sim11 * theta11).mean()
    term22 = ((1 + paddle.exp(theta12)).log() - Sim12 * theta12).mean()
    term23 = ((1 + paddle.exp(theta22)).log() - Sim22 * theta22).mean()
    term2 = term21 + term22 + term23

    term3 = ((view1_feature - view2_feature)**2).sum(1).sqrt().mean()

    im_loss = term1 + alpha * term2 + beta * term3
    return im_loss



def train_model(model, data_loaders, optimizer, alpha, beta, device="cpu", num_epochs=500):
    since = time.time()
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history =[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()

            running_loss = 0.0
            running_corrects_img = 0
            running_corrects_txt = 0
            # Iterate over data.
            for imgs, txts, labels in data_loaders[phase]:
                # imgs = imgs.to(device)
                # txts = txts.to(device)
                # labels = labels.to(device)
                if paddle.sum(imgs != imgs)>1 or paddle.sum(txts != txts)>1:
                    print("Data contains Nan.")

                # zero the parameter gradients
                # optimizer.zero_grad()

                # forward
                # track history if only in train
                with paddle.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    imgs = imgs.cpu()
                    txts = txts.cpu()
                    labels = labels.cpu()

                    # zero the parameter gradients
                    # optimizer.zero_grad()

                    # Forward
                    view1_feature, view2_feature, view1_predict, view2_predict = model(imgs, txts)

                    loss = calc_loss(view1_feature, view2_feature, view1_predict,
                                     view2_predict, labels, labels, alpha, beta)

                    img_preds = view1_predict
                    txt_preds = view2_predict

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects_img += paddle.sum(paddle.argmax(img_preds, axis=1) == paddle.argmax(labels, axis=1))
                running_corrects_txt += paddle.sum(paddle.argmax(txt_preds, axis=1) == paddle.argmax(labels, axis=1))

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            # epoch_img_acc = running_corrects_img.double() / len(data_loaders[phase].dataset)
            # epoch_txt_acc = running_corrects_txt.double() / len(data_loaders[phase].dataset)
            t_imgs, t_txts, t_labels = [], [], []
            with paddle.no_grad():
                for imgs, txts, labels in data_loaders['test']:
                    imgs = imgs.cpu()
                    txts = txts.cpu()
                    labels = labels.cpu()
                    t_view1_feature, t_view2_feature, _, _ = model(imgs, txts)
                    t_imgs.append(t_view1_feature.cpu().numpy())
                    t_txts.append(t_view2_feature.cpu().numpy())
                    t_labels.append(labels.cpu().numpy())
            t_imgs = np.concatenate(t_imgs)
            t_txts = np.concatenate(t_txts)
            t_labels = np.concatenate(t_labels).argmax(1)
            img2text = fx_calc_map_label(t_imgs, t_txts, t_labels)
            txt2img = fx_calc_map_label(t_txts, t_imgs, t_labels)

            print('{} Loss: {:.4f} Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(phase, epoch_loss, img2text, txt2img))

            # deep copy the model
            if phase == 'test' and (img2text + txt2img) / 2. > best_acc:
                best_acc = (img2text + txt2img) / 2.
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                test_img_acc_history.append(img2text)
                test_txt_acc_history.append(txt2img)
                epoch_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))

    # load best model weights
    # print(type(model))
    # model.load(best_model_wts)
    return model, test_img_acc_history, test_txt_acc_history, epoch_loss_history
