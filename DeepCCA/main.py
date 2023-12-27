import os
import paddle
import paddle.nn as nn
import numpy as np
from linear_cca import linear_cca
from paddle.io import BatchSampler, SequenceSampler, RandomSampler
from DeepCCAModels import DeepCCA
from utils import load_data, svm_classify
import time
import logging
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
paddle.set_default_dtype("float64")


class Solver():
    def __init__(self, model, linear_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device='cpu'):
        # self.model = paddle.DataParallel(model)
        self.model = model
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = paddle.optimizer.RMSProp(
            learning_rate=learning_rate,
            parameters=self.model.parameters(),
            weight_decay=reg_par
        )
        self.device = device

        self.linear_cca = linear_cca

        self.outdim_size = outdim_size

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Paddle")
        fh = logging.FileHandler("DCCA.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)

    def fit(self, x1, x2, vx1=None, vx2=None, tx1=None, tx2=None, checkpoint='checkpoint.model'):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # x1.to(self.device)
        # x2.to(self.device)

        data_size = x1.shape[0]

        if vx1 is not None and vx2 is not None:
            best_val_loss = 0
            # vx1.to(self.device)
            # vx2.to(self.device)
        # if tx1 is not None and tx2 is not None:
        #     # tx1.to(self.device)
        #     # tx2.to(self.device)

        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            rand = list(RandomSampler(
                range(data_size)))
            batch_idxs = [rand[i : i + self.batch_size] for i in range(0,len(rand), self.batch_size)]
            num = 1
            for batch_idx in batch_idxs:
                print("-----------------------num--------------------", num)
                num += 1
                self.optimizer.clear_grad()
                x1_tmp = x1.tolist()
                x2_tmp = x2.tolist()
                tmp1 = []
                tmp2 = []
                # print(x1,type(x2),x2_tmp[0])
                for t in batch_idx:
                    tmp1.append(x1_tmp[t])
                    tmp2.append(x2_tmp[t])
                batch_x1 = paddle.to_tensor(tmp1)
                batch_x2 = paddle.to_tensor(tmp2)
                o1, o2 = self.model(batch_x1, batch_x2)
                loss  = self.loss(o1, o2)
                train_losses.append(loss.item())
                self.optimizer.clear_grad()
                print(loss.tolist())
                # paddle.autograd.backward(loss)
                self.optimizer.step()
            train_loss = np.mean(train_losses)

            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx1 is not None and vx2 is not None:
                with paddle.no_grad():
                    self.model.eval()
                    val_loss = self.test(vx1, vx2)
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        paddle.save(self.model.state_dict(), checkpoint)
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_val_loss))
            else:
                paddle.save(self.model.state_dict(), checkpoint)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss))
        paddle.save(self.model.state_dict(), checkpoint)
        # train_linear_cca
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(x1, x2)
            self.train_linear_cca(outputs[0], outputs[1])

        checkpoint_ = paddle.load(checkpoint)
        self.model.set_state_dict(checkpoint_)
        if vx1 is not None and vx2 is not None:
            loss = self.test(vx1, vx2)
            self.logger.info("loss on validation data: {:.4f}".format(loss))

        if tx1 is not None and tx2 is not None:
            loss = self.test(tx1, tx2)
            self.logger.info('loss on test data: {:.4f}'.format(loss))

    def test(self, x1, x2, use_linear_cca=False):
        with paddle.no_grad():
            losses, outputs = self._get_outputs(x1, x2)

            if use_linear_cca:
                print("Linear CCA started!")
                outputs = self.linear_cca.test(outputs[0], outputs[1])
                return np.mean(losses), outputs
            else:
                return np.mean(losses)

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)

    def _get_outputs(self, x1, x2):
        with paddle.no_grad():
            self.model.eval()
            data_size = x1.shape[0]
            batch_idxs = list(BatchSampler(SequenceSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_idx_tmp  = paddle.to_tensor(batch_idx, dtype='int32')
                batch_x1 = paddle.index_select(x=x1, index=batch_idx_tmp)
                batch_x2 = paddle.index_select(x=x2, index=batch_idx_tmp)
                o1, o2 = self.model(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [paddle.concat(outputs1, axis=0).cpu().numpy(),
                   paddle.concat(outputs2, axis=0).cpu().numpy()]
        return losses, outputs


if __name__ == '__main__':
    ############
    # Parameters Section

    device_ids = ["gpu:4", "gpu:5", "gpu:6"]

    for device_id in device_ids:
        paddle.set_device(device_id)
    # print("Using", paddle.cuda.device_count(), "GPUs")

    # the path to save the final learned features
    save_to = './new_features.gz'

    # the size of the new space learned by the model (number of the new features)
    outdim_size = 10

    # size of the input for view 1 and view 2
    input_shape1 = 784
    input_shape2 = 784

    # number of layers with nodes in each one
    layer_sizes1 = [1024, 1024, 1024, outdim_size]
    layer_sizes2 = [1024, 1024, 1024, outdim_size]

    # the parameters for training the network
    learning_rate = 1e-3
    epoch_num = 1
    batch_size = 800

    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5

    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False

    # if a linear CCA should get applied on the learned features extracted from the networks
    # it does not affect the performance on noisy MNIST significantly
    apply_linear_cca = True
    # end of parameters section
    ############

    # Each view is stored in a gzip file separately. They will get downloaded the first time the code gets executed.
    # Datasets get stored under the datasets folder of user's Keras folder
    # normally under [Home Folder]/.keras/datasets/
    data1 = load_data('./noisymnist_view1.gz')
    data2 = load_data('./noisymnist_view2.gz')
    # Building, training, and producing the new features by DCCA
    model = DeepCCA(layer_sizes1, layer_sizes2, input_shape1,
                    input_shape2, outdim_size, use_all_singular_values)
    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()
    solver = Solver(model, l_cca, outdim_size, epoch_num, batch_size,
                    learning_rate, reg_par)
    train1, train2 = data1[0][0], data2[0][0]
    val1, val2 = data1[1][0], data2[1][0]
    test1, test2 = data1[2][0], data2[2][0]

    solver.fit(train1, train2, val1, val2, test1, test2)
    # TODO: Save l_cca model if needed

    set_size = [0, train1.shape[0], train1.shape[
        0] + val1.shape[0], train1.shape[0] + val1.shape[0] + test1.shape[0]]
    loss, outputs = solver.test(paddle.concat([train1, val1, test1], axis=0), paddle.concat(
        [train2, val2, test2], axis=0), apply_linear_cca)
    new_data = []
    # print(outputs)
    for idx in range(3):
        new_data.append([outputs[0][set_size[idx]:set_size[idx + 1], :],
                         outputs[1][set_size[idx]:set_size[idx + 1], :], data1[idx][1]])
    # Training and testing of SVM with linear kernel on the view 1 with new features
    [test_acc, valid_acc] = svm_classify(new_data, C=0.01)
    print("Accuracy on view 1 (validation data) is:", valid_acc * 100.0)
    print("Accuracy on view 1 (test data) is:", test_acc*100.0)
    # Saving new features in a gzip pickled file specified by save_to
    print('saving new features ...')
    f1 = gzip.open(save_to, 'wb')
    thepickle.dump(new_data, f1)
    f1.close()
    d = paddle.load('checkpoint.model')
    solver.model.set_state_dict(d)
    solver.model.parameters()
