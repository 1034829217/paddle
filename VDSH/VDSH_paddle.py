import paddle
# import torch.autograd as autograd
# from torch.autograd import Variable
import numpy
import paddle.nn as nn
import paddle.nn.functional as F

class VDSH(nn.Layer):
    
    def __init__(self, dataset, vocabSize, latentDim, device, dropoutProb=0.):
        super(VDSH, self).__init__()
        
        self.dataset = dataset
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
        self.device = device
        
        self.encoder = nn.Sequential(nn.Linear(self.vocabSize, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Dropout(p=dropoutProb))
        
        self.h_to_mu = nn.Linear(self.hidden_dim, self.latentDim)
        self.h_to_logvar = nn.Sequential(nn.Linear(self.hidden_dim, self.latentDim),
                                         nn.Sigmoid())
        
        self.decoder = nn.Sequential(nn.Linear(self.latentDim, self.vocabSize),
                                     nn.LogSoftmax(axis=1))
        
    def encode(self, doc_mat):
        h = self.encoder(doc_mat)
        z_mu = self.h_to_mu(h)
        z_logvar = self.h_to_logvar(h)
        return z_mu, z_logvar
        
    def reparametrize(self, mu, logvar):
        std = paddle.sqrt(paddle.exp(logvar))
        temp = numpy.round(numpy.random.normal(10, 0.2, size=std.shape),2)
        eps = paddle.to_tensor(temp, dtype='float32')
        # eps = Variable(eps)
        eps.stop_gradient = False
        return eps.multiply(std).add(mu)
    
    def forward(self, document_mat):
        mu, logvar = self.encode(document_mat)
        z = self.reparametrize(mu, logvar)
        prob_w = self.decoder(z)
        return prob_w, mu, logvar
    
    def get_name(self):
        return "VDSH"
    
    @staticmethod
    def calculate_KL_loss(mu, logvar):
        # print("mu>>>>>>>>>>", mu, type(mu), logvar, type(logvar))
        # KLD_element = mu.pow(2).add(logvar.exp()).multiply(-1).add(1).add(logvar)
        KLD_element = mu.pow(2).add(logvar.exp()).multiply(paddle.to_tensor(-1, dtype='float32')).add(paddle.to_tensor(1, dtype='float32')).add(logvar)
        KLD = paddle.sum(KLD_element, axis=1)
        KLD = paddle.mean(KLD).multiply(paddle.to_tensor(-0.5, dtype='float32'))
        return KLD

    @staticmethod
    def compute_reconstr_loss(logprob_word, doc_mat):
        return -paddle.mean(paddle.sum(logprob_word * doc_mat, axis=1))
    
    def get_binary_code(self, train, test):
        train_zy = [(self.encode(paddle.to_tensor(xb.numpy()))[0], paddle.to_tensor(yb.numpy())) for xb, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = paddle.concat(train_z, axis=0)
        train_y = paddle.concat(train_y, axis=0)

        test_zy = [(self.encode(paddle.to_tensor(xb.numpy()))[0], paddle.to_tensor(yb.numpy())) for xb, yb in test]
        test_z, test_y = zip(*test_zy)
        test_z = paddle.concat(test_z, axis=0)
        test_y = paddle.concat(test_y, axis=0)

        mid_val= paddle.median(train_z, axis=0)
        x = paddle.to_tensor([0, 1, 2, 3])
        y = x.astype('uint8')
        train_b = (train_z > mid_val)
        train_b = paddle.to_tensor(train_b, dtype='uint8')
        print("tttt>>>>>", train_b, type(train_b))
        test_b = (test_z > mid_val)
        test_b = paddle.to_tensor(test_b, dtype='uint8')

        del train_z
        del test_z

        return train_b, test_b, train_y, test_y
        