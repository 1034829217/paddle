import paddle
import numpy
import paddle.nn as nn
import paddle.nn.functional as F

class VDSH_SP(nn.Layer):
    
    def __init__(self, dataset, vocabSize, latentDim, num_classes, device, dropoutProb=0., use_softmax=True):
        super(VDSH_SP, self).__init__()
        
        self.dataset = dataset
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.num_classes = num_classes
        self.dropoutProb = dropoutProb
        self.device = device
        self.use_softmax = use_softmax
        
        self.enc_z = nn.Sequential(nn.Linear(self.vocabSize, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Dropout(p=dropoutProb))
        
        self.h_to_z_mu = nn.Linear(self.hidden_dim, self.latentDim)
        self.h_to_z_logvar = nn.Sequential(nn.Linear(self.hidden_dim, self.latentDim),
                                         nn.Sigmoid())
        
        self.enc_v = nn.Sequential(nn.Linear(self.vocabSize, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Dropout(p=dropoutProb))
        
        self.h_to_v_mu = nn.Linear(self.hidden_dim, self.latentDim)
        self.h_to_v_logvar = nn.Sequential(nn.Linear(self.hidden_dim, self.latentDim),
                                         nn.Sigmoid())
        
        self.decoder = nn.Sequential(nn.Linear(self.latentDim, self.vocabSize),
                                     nn.LogSoftmax(axis=1))
        
        if use_softmax:
            self.pred = nn.Sequential(nn.Linear(self.latentDim, self.num_classes))
            self.pred_loss = nn.CrossEntropyLoss() # combine log_softmax and NLLloss
        else:
            self.pred = nn.Sequential(nn.Linear(self.latentDim, self.num_classes),
                                      nn.Sigmoid())
        
    def encode(self, doc_mat):
        h1 = self.enc_z(doc_mat)
        z_mu = self.h_to_z_mu(h1)
        z_logvar = self.h_to_z_logvar(h1)
        
        h2 = self.enc_v(doc_mat)
        v_mu = self.h_to_v_mu(h2)
        v_logvar = self.h_to_v_logvar(h2)
        
        return z_mu, z_logvar, v_mu, v_logvar
        
    def reparametrize(self, mu, logvar):
        std = paddle.sqrt(paddle.exp(logvar))
        temp = numpy.round(numpy.random.normal(10, 0.2, size=std.shape),2)
        eps = paddle.to_tensor(temp, dtype='float32')
        # eps = Variable(eps)
        eps.stop_gradient = False
        return eps.multiply(std).add(mu)
    
    def forward(self, doc_mat):
        z_mu, z_logvar, v_mu, v_logvar = self.encode(doc_mat)
        z = self.reparametrize(z_mu, z_logvar)
        v = self.reparametrize(v_mu, v_logvar)
        logprob_w = self.decoder(z+v)
        score_c = self.pred(z)
        print("DDDDDDDDDevice", self.device)
        return logprob_w, score_c, z_mu, z_logvar, v_mu, v_logvar
    
    def get_name(self):
        return "VDSH_SP"
    
    @staticmethod
    def calculate_KL_loss(mu, logvar):
        KLD_element = mu.pow(2).add(logvar.exp()).multiply(paddle.to_tensor(-1, dtype='float32')).add(paddle.to_tensor(1, dtype='float32')).add(logvar)
        KLD = paddle.sum(KLD_element, axis=1)
        KLD = paddle.mean(KLD).multiply(paddle.to_tensor(-0.5, dtype='float32'))
        return KLD

    @staticmethod
    def compute_reconstr_loss(logprob_word, doc_mat):
        return -paddle.mean(paddle.sum(logprob_word * doc_mat, axis=1))
    
    def compute_prediction_loss(self, scores, labels):
        if self.use_softmax:
            return self.pred_loss(scores, labels)
        else:
            # compute L2 distance
            return paddle.mean(paddle.sum((scores - labels)**2., axis=1))
        
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
        test_b = (test_z > mid_val)
        test_b = paddle.to_tensor(test_b, dtype='uint8')

        del train_z
        del test_z

        return train_b, test_b, train_y, test_y
        