import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Linear:
        print("type<<<<<<<<<<<<<", type(m.weight), type(m), m)
        torch.nn.init.xavier_uniform_(m.weight)
        print("type>>>>>>>>>>>>>", type(m))
        m.bias.data.fill_(0.01)
                
class VDSH_S(nn.Module):
    
    def __init__(self, dataset, vocabSize, latentDim, num_classes, device, dropoutProb=0., use_softmax=True):
        super(VDSH_S, self).__init__()
        
        self.dataset = dataset
        self.hidden_dim = 1000
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.num_classes = num_classes
        self.dropoutProb = dropoutProb
        self.device = device
        self.use_softmax = use_softmax
        
        self.encoder = nn.Sequential(nn.Linear(self.vocabSize, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=dropoutProb))
        
        self.h_to_mu = nn.Linear(self.hidden_dim, self.latentDim)
        self.h_to_logvar = nn.Sequential(nn.Linear(self.hidden_dim, self.latentDim),
                                         nn.Sigmoid())
        
        self.decoder = nn.Sequential(nn.Linear(self.latentDim, self.vocabSize),
                                     nn.LogSoftmax(dim=1))
        
        if use_softmax:
            self.pred = nn.Sequential(nn.Linear(self.latentDim, self.num_classes))
            self.pred_loss = nn.CrossEntropyLoss() # combine log_softmax and NLLloss
        else:
            self.pred = nn.Sequential(nn.Linear(self.latentDim, self.num_classes),
                                      nn.Sigmoid())
        
        # init the weights
        self.encoder.apply(init_weights)
        init_weights(self.h_to_mu)
        init_weights(self.h_to_logvar)
        
        self.decoder.apply(init_weights)
        self.pred.apply(init_weights)
        
    def encode(self, doc_mat):
        h = self.encoder(doc_mat)
        z_mu = self.h_to_mu(h)
        z_logvar = self.h_to_logvar(h)
        return z_mu, z_logvar
        
    def reparametrize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, document_mat):
        mu, logvar = self.encode(document_mat)
        z = self.reparametrize(mu, logvar)
        logprob_w = self.decoder(z)
        score_c = self.pred(z)
        return logprob_w, score_c, mu, logvar
    
    def get_name(self):
        return "VDSH_S"
    
    @staticmethod
    def calculate_KL_loss(mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element, dim=1)
        KLD = torch.mean(KLD).mul_(-0.5)
        return KLD

    @staticmethod
    def compute_reconstr_loss(logprob_word, doc_mat):
        return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))
    
    def compute_prediction_loss(self, scores, labels):
        if self.use_softmax:
            return self.pred_loss(scores, labels)
        else:
            # compute L2 distance
            return torch.mean(torch.sum((scores - labels)**2., dim=1))
        
    def get_binary_code(self, train, test):
        train_zy = [(self.encode(xb.to(self.device))[0], yb) for xb, yb in train]
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = [(self.encode(xb.to(self.device))[0], yb) for xb, yb in test]
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)

        mid_val, _ = torch.median(train_z, dim=0)
        train_b = (train_z > mid_val).type(torch.cuda.ByteTensor)
        test_b = (test_z > mid_val).type(torch.cuda.ByteTensor)

        del train_z
        del test_z

        return train_b, test_b, train_y, test_y
        