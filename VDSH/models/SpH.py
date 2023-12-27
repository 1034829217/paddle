import numpy as np

class SpH:
    def __init__(self, num_bits):
        super(SpH, self).__init__() 
        self.num_bits = int(num_bits)
        self.model_params = None
        
    def eigs(self, X, npca):
        l, pc = np.linalg.eig(X)
        idx = l.argsort()[::-1][:npca]
        return pc[:, idx], l[idx]

    def trainSH(self, X, nbits):
        """
        Input
          X = features matrix [Nsamples, Nfeatures]
          param.nbits = number of bits (nbits do not need to be a multiple of 8)
        """
        [Nsamples, Ndim] = X.shape
        SHparam = {'nbits': nbits}

        # algo:
        # 1) PCA
        npca = min(nbits, Ndim)
        pc, l = self.eigs(np.cov(X.T), npca)
        # pc[:, 3] *= -1
        X = X.dot(pc)   # no need to remove the mean

        # 2) fit uniform distribution
        eps = np.finfo(float).eps
        mn = np.percentile(X, 5)
        mx = np.percentile(X, 95)
        mn = X.min(0) - eps
        mx = X.max(0) + eps

        # 3) enumerate eigenfunctions
        R = mx - mn
        R = R.real
        maxMode = np.ceil((nbits+1) * R / R.max()).astype(int)
        nModes = int(maxMode.sum() - maxMode.size + 1)
        modes = np.ones((nModes, npca))
        m = 0
        for i in range(npca):
            modes[m+1:m+maxMode[i], i] = np.arange(1, maxMode[i]) + 1
            m = m + maxMode[i] - 1
        modes = modes - 1
        omega0 = np.pi / R
        omegas = modes * omega0.reshape(1, -1).repeat(nModes, 0)
        eigVal = -(omegas ** 2).sum(1)
        ii = (-eigVal).argsort()
        modes = modes[ii[1:nbits+1], :]

        SHparam['pc'] = pc
        SHparam['mn'] = mn
        SHparam['mx'] = mx
        SHparam['modes'] = modes
        return SHparam

    def compressSH(self, X, SHparam):
        """
        [B, U] = compressSH(X, SHparam)
        Input
        X = features matrix [Nsamples, Nfeatures]
        SHparam =  parameters (output of trainSH)
        Output
        B = bits (compacted in 8 bits words)
        U = value of eigenfunctions (bits in B correspond to U>0)
        """

        if X.ndim == 1:
            X = X.reshape((1, -1))

        Nsamples, Ndim = X.shape
        nbits = SHparam['nbits']

        X = X.dot(SHparam['pc'])
        X = X - SHparam['mn'].reshape((1, -1))
        omega0 = np.pi / (SHparam['mx'] - SHparam['mn'])
        omegas = SHparam['modes'] * omega0.reshape((1, -1))

        U = np.zeros((Nsamples, nbits))
        for i in range(nbits):
            omegai = omegas[i, :]
            ys = np.sin(X * omegai + np.pi/2)
            yi = np.prod(ys, 1)
            U[:, i] = yi

        b = np.require(U > 0, dtype=np.int)
        #B = compactbit(b)
        return b, U

    def fit_transform(self, train_mat):
        self.model_params = self.trainSH(train_mat.toarray(), self.num_bits)
        cbTrain, _ = self.compressSH(train_mat.toarray(), self.model_params)
        return cbTrain
    
    def transform(self, test_mat):
        cbTest, _ = self.compressSH(test_mat.toarray(), self.model_params)
        return cbTest