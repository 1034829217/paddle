import paddle
import numpy as np


class cca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device
        # print(device)

    def loss(self, H1, H2):
        """

        It is the loss function of CCA as introduced in the original paper. There can be other formulations.

        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0

        o1 = H1.shape[0]
        o2 = H1.shape[0]

        m = H1.shape[1]
#         print(H1.size())

        H1bar = H1 - H1.mean(axis=1).unsqueeze(axis=1)
        H2bar = H2 - H2.mean(axis=1).unsqueeze(axis=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0

        SigmaHat12 = (1.0 / (m - 1)) * paddle.matmul(H1bar, paddle.tensor.transpose(H2bar, perm=[1, 0]))
        SigmaHat11 = (1.0 / (m - 1)) * paddle.matmul(H1bar,
                                                    paddle.tensor.transpose(H1bar, perm=[1, 0])) + r1 * paddle.eye(o1)
        SigmaHat22 = (1.0 / (m - 1)) * paddle.matmul(H2bar,
                                                    paddle.tensor.transpose(H2bar, perm=[1, 0])) + r2 * paddle.eye(o2)
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0

        
        # Calculating the root inverse of covariance matrices by using eigen decomposition
        SigmaHat11 = SigmaHat11.astype('float32')
        SigmaHat22 = SigmaHat22.astype('float32')
        SigmaHat11_cpu = SigmaHat11.cpu()
        SigmaHat22_cpu = SigmaHat22.cpu()
        SigmaHat11_cpu_np = SigmaHat11_cpu.numpy()
        SigmaHat22_cpu_np = SigmaHat22_cpu.numpy()
        D1_np, V1_np = np.linalg.eig(SigmaHat11_cpu_np)
        D2_np, V2_np = np.linalg.eig(SigmaHat22_cpu_np)
        D1 = paddle.to_tensor(D1_np)
        V1 = paddle.to_tensor(V1_np)
        D2 = paddle.to_tensor(D2_np)
        V2 = paddle.to_tensor(V2_np)
        # D1, V1 = paddle.linalg.eig(SigmaHat11_cpu)
        # D2, V2 = paddle.linalg.eig(SigmaHat22_cpu)
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0
        print("-----------Tval---------",D1.shape, V1.shape)
        # Added to increase stability
        posInd1 = paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int32')
        # D1 = paddle.as_real(D1)
        # D1 = D1[:, 0]
        # V1 = paddle.as_real(V1)
        # V1 = V1[:, :, 0]
        D1 = D1[posInd1]
        V1 = paddle.index_select(x=V1, index=posInd1)
        posInd2 = paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # D2 = paddle.as_real(D2)
        # D2 = D2[:, 0]
        # V2 = paddle.as_real(V2)
        # V2 = V2[:, :, 0]
        D2 = D2[posInd2]
        V2 = paddle.index_select(x=V2, index=posInd2)
        
        SigmaHat11RootInv = paddle.matmul(
            paddle.matmul(V1, paddle.diag(D1 ** -0.5)), paddle.tensor.transpose(V1, perm=[0, 1]))
        SigmaHat22RootInv = paddle.matmul(
            paddle.matmul(V2, paddle.diag(D2 ** -0.5)), paddle.tensor.transpose(V2, perm=[0, 1]))
        SigmaHat11RootInv = SigmaHat11RootInv.astype('float32')
        SigmaHat22RootInv = SigmaHat22RootInv.astype('float32')
        SigmaHat12 = SigmaHat12.astype('float32')
        Tval = paddle.matmul(paddle.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)
#         print(Tval.size())

        
        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = paddle.matmul(Tval.t(), Tval)
            corr = paddle.trace(paddle.sqrt(tmp))
            # assert paddle.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = paddle.matmul(paddle.tensor.transpose(Tval, perm=[0, 1]), Tval)
            trace_TT = paddle.add(trace_TT, (paddle.eye(trace_TT.shape[0]).astype(trace_TT.dtype)*r1)) # regularization for more stability
            trace_TT = trace_TT.astype('float64')
            with paddle.fluid.dygraph.guard(paddle.fluid.CPUPlace()):
                U, V = paddle.linalg.eig(trace_TT) 
            U_real = paddle.real(U)  # Extract real part
            U_imag = paddle.imag(U)
            U_real = paddle.where(U_real > eps, U_real, paddle.ones_like(U_real) * eps)
            # Move complex tensor U to CPU
            U = paddle.to_tensor(U_real) + 1j * paddle.to_tensor(U_imag)
            U_cpu = paddle.to_tensor(U, place=paddle.CPUPlace())

            # Convert complex tensor to real tensor for topk operation
            U_real_cpu = paddle.real(U_cpu)
            U_imag_cpu = paddle.imag(U_cpu)

            # Apply the topk operation on the real part
            topk_result = paddle.topk(U_real_cpu, k=self.outdim_size)

            # Extract the corresponding indices and values
            indices = topk_result[1]  # Indices
            U_real_cpu = topk_result[0]  # Values

            # Extract the corresponding imaginary part
            U_imag_cpu = paddle.gather(U_imag_cpu, indices)

            # Move the results back to GPU if needed
            U_real = paddle.to_tensor(U_real_cpu, place=paddle.CUDAPlace(6))
            U_imag = paddle.to_tensor(U_imag_cpu, place=paddle.CUDAPlace(6))



            
            # Separate the real and imaginary parts
            U_real_sqrt = paddle.sqrt(paddle.real(U))
            U_imag_sqrt = paddle.sqrt(paddle.imag(U))

            # Combine the results back into a complex tensor
            U_sqrt = paddle.to_tensor(U_real_sqrt) + 1j * paddle.to_tensor(U_imag_sqrt)

            # Sum the square roots
            corr = paddle.sum(U_sqrt)
            loss = paddle.multiply(corr, paddle.to_tensor([-1.0], dtype='float32'))

        return loss
