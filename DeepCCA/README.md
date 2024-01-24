# DCCA: Deep Canonical Correlation Analysis

This is an implementation of Deep Canonical Correlation Analysis (DCCA or Deep CCA) in Python with pytorch, which supports for multi-GPU training.

DCCA is a non-linear version of CCA which uses neural networks as the mapping functions instead of linear transformers. DCCA is originally proposed in the following paper:

Galen Andrew, Raman Arora, Jeff Bilmes, Karen Livescu, "[Deep Canonical Correlation Analysis.](http://www.jmlr.org/proceedings/papers/v28/andrew13.pdf)", ICML, 2013.

It uses the latest pytorch1.0-preview. Because the loss function of the network needs to calculate the gradient of eigenvalue decomposition for symmetric matrix. The base modeling network can easily get substituted with a more efficient and powerful network like CNN.

Most of the configuration and parameters are set based on the following paper:

Weiran Wang, Raman Arora, Karen Livescu, and Jeff Bilmes. "[On Deep Multi-View Representation Learning.](http://proceedings.mlr.press/v37/wangb15.pdf)", ICML, 2015.

```

### Dataset
The model is evaluated on a noisy version of MNIST dataset. I use the dataset built by @VahidooX which is exactly like the way it is introduced in the paper. The train/validation/test split is the original split of MNIST.

The dataset was large and could not get uploaded on GitHub. So it is uploaded on another server. You can download them from [noisymnist_view1.gz](https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz) and [noisymnist_view2.gz](https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz), or use the download_data.sh. (Thanks to @VahidooX)


