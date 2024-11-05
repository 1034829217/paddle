# DCPH based on paddlepaddle approach

This paper propose a novel cross-modal hashing method without defining the similarity between datapoints, called Deep Cross-modal Proxy Hashing (DCPH). Specifically, DCPH first trains a proxy hashing network to transform each category information of a dataset into a semantic discriminative hash code, called proxy hash code. Each proxy hash code can preserve the semantic information of its corresponding category well. Next, without defining the similarity between datapoints to supervise the training process of the modality-specific hashing networks, we propose a novel margin-dynamic-softmax loss to directly utilize the proxy hashing codes as supervised information. Finally, by minimizing the novel margin-dynamic-softmax loss, the modality-specific hashing networks can be trained to generate hash codes that can simultaneously preserve the cross-modal similarity and abundant semantic information well

## Training

```
python CCMH_BS.py
